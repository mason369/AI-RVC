# -*- coding: utf-8 -*-
"""Run real AI cover processing modes and write a JSON report."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CASES: tuple[dict[str, Any], ...] = (
    dict(name="current_roformer_auto_source_auto", separator="roformer", vc_preprocess_mode="auto", source_constraint_mode="auto", vc_pipeline_mode="current", karaoke_separation=False),
    dict(name="current_roformer_uvr_deecho_source_on", separator="roformer", vc_preprocess_mode="uvr_deecho", source_constraint_mode="on", vc_pipeline_mode="current", karaoke_separation=False),
    dict(name="current_roformer_karaoke_source_off", separator="roformer", vc_preprocess_mode="auto", source_constraint_mode="off", vc_pipeline_mode="current", karaoke_separation=True),
    dict(name="current_demucs_auto_source_auto", separator="demucs", vc_preprocess_mode="auto", source_constraint_mode="auto", vc_pipeline_mode="current", karaoke_separation=False),
    dict(name="current_uvr5_auto_source_auto", separator="uvr5", vc_preprocess_mode="auto", source_constraint_mode="auto", vc_pipeline_mode="current", karaoke_separation=False),
    dict(name="official_uvr5_one_to_one", separator="uvr5", vc_preprocess_mode="auto", source_constraint_mode="auto", vc_pipeline_mode="official", karaoke_separation=False),
)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if resolved.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty: {resolved}")
    return resolved


def _default_index(model_path: Path) -> Optional[Path]:
    direct = model_path.with_suffix(".index")
    if direct.exists():
        return direct
    indexes = sorted(model_path.parent.glob("*.index"))
    return indexes[0] if len(indexes) == 1 else None


def _device(require_cuda: bool) -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    return "cpu"


def _validate_outputs(result: Dict[str, str], keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        value = result.get(key)
        if not value:
            raise RuntimeError(f"Pipeline result missing key: {key}")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline output for {key} does not exist: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Pipeline output for {key} is empty: {path}")
        files[key] = {"path": str(path), "bytes": size}
    return files


def _run_case(case: Dict[str, Any], input_audio: Path, model_path: Path, index_path: Optional[Path], output_root: Path, device: str) -> Dict[str, Any]:
    from infer.cover_pipeline import CoverPipeline

    started = time.time()
    case_output = output_root / case["name"]
    case_output.mkdir(parents=True, exist_ok=True)
    pipeline = CoverPipeline(device=device)
    try:
        result = pipeline.process(
            input_audio=str(input_audio),
            model_path=str(model_path),
            index_path=str(index_path) if index_path else None,
            pitch_shift=0,
            index_ratio=0.5,
            filter_radius=3,
            rms_mix_rate=0.0,
            protect=0.33,
            speaker_id=0,
            f0_method="rmvpe",
            demucs_model="htdemucs_ft",
            demucs_shifts=1,
            demucs_overlap=0.25,
            demucs_split=True,
            roformer_model="hybrid:leap_xe90_vocals+polarformer62_instrumental",
            separator=case["separator"],
            uvr5_model="HP2_all_vocals",
            uvr5_agg=10,
            uvr5_format="wav",
            use_official=True,
            hubert_layer=12,
            vocals_volume=1.0,
            accompaniment_volume=1.0,
            reverb_amount=0.0,
            backing_mix=0.0,
            karaoke_separation=bool(case["karaoke_separation"]),
            karaoke_model="ensemble:mvsep_9205_avg",
            karaoke_merge_backing_into_accompaniment=True,
            vc_preprocess_mode=case["vc_preprocess_mode"],
            source_constraint_mode=case["source_constraint_mode"],
            vc_pipeline_mode=case["vc_pipeline_mode"],
            output_dir=str(case_output),
            model_display_name="matrix",
        )
        keys = ["cover", "vocals", "converted_vocals", "accompaniment", "all_files_dir"]
        if case["karaoke_separation"]:
            keys.extend(
                [
                    "lead_vocals",
                    "backing_vocals",
                    "accompaniment_without_harmony",
                ]
            )
        return {
            "name": case["name"],
            "status": "pass",
            "seconds": round(time.time() - started, 3),
            "parameters": case,
            "result": result,
            "files": _validate_outputs(result, keys),
        }
    except Exception as exc:
        return {
            "name": case["name"],
            "status": "fail",
            "seconds": round(time.time() - started, 3),
            "parameters": case,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if pipeline.separator is not None:
            pipeline.separator.unload_model()
        if pipeline.karaoke_separator is not None:
            pipeline.karaoke_separator.unload_model()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-path", default=str(REPO_ROOT / "assets" / "weights" / "characters" / "rin" / "Rin.pth"))
    parser.add_argument("--index-path")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "mode_matrix"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--case", action="append", choices=[case["name"] for case in CASES])
    args = parser.parse_args()

    input_audio = _require_file(Path(args.input), "Input audio")
    model_path = _require_file(Path(args.model_path), "RVC model")
    index_path = _require_file(Path(args.index_path), "RVC index") if args.index_path else _default_index(model_path)
    device = _device(args.require_cuda)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir).expanduser().resolve() / f"{platform.system().lower()}_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.case or [])
    cases = [case for case in CASES if not selected or case["name"] in selected]
    report = {
        "platform": {"system": platform.system(), "release": platform.release(), "python": platform.python_version()},
        "device": device,
        "input_audio": str(input_audio),
        "model_path": str(model_path),
        "index_path": str(index_path) if index_path else None,
        "output_root": str(output_root),
        "cases": [],
    }
    report_path = output_root / "mode_matrix_results.json"
    for case in cases:
        print(f"\n=== {case['name']} ===", flush=True)
        record = _run_case(case, input_audio, model_path, index_path, output_root, device)
        print(f"{case['name']}: {record['status']} ({record['seconds']}s)", flush=True)
        if record["status"] == "fail":
            print(record["error"], flush=True)
        report["cases"].append(record)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [case["name"] for case in report["cases"] if case["status"] != "pass"]
    report["summary"] = {"total": len(cases), "passed": len(cases) - len(failed), "failed": len(failed), "failed_cases": failed}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
