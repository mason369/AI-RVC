"""Validate the actual default cover route and every delivered audio file."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer.cover_pipeline import CoverPipeline
from tools.verify_feature_contracts import audio_evidence
from tools.default_quality_audit import build_default_cover_kwargs


def main():
    parser = argparse.ArgumentParser(description="真实默认翻唱链路验收")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/config.json").read_text(encoding="utf-8"))
    model_dir = ROOT / "assets/weights/characters/dia_v2"
    case = {"name": "default_cover", "role_id": "Dia2", "tags": ["runtime_acceptance"], "input_audio": str(args.input),
            "model_path": str(model_dir / "Dia2.pth"), "index_path": str(model_dir / "dia2.index")}
    kwargs = build_default_cover_kwargs(case, config, args.output)
    started = time.perf_counter()
    pipeline = CoverPipeline(device=config["device"])
    result = pipeline.process(**kwargs)
    evidence = {key: audio_evidence(Path(path)) for key, path in result.items()
                if isinstance(path, str) and Path(path).suffix.lower() == ".wav"}
    if len(evidence) < 7 or any(item["subtype"] != "FLOAT" for item in evidence.values()):
        raise RuntimeError(f"默认链路输出缺失或没有保留 Float WAV：{evidence}")
    report = {"seconds": time.perf_counter() - started, "parameters": kwargs, "outputs": evidence,
              "session": result.get("all_files_dir")}
    (args.output / "cover_runtime_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
