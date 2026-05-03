#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate local karaoke lead/backing split candidates with reproducible metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer.separator import KARAOKE_DEFAULT_MODEL, KARAOKE_FALLBACK_MODELS, KaraokeSeparator
from lib.audio_metrics import evaluate_reference_stems


KARAOKE_MIN_LENGTH_COVERAGE = 0.999


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
    return mono.astype(np.float32), int(sr)


def _rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    if n < 8:
        return 0.0
    a = a[:n]
    b = b[:n]
    if float(np.std(a)) <= 1e-8 or float(np.std(b)) <= 1e-8:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def score_karaoke_stems(
    input_vocals_path: Path,
    lead_path: Path,
    backing_path: Path,
) -> Dict[str, float]:
    """Score a karaoke split without ground-truth stems.

    This is a proxy, not SDR: good candidates reconstruct the input when summed,
    keep lead/backing decorrelated, and preserve a plausible backing bed instead
    of collapsing everything into either stem.
    """
    input_audio, input_sr = _load_mono(Path(input_vocals_path))
    lead_audio, lead_sr = _load_mono(Path(lead_path))
    backing_audio, backing_sr = _load_mono(Path(backing_path))
    if lead_sr != input_sr or backing_sr != input_sr:
        raise ValueError("Karaoke scoring expects matching sample rates.")

    input_len = input_audio.size
    aligned_len = min(input_audio.size, lead_audio.size, backing_audio.size)
    if aligned_len <= 0:
        raise ValueError("Karaoke scoring received empty audio.")
    length_coverage = float(aligned_len / max(1, input_len))

    input_audio = input_audio[:aligned_len]
    lead_audio = lead_audio[:aligned_len]
    backing_audio = backing_audio[:aligned_len]

    input_rms = _rms(input_audio)
    lead_rms = _rms(lead_audio)
    backing_rms = _rms(backing_audio)
    reconstruction_error = _rms(input_audio - lead_audio - backing_audio) / (input_rms + 1e-12)
    lead_backing_abs_corr = _abs_corr(lead_audio, backing_audio)
    lead_input_abs_corr = _abs_corr(lead_audio, input_audio)
    lead_ratio = lead_rms / (input_rms + 1e-12)
    backing_ratio = backing_rms / (input_rms + 1e-12)

    backing_target = 0.24
    backing_balance_penalty = abs(np.log2(max(float(backing_ratio), 1e-4) / backing_target))
    lead_body_penalty = max(0.0, 0.70 - float(lead_ratio)) + max(0.0, float(lead_ratio) - 1.15)
    length_penalty = max(0.0, 1.0 - float(length_coverage))
    score = float(
        100.0
        - 46.0 * float(reconstruction_error)
        - 30.0 * float(lead_backing_abs_corr)
        - 8.0 * float(backing_balance_penalty)
        - 12.0 * float(lead_body_penalty)
        - 200.0 * float(length_penalty)
        + 3.0 * float(lead_input_abs_corr)
    )

    return {
        "score": score,
        "input_rms": float(input_rms),
        "lead_rms": float(lead_rms),
        "backing_rms": float(backing_rms),
        "lead_ratio": float(lead_ratio),
        "backing_ratio": float(backing_ratio),
        "length_coverage": float(length_coverage),
        "length_penalty": float(length_penalty),
        "reconstruction_error": float(reconstruction_error),
        "lead_backing_abs_corr": float(lead_backing_abs_corr),
        "lead_input_abs_corr": float(lead_input_abs_corr),
    }


def score_reference_stems(
    reference_lead_path: Path,
    reference_backing_path: Path,
    lead_path: Path,
    backing_path: Path,
) -> Dict[str, object]:
    """Compute true reference-based SI-SDR/SDR when reference stems exist."""
    reference_lead, reference_lead_sr = _load_mono(Path(reference_lead_path))
    reference_backing, reference_backing_sr = _load_mono(Path(reference_backing_path))
    lead_audio, lead_sr = _load_mono(Path(lead_path))
    backing_audio, backing_sr = _load_mono(Path(backing_path))
    if len({reference_lead_sr, reference_backing_sr, lead_sr, backing_sr}) != 1:
        raise ValueError("Reference scoring expects matching sample rates.")

    return evaluate_reference_stems(
        references={"lead": reference_lead, "backing": reference_backing},
        estimates={"lead": lead_audio, "backing": backing_audio},
    )


def _unique(items: Iterable[str]) -> List[str]:
    result: List[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocals-path", required=True, help="Separated vocals.wav to split into lead/backing.")
    parser.add_argument("--output-dir", required=True, help="Directory for candidate stems and reports.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Karaoke model filenames. Defaults to current default plus fallbacks.",
    )
    parser.add_argument("--reference-lead", default=None, help="Optional ground-truth/reference lead stem.")
    parser.add_argument("--reference-backing", default=None, help="Optional ground-truth/reference backing stem.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve_existing_path(path_value: str | None, label: str) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _result_sort_key(item: dict) -> tuple:
    reference_metrics = item.get("reference_metrics")
    if reference_metrics:
        return (1, float(reference_metrics["mean_si_sdr"]))
    metrics = item["metrics"]
    return (
        0,
        float(metrics["length_coverage"]) >= KARAOKE_MIN_LENGTH_COVERAGE,
        float(metrics["score"]),
    )


def main() -> int:
    args = parse_args()
    vocals_path = Path(args.vocals_path)
    if not vocals_path.is_absolute():
        vocals_path = (REPO_ROOT / vocals_path).resolve()
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals path not found: {vocals_path}")

    reference_lead_path = _resolve_existing_path(args.reference_lead, "Reference lead")
    reference_backing_path = _resolve_existing_path(args.reference_backing, "Reference backing")
    has_references = reference_lead_path is not None or reference_backing_path is not None
    if has_references and not (reference_lead_path and reference_backing_path):
        raise ValueError("--reference-lead and --reference-backing must be provided together.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _unique(args.models or [KARAOKE_DEFAULT_MODEL, *KARAOKE_FALLBACK_MODELS])
    results = []
    for model_name in models:
        candidate_dir = output_dir / Path(model_name).stem
        candidate_dir.mkdir(parents=True, exist_ok=True)
        separator = KaraokeSeparator(model_filename=model_name, device=args.device)
        try:
            lead_path, backing_path = separator.separate(str(vocals_path), str(candidate_dir))
            metrics = score_karaoke_stems(vocals_path, Path(lead_path), Path(backing_path))
            reference_metrics = None
            if reference_lead_path and reference_backing_path:
                reference_metrics = score_reference_stems(
                    reference_lead_path=reference_lead_path,
                    reference_backing_path=reference_backing_path,
                    lead_path=Path(lead_path),
                    backing_path=Path(backing_path),
                )
            results.append(
                {
                    "model": model_name,
                    "lead_path": str(lead_path),
                    "backing_path": str(backing_path),
                    "metrics": metrics,
                    "reference_metrics": reference_metrics,
                }
            )
            if reference_metrics:
                print(
                    f"{model_name}: mean_si_sdr={reference_metrics['mean_si_sdr']:.3f}, "
                    f"diagnostic_score={metrics['score']:.3f}"
                )
            else:
                print(f"{model_name}: score={metrics['score']:.3f}, backing_ratio={metrics['backing_ratio']:.3f}")
        finally:
            separator.unload_model()

    results.sort(key=_result_sort_key, reverse=True)
    summary = {
        "vocals_path": str(vocals_path),
        "output_dir": str(output_dir),
        "reference_lead_path": str(reference_lead_path) if reference_lead_path else None,
        "reference_backing_path": str(reference_backing_path) if reference_backing_path else None,
        "ranking": [
            {
                "rank": index + 1,
                "model": item["model"],
                **item["metrics"],
                **(
                    {
                        "reference_mean_si_sdr": item["reference_metrics"]["mean_si_sdr"],
                        "reference_mean_sdr": item["reference_metrics"]["mean_sdr"],
                    }
                    if item.get("reference_metrics")
                    else {}
                ),
            }
            for index, item in enumerate(results)
        ],
        "results": results,
        "score_note": (
            "SI-SDR/SDR are only computed when reference stems are provided. "
            "Without references, score is a diagnostic proxy for reconstruction, "
            "decorrelation, plausible backing level, lead/input coherence, and full-length coverage."
        ),
    }
    summary_path = output_dir / "karaoke_model_report.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Karaoke Model Report",
        "",
        f"- vocals: `{vocals_path}`",
        f"- output: `{output_dir}`",
        "",
    ]
    if has_references:
        markdown_lines.extend(
            [
                f"- reference lead: `{reference_lead_path}`",
                f"- reference backing: `{reference_backing_path}`",
                "",
                "| rank | model | mean_si_sdr | mean_sdr | diag_score | len | recon_err | corr |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        markdown_lines.extend(
            [
                "| rank | model | score | len | recon_err | corr | lead_in_corr | lead_ratio | backing_ratio |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for index, item in enumerate(results, start=1):
        metrics = item["metrics"]
        reference_metrics = item.get("reference_metrics")
        if reference_metrics:
            markdown_lines.append(
                f"| {index} | `{item['model']}` | {reference_metrics['mean_si_sdr']:.3f} | "
                f"{reference_metrics['mean_sdr']:.3f} | {metrics['score']:.3f} | "
                f"{metrics['length_coverage']:.4f} | {metrics['reconstruction_error']:.4f} | "
                f"{metrics['lead_backing_abs_corr']:.4f} |"
            )
        else:
            markdown_lines.append(
                f"| {index} | `{item['model']}` | {metrics['score']:.3f} | "
                f"{metrics['length_coverage']:.4f} | "
                f"{metrics['reconstruction_error']:.4f} | {metrics['lead_backing_abs_corr']:.4f} | "
                f"{metrics['lead_input_abs_corr']:.4f} | "
                f"{metrics['lead_ratio']:.3f} | {metrics['backing_ratio']:.3f} |"
            )
    markdown_path = output_dir / "karaoke_model_report.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(f"Summary written to: {summary_path}")
    print(f"Report written to: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
