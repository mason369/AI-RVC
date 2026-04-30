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


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
    return mono.astype(np.float32), int(sr)


def _rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or b.size < 8:
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

    aligned_len = min(input_audio.size, lead_audio.size, backing_audio.size)
    if aligned_len <= 0:
        raise ValueError("Karaoke scoring received empty audio.")

    input_audio = input_audio[:aligned_len]
    lead_audio = lead_audio[:aligned_len]
    backing_audio = backing_audio[:aligned_len]

    input_rms = _rms(input_audio)
    lead_rms = _rms(lead_audio)
    backing_rms = _rms(backing_audio)
    reconstruction = lead_audio + backing_audio
    reconstruction_error = _rms(input_audio - reconstruction) / (input_rms + 1e-12)
    backing_ratio = backing_rms / (input_rms + 1e-12)
    lead_ratio = lead_rms / (input_rms + 1e-12)
    lead_backing_abs_corr = _corr_abs(lead_audio, backing_audio)

    backing_target = 0.24
    backing_balance_penalty = abs(np.log2(max(backing_ratio, 1e-4) / backing_target))
    lead_body_penalty = max(0.0, 0.70 - lead_ratio) + max(0.0, lead_ratio - 1.15)
    score = float(
        100.0
        - 42.0 * reconstruction_error
        - 26.0 * lead_backing_abs_corr
        - 8.0 * backing_balance_penalty
        - 12.0 * lead_body_penalty
    )

    return {
        "score": score,
        "input_rms": input_rms,
        "lead_rms": lead_rms,
        "backing_rms": backing_rms,
        "lead_ratio": float(lead_ratio),
        "backing_ratio": float(backing_ratio),
        "reconstruction_error": float(reconstruction_error),
        "lead_backing_abs_corr": lead_backing_abs_corr,
    }


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
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vocals_path = Path(args.vocals_path)
    if not vocals_path.is_absolute():
        vocals_path = (REPO_ROOT / vocals_path).resolve()
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals path not found: {vocals_path}")

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
            results.append(
                {
                    "model": model_name,
                    "lead_path": str(lead_path),
                    "backing_path": str(backing_path),
                    "metrics": metrics,
                }
            )
            print(f"{model_name}: score={metrics['score']:.3f}, backing_ratio={metrics['backing_ratio']:.3f}")
        finally:
            separator.unload_model()

    results.sort(key=lambda item: float(item["metrics"]["score"]), reverse=True)
    summary = {
        "vocals_path": str(vocals_path),
        "output_dir": str(output_dir),
        "ranking": [
            {
                "rank": index + 1,
                "model": item["model"],
                **item["metrics"],
            }
            for index, item in enumerate(results)
        ],
        "results": results,
        "score_note": (
            "Proxy score, not SDR. Higher is better; it rewards input reconstruction, "
            "low lead/backing correlation, and plausible backing level."
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
        "| rank | model | score | recon_err | corr | lead_ratio | backing_ratio |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(results, start=1):
        metrics = item["metrics"]
        markdown_lines.append(
            f"| {index} | `{item['model']}` | {metrics['score']:.3f} | "
            f"{metrics['reconstruction_error']:.4f} | {metrics['lead_backing_abs_corr']:.4f} | "
            f"{metrics['lead_ratio']:.3f} | {metrics['backing_ratio']:.3f} |"
        )
    markdown_path = output_dir / "karaoke_model_report.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print(f"Summary written to: {summary_path}")
    print(f"Report written to: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
