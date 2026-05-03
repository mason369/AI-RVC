# -*- coding: utf-8 -*-
"""Reference-based audio metrics for separation/cover evaluation."""
from __future__ import annotations

from typing import Mapping

import numpy as np


EPS = 1e-10


def _as_mono_float(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    return arr.reshape(-1)


def _align_pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = _as_mono_float(reference)
    est = _as_mono_float(estimate)
    n = min(ref.size, est.size)
    if n <= 0:
        raise ValueError("Audio metric received empty reference or estimate.")
    return ref[:n], est[:n]


def _power(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    return float(np.sum(arr * arr))


def _db_ratio(signal_power: float, noise_power: float) -> float:
    signal_power = max(float(signal_power), EPS)
    noise_power = max(float(noise_power), EPS)
    return float(10.0 * np.log10(signal_power / noise_power))


def signal_distortion_ratio(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-dependent SDR: 10 log10(||s||^2 / ||s - shat||^2)."""
    ref, est = _align_pair(reference, estimate)
    return _db_ratio(_power(ref), _power(ref - est))


def scale_invariant_signal_distortion_ratio(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SI-SDR as used by modern source-separation literature."""
    ref, est = _align_pair(reference, estimate)
    ref = ref - float(np.mean(ref))
    est = est - float(np.mean(est))
    ref_power = _power(ref)
    if ref_power <= EPS:
        raise ValueError("SI-SDR reference is silent.")
    scale = float(np.dot(est, ref) / (ref_power + EPS))
    target = scale * ref
    residual = est - target
    return _db_ratio(_power(target), _power(residual))


def signal_to_noise_ratio(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Alias for scale-dependent reconstruction SNR."""
    return signal_distortion_ratio(reference, estimate)


def evaluate_reference_stems(
    references: Mapping[str, np.ndarray],
    estimates: Mapping[str, np.ndarray],
) -> dict:
    """Compute true reference-based metrics for matching stems.

    The caller must provide time-aligned reference stems. Without references,
    SI-SDR/SDR cannot be interpreted as source-separation quality.
    """
    stem_metrics: dict[str, dict[str, float]] = {}
    for stem_name, reference_audio in references.items():
        if stem_name not in estimates:
            raise KeyError(f"Missing estimated stem for reference: {stem_name}")
        estimate_audio = estimates[stem_name]
        stem_metrics[stem_name] = {
            "si_sdr": scale_invariant_signal_distortion_ratio(reference_audio, estimate_audio),
            "sdr": signal_distortion_ratio(reference_audio, estimate_audio),
            "snr": signal_to_noise_ratio(reference_audio, estimate_audio),
        }

    if not stem_metrics:
        raise ValueError("No reference stems were provided.")

    return {
        "mean_si_sdr": float(np.mean([metrics["si_sdr"] for metrics in stem_metrics.values()])),
        "mean_sdr": float(np.mean([metrics["sdr"] for metrics in stem_metrics.values()])),
        "mean_snr": float(np.mean([metrics["snr"] for metrics in stem_metrics.values()])),
        "stems": stem_metrics,
    }
