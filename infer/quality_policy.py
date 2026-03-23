from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FALLBACK_HYBRID_MODES = {
    "fallback",
    "smart",
    "rmvpe+fallback",
    "rmvpe_fallback",
    "rmvpe-fallback",
    "hybrid_fallback",
    "hybrid-fallback",
}


@dataclass(frozen=True)
class F0RoutingPolicy:
    requested_method: str
    vc_method: str
    hybrid_mode: str
    gate_method: str
    description: str


def resolve_cover_f0_policy(
    requested_method: str,
    configured_hybrid_mode: str = "off",
    repair_profile: bool = False,
) -> F0RoutingPolicy:
    requested = str(requested_method or "rmvpe").strip().lower()
    configured = str(configured_hybrid_mode or "off").strip().lower()

    if requested != "hybrid":
        return F0RoutingPolicy(
            requested_method=requested,
            vc_method=requested,
            hybrid_mode=configured or "off",
            gate_method=requested,
            description=f"{requested} uses the configured routing directly.",
        )

    hybrid_mode = configured if configured in FALLBACK_HYBRID_MODES else "fallback"
    if repair_profile:
        hybrid_mode = "fallback"

    return F0RoutingPolicy(
        requested_method="hybrid",
        vc_method="rmvpe",
        hybrid_mode=hybrid_mode,
        gate_method="rmvpe",
        description="hybrid request routed to RMVPE with conservative fallback; post-gate uses RMVPE only.",
    )


def build_conservative_crepe_fill_mask(
    f0_rmvpe: np.ndarray,
    f0_crepe: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float,
    max_ratio: float = 0.02,
    max_frames: int = 320,
    context_radius: int = 6,
    energy_mask: np.ndarray | None = None,
) -> np.ndarray:
    f0_rmvpe = np.asarray(f0_rmvpe, dtype=np.float32).reshape(-1)
    f0_crepe = np.asarray(f0_crepe, dtype=np.float32).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    n = min(len(f0_rmvpe), len(f0_crepe), len(confidence))
    if n == 0:
        return np.zeros(0, dtype=bool)

    f0_rmvpe = f0_rmvpe[:n]
    f0_crepe = f0_crepe[:n]
    confidence = confidence[:n]
    threshold = max(0.0, float(confidence_threshold))
    context_radius = max(1, int(context_radius))
    max_ratio = max(0.0, float(max_ratio))
    max_frames = max(0, int(max_frames))

    if energy_mask is None:
        energy_mask = np.ones(n, dtype=bool)
    else:
        energy_mask = np.asarray(energy_mask, dtype=bool).reshape(-1)
        if len(energy_mask) < n:
            energy_mask = np.pad(energy_mask, (0, n - len(energy_mask)), mode="edge")
        else:
            energy_mask = energy_mask[:n]

    voiced_seed = f0_rmvpe > 0
    if not np.any(voiced_seed):
        return np.zeros(n, dtype=bool)

    idx = np.arange(n)
    left_seen = np.where(voiced_seed, idx, -10**9)
    left_seen = np.maximum.accumulate(left_seen)
    right_seen = np.where(voiced_seed, idx, 10**9)
    right_seen = np.minimum.accumulate(right_seen[::-1])[::-1]
    voiced_context = ((idx - left_seen) <= context_radius) & ((right_seen - idx) <= context_radius)

    fill_mask = (
        (f0_rmvpe <= 0)
        & (f0_crepe > 0)
        & (confidence >= threshold)
        & energy_mask
        & voiced_context
    )

    fill_count = int(np.sum(fill_mask))
    if fill_count == 0:
        return fill_mask

    fill_ratio = float(fill_count) / float(n)
    if fill_count > max_frames or fill_ratio > max_ratio:
        return np.zeros(n, dtype=bool)

    return fill_mask


def should_allow_crepe_fallback(
    dropout_mask: np.ndarray,
    total_frames: int,
    max_ratio: float,
    max_frames: int,
    fragmented_max_run: int = 12,
    relaxed_ratio: float = 0.02,
    relaxed_frames: int = 512,
    min_fragmented_runs: int = 6,
) -> bool:
    dropout_mask = np.asarray(dropout_mask, dtype=bool).reshape(-1)
    total_frames = int(max(total_frames, 0))
    if total_frames <= 0:
        total_frames = int(dropout_mask.size)
    if dropout_mask.size == 0 or total_frames <= 0:
        return False

    if dropout_mask.size < total_frames:
        dropout_mask = np.pad(dropout_mask, (0, total_frames - dropout_mask.size), mode="constant")
    else:
        dropout_mask = dropout_mask[:total_frames]

    dropout_count = int(np.sum(dropout_mask))
    if dropout_count <= 0:
        return False

    dropout_ratio = float(dropout_count) / float(total_frames)
    if dropout_count <= int(max_frames) and dropout_ratio <= float(max_ratio):
        return True

    fragmented_max_run = max(1, int(fragmented_max_run))
    relaxed_ratio = max(float(max_ratio), float(relaxed_ratio))
    relaxed_frames = max(int(max_frames), int(relaxed_frames))
    min_fragmented_runs = max(1, int(min_fragmented_runs))

    padded = np.pad(dropout_mask.astype(np.int8), (1, 1), mode="constant")
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return False

    run_lengths = ends - starts
    max_run = int(np.max(run_lengths))
    run_count = int(run_lengths.size)
    if max_run > fragmented_max_run:
        return False
    if run_count < min_fragmented_runs:
        return False
    if dropout_count > relaxed_frames or dropout_ratio > relaxed_ratio:
        return False
    return True


def compute_active_source_replace(
    activity: np.ndarray,
    soft_mask: np.ndarray,
    echo_ratio: np.ndarray,
    direct_ratio: np.ndarray,
    max_replace: float = 0.70,
) -> np.ndarray:
    activity = np.asarray(activity, dtype=np.float32).reshape(-1)
    direct_ratio = np.asarray(direct_ratio, dtype=np.float32).reshape(-1)
    soft_mask = np.asarray(soft_mask, dtype=np.float32)
    echo_ratio = np.asarray(echo_ratio, dtype=np.float32)

    if soft_mask.ndim == 1:
        soft_mask = soft_mask[np.newaxis, :]
    if echo_ratio.ndim == 1:
        echo_ratio = echo_ratio[np.newaxis, :]

    frame_count = min(soft_mask.shape[-1], echo_ratio.shape[-1], len(activity), len(direct_ratio))
    if frame_count <= 0:
        return np.zeros_like(soft_mask, dtype=np.float32)

    soft_mask = soft_mask[..., :frame_count]
    echo_ratio = echo_ratio[..., :frame_count]
    activity = activity[:frame_count][np.newaxis, :]
    direct_ratio = direct_ratio[:frame_count][np.newaxis, :]

    base_replace = 0.85 * (1.0 - activity) * (1.0 - soft_mask)
    active_echo_pressure = np.clip(echo_ratio * (1.0 - direct_ratio), 0.0, 1.0)
    active_replace = 0.40 * activity * active_echo_pressure * (1.0 - soft_mask)
    source_replace = np.clip(base_replace + active_replace, 0.0, float(max_replace))
    return source_replace.astype(np.float32)


def compute_source_cleanup_budget(
    energy_guard: np.ndarray,
    phrase_activity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    energy_guard = np.clip(np.asarray(energy_guard, dtype=np.float32), 0.0, 1.0)
    phrase_activity = np.clip(np.asarray(phrase_activity, dtype=np.float32), 0.0, 1.0)
    allowed_boost = 0.35 + 1.00 * energy_guard
    cleanup_floor = 0.62 + 0.16 * phrase_activity
    return allowed_boost.astype(np.float32), cleanup_floor.astype(np.float32)


def compute_breath_preserving_energy_gates(
    energy_db: np.ndarray,
    ref_db: float,
    unvoiced_mask: np.ndarray | None,
    quiet_floor: float = 0.05,
    breath_floor: float = 0.28,
    breath_active_margin_db: float = 52.0,
    transition_width_db: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    energy_db = np.asarray(energy_db, dtype=np.float32).reshape(-1)
    if energy_db.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty

    quiet_floor = float(np.clip(quiet_floor, 0.0, 1.0))
    breath_floor = float(np.clip(max(breath_floor, quiet_floor), quiet_floor, 1.0))
    breath_active_margin_db = float(max(1.0, breath_active_margin_db))
    transition_width_db = float(max(0.5, transition_width_db))
    ref_db = float(ref_db)

    silence_center = ref_db - 45.0
    slope = transition_width_db / 4.0
    base_gate = 1.0 / (1.0 + np.exp(-((energy_db - silence_center) / slope)))
    base_gate = np.clip(base_gate, quiet_floor, 1.0).astype(np.float32)

    if unvoiced_mask is None:
        return base_gate, base_gate.copy()

    unvoiced_mask = np.asarray(unvoiced_mask, dtype=bool).reshape(-1)
    if len(unvoiced_mask) < len(base_gate):
        unvoiced_mask = np.pad(unvoiced_mask, (0, len(base_gate) - len(unvoiced_mask)), mode="edge")
    else:
        unvoiced_mask = unvoiced_mask[: len(base_gate)]

    breath_activity = np.clip(
        (energy_db - (ref_db - breath_active_margin_db)) / 10.0,
        0.0,
        1.0,
    ).astype(np.float32)
    feature_floor = quiet_floor + (breath_floor - quiet_floor) * breath_activity

    feature_gate = base_gate.copy()
    feature_gate[unvoiced_mask] = np.maximum(feature_gate[unvoiced_mask], feature_floor[unvoiced_mask])
    feature_gate = np.clip(feature_gate, quiet_floor, 1.0).astype(np.float32)
    return feature_gate, base_gate.copy()
