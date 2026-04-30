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

OFFICIAL_COVER_VC_PROFILE = {
    "separator": "uvr5",
    "karaoke_separation": False,
    "karaoke_merge_backing_into_accompaniment": False,
    "vc_preprocess_mode": "direct",
    "source_constraint_mode": "off",
    "f0_method": "rmvpe",
    "index_rate": 0.75,
    "filter_radius": 3,
    # App/UI value. The upstream official runner receives 1 - this value.
    "rms_mix_rate": 0.75,
    "official_rms_mix_rate": 0.25,
    "protect": 0.33,
    "singing_repair": False,
}


def get_official_cover_vc_profile() -> dict:
    return dict(OFFICIAL_COVER_VC_PROFILE)


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


def build_conservative_harvest_fill_mask(
    reference_f0: np.ndarray,
    fallback_f0: np.ndarray,
    dropout_mask: np.ndarray,
    max_run: int = 10,
    local_radius: int = 4,
    max_semitones: float = 4.0,
    min_neighbors: int = 2,
) -> np.ndarray:
    reference_f0 = np.asarray(reference_f0, dtype=np.float32).reshape(-1)
    fallback_f0 = np.asarray(fallback_f0, dtype=np.float32).reshape(-1)
    dropout_mask = np.asarray(dropout_mask, dtype=bool).reshape(-1)

    n = min(reference_f0.size, fallback_f0.size, dropout_mask.size)
    if n <= 0:
        return np.zeros(0, dtype=bool)

    reference_f0 = reference_f0[:n]
    fallback_f0 = fallback_f0[:n]
    dropout_mask = dropout_mask[:n]
    accepted = np.zeros(n, dtype=bool)

    max_run = max(1, int(max_run))
    local_radius = max(1, int(local_radius))
    max_semitones = max(0.0, float(max_semitones))
    min_neighbors = max(1, int(min_neighbors))

    padded = np.pad(dropout_mask.astype(np.int8), (1, 1), mode="constant")
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    eps = 1e-6

    for start, end in zip(starts, ends):
        run_slice = slice(start, end)
        run_len = end - start
        if run_len <= 0 or run_len > max_run:
            continue

        run_fallback = fallback_f0[run_slice]
        voiced_run = run_fallback > 0
        if not np.any(voiced_run):
            continue

        left = reference_f0[max(0, start - local_radius) : start]
        right = reference_f0[end : min(n, end + local_radius)]
        neighbors = np.concatenate([left[left > 0], right[right > 0]])
        if neighbors.size < min_neighbors:
            continue

        local_median = float(np.median(neighbors))
        if local_median <= 0:
            continue

        semitone_diff = np.abs(
            12.0 * np.log2((run_fallback + eps) / (local_median + eps))
        )
        accepted[run_slice] = voiced_run & (semitone_diff <= max_semitones)

    return accepted


def compute_chunk_crossfade_samples(
    tgt_sr: int,
    t_pad_tgt: int,
    segment_count: int,
) -> int:
    tgt_sr = int(max(tgt_sr, 0))
    t_pad_tgt = int(max(t_pad_tgt, 0))
    segment_count = int(max(segment_count, 0))
    if tgt_sr <= 0 or t_pad_tgt <= 0 or segment_count <= 1:
        return 0

    base = int(round(tgt_sr * 0.018))
    extra = int(round(tgt_sr * 0.002 * max(0, segment_count - 2)))
    min_crossfade = int(round(tgt_sr * 0.012))
    max_crossfade = max(min_crossfade, t_pad_tgt // 3)
    return int(np.clip(base + extra, min_crossfade, max_crossfade))


def compute_active_source_replace(
    activity: np.ndarray,
    soft_mask: np.ndarray,
    echo_ratio: np.ndarray,
    direct_ratio: np.ndarray,
    max_replace: float = 0.82,
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
    active_echo_presence = np.clip(
        echo_ratio * (0.35 + 0.65 * (1.0 - direct_ratio)),
        0.0,
        1.0,
    )
    active_replace = 0.65 * activity * active_echo_presence * (1.0 - soft_mask)
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


def compute_residual_quiet_hf_blend_curve(
    source_rms: np.ndarray,
    converted_rms: np.ndarray,
    source_hf: np.ndarray,
    converted_hf: np.ndarray,
    max_blend: float = 0.68,
) -> np.ndarray:
    """Target residual high-frequency VC artifacts outside strong vocal body.

    The curve is intentionally conservative: active vocal-body frames stay near
    zero even when the converted signal is brighter, while low/mid-energy
    frames with clear HF/RMS excess receive enough blend pressure for cleanup.
    """
    source_rms = np.asarray(source_rms, dtype=np.float32).reshape(-1)
    converted_rms = np.asarray(converted_rms, dtype=np.float32).reshape(-1)
    source_hf = np.asarray(source_hf, dtype=np.float32).reshape(-1)
    converted_hf = np.asarray(converted_hf, dtype=np.float32).reshape(-1)

    n = min(source_rms.size, converted_rms.size, source_hf.size, converted_hf.size)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    source_rms = source_rms[:n]
    converted_rms = converted_rms[:n]
    source_hf = source_hf[:n]
    converted_hf = converted_hf[:n]

    eps = 1e-8
    source_db = 20.0 * np.log10(source_rms + eps)
    ref_db = float(np.percentile(source_db, 95)) if source_db.size else -20.0

    low_mid_pressure = np.clip((ref_db - 10.0 - source_db) / 14.0, 0.0, 1.0)
    deep_gap_guard = 1.0 - np.clip((ref_db - 45.0 - source_db) / 9.0, 0.0, 1.0)
    body_guard = 1.0 - np.clip((source_db - (ref_db - 20.0)) / 8.0, 0.0, 1.0)

    hf_excess = np.clip(
        (converted_hf - 1.08 * source_hf) / (converted_hf + eps),
        0.0,
        1.0,
    )
    rms_excess = np.clip(
        (converted_rms - 1.08 * source_rms) / (converted_rms + eps),
        0.0,
        1.0,
    )
    excess = np.maximum(hf_excess, 0.55 * rms_excess)

    blend = low_mid_pressure * deep_gap_guard * body_guard * excess
    max_blend = float(np.clip(max_blend, 0.0, 1.0))
    return np.clip(max_blend * blend, 0.0, max_blend).astype(np.float32)


def compute_midquiet_transition_hf_blend_curve(
    source_rms: np.ndarray,
    converted_rms: np.ndarray,
    source_hf: np.ndarray,
    converted_hf: np.ndarray,
    max_blend: float = 0.46,
) -> np.ndarray:
    """Target mid-quiet transition fizz while leaving body and deep gaps alone."""
    source_rms = np.asarray(source_rms, dtype=np.float32).reshape(-1)
    converted_rms = np.asarray(converted_rms, dtype=np.float32).reshape(-1)
    source_hf = np.asarray(source_hf, dtype=np.float32).reshape(-1)
    converted_hf = np.asarray(converted_hf, dtype=np.float32).reshape(-1)

    n = min(source_rms.size, converted_rms.size, source_hf.size, converted_hf.size)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    source_rms = source_rms[:n]
    converted_rms = converted_rms[:n]
    source_hf = source_hf[:n]
    converted_hf = converted_hf[:n]

    eps = 1e-8
    source_db = 20.0 * np.log10(source_rms + eps)
    ref_db = float(np.percentile(source_db, 95)) if source_db.size else -20.0

    not_body = np.clip((ref_db - source_db) / 5.0, 0.0, 1.0)
    not_gap = np.clip((source_db - (ref_db - 38.0)) / 11.0, 0.0, 1.0)
    not_too_quiet = 1.0 - np.clip((ref_db - source_db - 13.5) / 4.5, 0.0, 1.0)
    midquiet_focus = not_body * not_gap * not_too_quiet

    hf_excess = np.clip(
        (converted_hf - 1.04 * source_hf) / (converted_hf + eps),
        0.0,
        1.0,
    )
    rms_excess = np.clip(
        (converted_rms - 1.04 * source_rms) / (converted_rms + eps),
        0.0,
        1.0,
    )

    source_delta = np.abs(np.diff(source_rms, prepend=source_rms[:1]))
    converted_delta = np.abs(np.diff(converted_rms, prepend=converted_rms[:1]))
    delta_excess = np.clip(
        (converted_delta - (0.0035 + 1.18 * source_delta)) / (converted_delta + eps),
        0.0,
        1.0,
    )

    pressure = np.maximum(
        0.72 * hf_excess,
        np.maximum(0.42 * rms_excess, 0.58 * delta_excess),
    )
    blend = midquiet_focus * pressure
    max_blend = float(np.clip(max_blend, 0.0, 1.0))
    return np.clip(max_blend * blend, 0.0, max_blend).astype(np.float32)


def _metric_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
    values = _metric_vector(values)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(values.size, mask.size)
    if n <= 0:
        return 0.0
    values = values[:n]
    mask = mask[:n]
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(np.square(values[mask])) + 1e-12))


def _safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = _metric_vector(a)
    b = _metric_vector(b)
    n = min(a.size, b.size)
    if n <= 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)[:n]
        if np.sum(mask) <= 2:
            return 0.0
        a = a[mask]
        b = b[mask]
    if a.size <= 2 or float(np.std(a)) <= 1e-8 or float(np.std(b)) <= 1e-8:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _activity_masks(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = _metric_vector(reference)
    if reference.size <= 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    eps = 1e-8
    ref_db = 20.0 * np.log10(reference + eps)
    ref_peak = float(np.percentile(ref_db, 95))
    active = ref_db >= (ref_peak - 24.0)
    quiet = ref_db <= (ref_peak - 42.0)
    if not np.any(quiet):
        quiet = ref_db <= float(np.percentile(ref_db, 25))
    return active, quiet


def _strong_activity_mask(values: np.ndarray, floor_ratio: float = 0.45) -> np.ndarray:
    values = _metric_vector(values)
    if values.size <= 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.percentile(values, 95))
    threshold = max(float(np.percentile(values, 60)), peak * float(floor_ratio), 1e-8)
    return values >= threshold


def compute_accompaniment_leakage_metrics(
    accompaniment_voiceband_rms: np.ndarray,
    vocal_voiceband_rms: np.ndarray,
) -> dict[str, float]:
    """Estimate vocal-shaped leakage in an instrumental stem.

    This is a no-reference proxy: it does not claim SDR. It flags whether the
    accompaniment voice band follows the separated vocal envelope too closely.
    """
    accompaniment = _metric_vector(accompaniment_voiceband_rms)
    vocal = _metric_vector(vocal_voiceband_rms)
    n = min(accompaniment.size, vocal.size)
    if n <= 0:
        return {
            "active_frame_ratio": 0.0,
            "vocal_activity_correlation": 0.0,
            "active_to_quiet_voiceband_db": 0.0,
            "relative_vocal_band_db": -120.0,
            "leakage_risk_score": 0.0,
        }

    accompaniment = accompaniment[:n]
    vocal = vocal[:n]
    active_mask, quiet_mask = _activity_masks(vocal)
    active_mask = active_mask[:n]
    quiet_mask = quiet_mask[:n]
    eps = 1e-8

    acc_active = _masked_rms(accompaniment, active_mask)
    acc_quiet = _masked_rms(accompaniment, quiet_mask)
    vocal_active = _masked_rms(vocal, active_mask)
    vocal_corr = _safe_corr(accompaniment, vocal, active_mask | quiet_mask)
    active_to_quiet_db = 20.0 * np.log10((acc_active + eps) / (acc_quiet + eps))
    relative_vocal_db = 20.0 * np.log10((acc_active + eps) / (vocal_active + eps))

    corr_component = np.clip((max(0.0, vocal_corr) - 0.35) / 0.55, 0.0, 1.0)
    active_excess_component = np.clip((active_to_quiet_db - 3.0) / 9.0, 0.0, 1.0)
    relative_component = np.clip((relative_vocal_db + 24.0) / 18.0, 0.0, 1.0)
    leakage_risk = np.clip(
        0.45 * corr_component + 0.35 * active_excess_component + 0.20 * relative_component,
        0.0,
        1.0,
    )

    return {
        "active_frame_ratio": float(np.mean(active_mask)) if active_mask.size else 0.0,
        "vocal_activity_correlation": float(vocal_corr),
        "active_to_quiet_voiceband_db": float(active_to_quiet_db),
        "relative_vocal_band_db": float(relative_vocal_db),
        "leakage_risk_score": float(leakage_risk),
    }


def compute_karaoke_stem_separation_metrics(
    lead_rms: np.ndarray,
    backing_rms: np.ndarray,
) -> dict[str, float]:
    """Estimate lead/backing stem duplication and mutual leakage risk."""
    lead = _metric_vector(lead_rms)
    backing = _metric_vector(backing_rms)
    n = min(lead.size, backing.size)
    if n <= 0:
        return {
            "envelope_correlation": 0.0,
            "mutual_active_ratio": 0.0,
            "backing_vs_lead_db_when_lead_dominant": -120.0,
            "lead_vs_backing_db_when_backing_dominant": -120.0,
            "duplication_risk_score": 0.0,
        }

    lead = lead[:n]
    backing = backing[:n]
    eps = 1e-8
    lead_active = _strong_activity_mask(lead)
    backing_active = _strong_activity_mask(backing)
    lead_active = lead_active[:n]
    backing_active = backing_active[:n]
    any_active = lead_active | backing_active
    both_active = lead_active & backing_active

    lead_dominant = lead > (backing * 1.85 + eps)
    backing_dominant = backing > (lead * 1.85 + eps)
    backing_vs_lead_db = 20.0 * np.log10(
        (_masked_rms(backing, lead_dominant) + eps)
        / (_masked_rms(lead, lead_dominant) + eps)
    )
    lead_vs_backing_db = 20.0 * np.log10(
        (_masked_rms(lead, backing_dominant) + eps)
        / (_masked_rms(backing, backing_dominant) + eps)
    )
    envelope_corr = _safe_corr(lead, backing, any_active)
    mutual_active_ratio = float(np.sum(both_active) / max(1, int(np.sum(any_active))))

    corr_component = np.clip((max(0.0, envelope_corr) - 0.45) / 0.50, 0.0, 1.0)
    overlap_component = np.clip((mutual_active_ratio - 0.45) / 0.45, 0.0, 1.0)
    lead_leak_component = np.clip((backing_vs_lead_db + 18.0) / 18.0, 0.0, 1.0)
    backing_leak_component = np.clip((lead_vs_backing_db + 18.0) / 18.0, 0.0, 1.0)
    duplication_risk = np.clip(
        0.42 * corr_component
        + 0.28 * overlap_component
        + 0.18 * lead_leak_component
        + 0.12 * backing_leak_component,
        0.0,
        1.0,
    )

    return {
        "envelope_correlation": float(envelope_corr),
        "mutual_active_ratio": float(mutual_active_ratio),
        "backing_vs_lead_db_when_lead_dominant": float(backing_vs_lead_db),
        "lead_vs_backing_db_when_backing_dominant": float(lead_vs_backing_db),
        "duplication_risk_score": float(duplication_risk),
    }


def compute_mix_fusion_metrics(
    lead_rms: np.ndarray,
    backing_rms: np.ndarray,
    bed_rms: np.ndarray,
) -> dict[str, float]:
    """Estimate final-bed pumping and backing-vocal balance."""
    lead = _metric_vector(lead_rms)
    backing = _metric_vector(backing_rms)
    bed = _metric_vector(bed_rms)
    n = min(lead.size, backing.size, bed.size)
    if n <= 0:
        return {
            "bed_active_vs_quiet_db": 0.0,
            "bed_lead_correlation": 0.0,
            "ducking_risk_score": 0.0,
            "backing_to_lead_db_overlap": -120.0,
            "backing_excess_frame_ratio": 0.0,
            "harmony_presence_ratio": 0.0,
        }

    lead = lead[:n]
    backing = backing[:n]
    bed = bed[:n]
    eps = 1e-8
    lead_active = _strong_activity_mask(lead)
    _, lead_quiet = _activity_masks(lead)
    backing_active = _strong_activity_mask(backing)
    lead_active = lead_active[:n]
    lead_quiet = lead_quiet[:n]
    backing_active = backing_active[:n]
    overlap = lead_active & backing_active

    bed_active = _masked_rms(bed, lead_active)
    bed_quiet = _masked_rms(bed, lead_quiet)
    bed_active_vs_quiet_db = 20.0 * np.log10((bed_active + eps) / (bed_quiet + eps))
    bed_lead_corr = _safe_corr(bed, lead, lead_active | lead_quiet)

    duck_delta_component = np.clip((-1.0 - bed_active_vs_quiet_db) / 5.0, 0.0, 1.0)
    negative_corr_component = np.clip((-0.10 - bed_lead_corr) / 0.55, 0.0, 1.0)
    ducking_risk = np.clip(0.68 * duck_delta_component + 0.32 * negative_corr_component, 0.0, 1.0)

    backing_to_lead_db = 20.0 * np.log10(
        (_masked_rms(backing, overlap) + eps)
        / (_masked_rms(lead, overlap) + eps)
    )
    backing_excess = overlap & (backing >= 0.72 * lead)
    backing_excess_ratio = float(np.sum(backing_excess) / max(1, int(np.sum(overlap))))
    harmony_presence_ratio = float(np.sum(overlap) / max(1, int(np.sum(lead_active))))

    return {
        "bed_active_vs_quiet_db": float(bed_active_vs_quiet_db),
        "bed_lead_correlation": float(bed_lead_corr),
        "ducking_risk_score": float(ducking_risk),
        "backing_to_lead_db_overlap": float(backing_to_lead_db),
        "backing_excess_frame_ratio": float(backing_excess_ratio),
        "harmony_presence_ratio": float(harmony_presence_ratio),
    }


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
