#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a short A/B diagnostic matrix for an existing cover session."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer.cover_pipeline import CoverPipeline
from infer.official_adapter import convert_vocals_official, convert_vocals_official_upstream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run several short VC variants on an existing session and export "
            "per-clip WAVs plus summary.json for side-by-side diagnosis."
        )
    )
    parser.add_argument("--session-dir", required=True, help="Session directory under temp/cover/<id>")
    parser.add_argument(
        "--model-path",
        help=(
            "RVC .pth path. If omitted, try to infer from "
            "outputs/<song>_<model>_all_files_<id>."
        ),
    )
    parser.add_argument("--index-path", help="Optional .index path. If omitted, try best-effort matching.")
    parser.add_argument("--output-dir", help="Where to write diagnostic_matrix outputs.")
    parser.add_argument("--stage", default="vc_final_state", help="Preferred stage in quality_debug.json for suspect times.")
    parser.add_argument("--times", help="Comma-separated times in seconds. If set, skip automatic time selection.")
    parser.add_argument("--max-clips", type=int, default=4, help="Maximum suspect clips to export.")
    parser.add_argument("--window-sec", type=float, default=1.8, help="Window size around each suspect time.")
    parser.add_argument("--gap-sec", type=float, default=0.35, help="Silence gap inserted between concatenated clips.")
    parser.add_argument(
        "--variants",
        default="existing_raw,existing_final,upstream_raw,upstream_post,upstream_index0,official_repair",
        help=(
            "Comma-separated variants. Supported: "
            "existing_raw, existing_final, upstream_raw, upstream_post, "
            "upstream_index0, official_repair"
        ),
    )
    parser.add_argument("--f0-method", default="hybrid")
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.5)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--rms-mix-rate", type=float, default=0.0)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--source-constraint-mode", default="auto")
    return parser.parse_args()


def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _tokenize_name(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(token) >= 2]


def _load_quality_debug(session_dir: Path) -> Dict[str, object]:
    report_path = session_dir / "quality_debug.json"
    if not report_path.exists():
        return {}
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_stage_entry(entries: Sequence[Dict[str, object]], stage: str) -> Optional[Dict[str, object]]:
    for entry in reversed(entries):
        if entry.get("stage") == stage:
            return entry
    return None


def _dedupe_times(times: Iterable[float], min_delta_sec: float = 0.18) -> List[float]:
    ordered = sorted(float(t) for t in times if float(t) >= 0.0)
    deduped: List[float] = []
    for time_sec in ordered:
        if not deduped or abs(time_sec - deduped[-1]) >= min_delta_sec:
            deduped.append(round(time_sec, 3))
    return deduped


def _collect_suspect_times(
    session_dir: Path,
    preferred_stage: str,
    max_clips: int,
) -> Tuple[List[float], Dict[str, object]]:
    payload = _load_quality_debug(session_dir)
    entries = payload.get("stages", [])
    if not isinstance(entries, list):
        entries = []

    stage_order = [
        preferred_stage,
        "vc_final_state",
        "vc_refined",
        "vc_source_constrained",
        "vc_raw",
    ]
    collected: List[float] = []
    for stage in stage_order:
        entry = _get_stage_entry(entries, stage)
        if not entry:
            continue
        analysis = entry.get("analysis", {})
        if not isinstance(analysis, dict):
            continue
        times = analysis.get("transition_spike_times_sec", [])
        if isinstance(times, list):
            collected.extend(float(t) for t in times[:12])
        collected = _dedupe_times(collected)
        if len(collected) >= max_clips:
            break

    return collected[: max(1, int(max_clips))], payload if isinstance(payload, dict) else {}


def _choose_model_file(folder: Path) -> Path:
    candidates = sorted(folder.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth model found in {folder}")
    if len(candidates) == 1:
        return candidates[0]

    folder_norm = _normalize_name(folder.name)
    best_path = candidates[0]
    best_score = -1
    for path in candidates:
        stem_norm = _normalize_name(path.stem)
        score = 0
        if stem_norm == folder_norm:
            score += 1000
        if folder_norm and (folder_norm in stem_norm or stem_norm in folder_norm):
            score += 300
        score += len(set(_tokenize_name(folder.name)) & set(_tokenize_name(path.stem))) * 40
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _resolve_index_for_model(model_path: Path, explicit_index: Optional[str] = None) -> Optional[Path]:
    if explicit_index:
        explicit = Path(explicit_index)
        if explicit.exists():
            return explicit

    candidates = list(model_path.parent.glob("*.index"))
    official_indexes = REPO_ROOT / "assets" / "weights" / "official_indexes"
    if official_indexes.exists():
        candidates.extend(list(official_indexes.glob("*.index")))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    model_norm = _normalize_name(model_path.stem)
    model_tokens = set(_tokenize_name(model_path.stem))
    best_match = None
    best_score = -1
    for candidate in candidates:
        idx_norm = _normalize_name(candidate.stem)
        idx_tokens = set(_tokenize_name(candidate.stem))
        score = 0
        if idx_norm == model_norm:
            score += 1000
        if model_norm and (model_norm in idx_norm or idx_norm in model_norm):
            score += 300
        score += len(model_tokens & idx_tokens) * 40
        if "added" in candidate.stem.lower():
            score += 10
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match if best_score > 0 else None


def _infer_model_from_session(session_dir: Path) -> Tuple[Path, Optional[Path], Optional[str]]:
    session_id = session_dir.name
    outputs_root = REPO_ROOT / "outputs"
    characters_root = REPO_ROOT / "assets" / "weights" / "characters"
    if not outputs_root.exists() or not characters_root.exists():
        raise FileNotFoundError("Cannot infer model because outputs/ or assets/weights/characters/ is missing.")

    output_matches = sorted(outputs_root.glob(f"*_all_files_{session_id}"))
    if not output_matches:
        raise FileNotFoundError("Could not infer model from outputs/. Pass --model-path explicitly.")

    character_dirs = [path for path in characters_root.iterdir() if path.is_dir()]
    best_dir = None
    best_score = -1
    for output_dir in output_matches:
        prefix = output_dir.name[: -len(f"_all_files_{session_id}")]
        for character_dir in character_dirs:
            char_name = character_dir.name
            if prefix == char_name or prefix.endswith(f"_{char_name}"):
                score = len(char_name)
                if score > best_score:
                    best_score = score
                    best_dir = character_dir

    if best_dir is None:
        raise FileNotFoundError(
            "Could not match the session output folder to a character model folder. "
            "Pass --model-path explicitly."
        )

    model_path = _choose_model_file(best_dir)
    index_path = _resolve_index_for_model(model_path)
    return model_path, index_path, best_dir.name


def _extract_bundle(
    source_path: Path,
    output_path: Path,
    times_sec: Sequence[float],
    window_sec: float,
    gap_sec: float,
) -> List[Dict[str, float]]:
    audio, sr = sf.read(str(source_path), always_2d=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[0] <= 0:
        raise ValueError(f"Empty audio: {source_path}")

    duration_sec = float(audio.shape[0] / max(sr, 1))
    bundle_parts: List[np.ndarray] = []
    manifest: List[Dict[str, float]] = []
    cursor_sec = 0.0
    gap_samples = int(round(max(0.0, gap_sec) * sr))
    target_window = max(0.2, float(window_sec))

    for index, time_sec in enumerate(times_sec):
        clip_start_sec = 0.0
        if duration_sec > target_window:
            clip_start_sec = min(max(0.0, float(time_sec) - target_window / 2.0), duration_sec - target_window)
        clip_end_sec = min(duration_sec, clip_start_sec + target_window)
        start_sample = int(round(clip_start_sec * sr))
        end_sample = max(start_sample + 1, int(round(clip_end_sec * sr)))
        clip = audio[start_sample:end_sample]
        if clip.size == 0:
            continue

        actual_duration = float(clip.shape[0] / sr)
        bundle_parts.append(clip.astype(np.float32))
        manifest.append(
            {
                "clip_id": index,
                "source_time_sec": round(float(time_sec), 3),
                "clip_start_sec": round(float(clip_start_sec), 6),
                "clip_end_sec": round(float(clip_end_sec), 6),
                "bundle_start_sec": round(float(cursor_sec), 6),
                "bundle_end_sec": round(float(cursor_sec + actual_duration), 6),
                "duration_sec": round(actual_duration, 6),
            }
        )
        cursor_sec += actual_duration

        if gap_samples > 0 and index < len(times_sec) - 1:
            bundle_parts.append(np.zeros((gap_samples, audio.shape[1]), dtype=np.float32))
            cursor_sec += float(gap_samples / sr)

    if not bundle_parts:
        raise ValueError(f"No clips were extracted from {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_audio = np.concatenate(bundle_parts, axis=0).astype(np.float32)
    sf.write(str(output_path), bundle_audio, sr)
    return manifest


def _split_bundle(
    bundle_path: Path,
    manifest: Sequence[Dict[str, float]],
    output_dir: Path,
    prefix: str,
) -> Dict[str, str]:
    audio, sr = sf.read(str(bundle_path), always_2d=True)
    audio = np.asarray(audio, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, str] = {}

    for segment in manifest:
        bundle_start_sec = float(segment["bundle_start_sec"])
        bundle_end_sec = float(segment["bundle_end_sec"])
        start_sample = max(0, int(round(bundle_start_sec * sr)))
        end_sample = min(audio.shape[0], max(start_sample + 1, int(round(bundle_end_sec * sr))))
        if end_sample <= start_sample:
            continue
        clip = audio[start_sample:end_sample].astype(np.float32)
        clip_key = f"{float(segment['source_time_sec']):07.3f}s"
        output_path = output_dir / f"{prefix}_{clip_key}.wav"
        sf.write(str(output_path), clip, sr)
        exported[clip_key] = str(output_path)

    return exported


def _estimate_artifact_score(analysis: Dict[str, object]) -> float:
    quiet_rms = max(0.0, float(analysis.get("quiet_rms_ratio", 1.0) or 1.0) - 1.0)
    quiet_hf = max(0.0, float(analysis.get("quiet_hf_ratio", 1.0) or 1.0) - 1.0)
    midquiet_hf = max(0.0, float(analysis.get("midquiet_hf_ratio", 1.0) or 1.0) - 1.0)
    spike = max(0.0, float(analysis.get("transition_spike_ratio", 1.0) or 1.0) - 1.0)
    active = max(0.0, float(analysis.get("active_rms_ratio", 1.0) or 1.0) - 1.5)
    breaths = max(0.0, float(analysis.get("synthetic_breath_frames", 0) or 0) - 80.0) / 220.0
    corr_penalty = max(0.0, 0.85 - float(analysis.get("corr_active", 0.0) or 0.0))
    return float(
        0.22 * quiet_rms
        + 0.28 * quiet_hf
        + 0.08 * midquiet_hf
        + 0.22 * spike
        + 0.05 * active
        + 0.10 * breaths
        + 0.05 * corr_penalty
    )


def _aggregate_variant_metrics(analyses: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not analyses:
        return {"clips": 0}

    keys = [
        "active_rms_ratio",
        "quiet_rms_ratio",
        "quiet_hf_ratio",
        "midquiet_hf_ratio",
        "transition_spike_ratio",
        "corr_active",
        "synthetic_breath_frames",
    ]
    aggregate: Dict[str, object] = {"clips": len(analyses)}
    for key in keys:
        values = [float(item.get(key, 0.0) or 0.0) for item in analyses]
        aggregate[f"avg_{key}"] = float(sum(values) / max(len(values), 1))
    aggregate["avg_artifact_score_estimate"] = float(
        sum(_estimate_artifact_score(item) for item in analyses) / max(len(analyses), 1)
    )
    return aggregate


def _preprocess_mode_from_payload(payload: Dict[str, object]) -> str:
    entries = payload.get("stages", [])
    if not isinstance(entries, list):
        return "direct"
    vc_input = _get_stage_entry(entries, "vc_input")
    if not vc_input:
        return "direct"
    mode = vc_input.get("preprocess_mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    extra = vc_input.get("extra", {})
    if isinstance(extra, dict):
        mode = extra.get("effective_preprocess_mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    return "direct"


def _run_upstream_bundle(
    source_bundle: Path,
    output_bundle: Path,
    model_path: Path,
    index_path: Optional[Path],
    args: argparse.Namespace,
    index_rate: Optional[float] = None,
) -> None:
    convert_vocals_official_upstream(
        vocals_path=str(source_bundle),
        output_path=str(output_bundle),
        model_path=str(model_path),
        index_path=str(index_path) if index_path else None,
        f0_method=args.f0_method,
        pitch_shift=args.pitch_shift,
        index_rate=float(args.index_rate if index_rate is None else index_rate),
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        speaker_id=args.speaker_id,
    )


def _run_variant(
    variant: str,
    output_dir: Path,
    manifest: Sequence[Dict[str, float]],
    source_bundle: Path,
    lead_bundle: Optional[Path],
    session_dir: Path,
    model_path: Path,
    index_path: Optional[Path],
    args: argparse.Namespace,
    pipeline: CoverPipeline,
) -> Dict[str, object]:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = variant_dir / "bundle.wav"
    result: Dict[str, object] = {"name": variant, "bundle_path": str(bundle_path)}
    clip_times = [float(item["source_time_sec"]) for item in manifest]

    if variant == "existing_raw":
        existing_path = session_dir / "debug_converted_raw.wav"
        if not existing_path.exists():
            return {"name": variant, "skipped": "missing debug_converted_raw.wav"}
        _extract_bundle(existing_path, bundle_path, clip_times, args.window_sec, args.gap_sec)
    elif variant == "existing_final":
        existing_path = session_dir / "converted_vocals.wav"
        if not existing_path.exists():
            return {"name": variant, "skipped": "missing converted_vocals.wav"}
        _extract_bundle(existing_path, bundle_path, clip_times, args.window_sec, args.gap_sec)
    elif variant == "upstream_raw":
        _run_upstream_bundle(source_bundle, bundle_path, model_path, index_path, args)
    elif variant == "upstream_index0":
        _run_upstream_bundle(source_bundle, bundle_path, model_path, index_path, args, index_rate=0.0)
    elif variant == "upstream_post":
        raw_bundle_path = variant_dir / "bundle_raw.wav"
        _run_upstream_bundle(source_bundle, raw_bundle_path, model_path, index_path, args)
        should_apply = pipeline._should_apply_source_constraint(
            vc_preprocessed=True,
            source_constraint_mode=args.source_constraint_mode,
        )
        if should_apply:
            pipeline._constrain_converted_to_source(
                source_vocals_path=str(source_bundle),
                converted_vocals_path=str(raw_bundle_path),
                original_vocals_path=str(lead_bundle) if lead_bundle else None,
                output_path=str(bundle_path),
            )
            pipeline._refine_source_constrained_output(
                source_vocals_path=str(source_bundle),
                converted_vocals_path=str(bundle_path),
                source_constraint_mode=args.source_constraint_mode,
                f0_method=args.f0_method,
            )
        else:
            bundle_path.write_bytes(raw_bundle_path.read_bytes())
            result["note"] = (
                "postprocess skipped because current preprocess/source-constraint "
                "combination would not apply it"
            )
    elif variant == "official_repair":
        saved_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            convert_vocals_official(
                vocals_path=str(source_bundle),
                output_path=str(bundle_path),
                model_path=str(model_path),
                index_path=str(index_path) if index_path else None,
                f0_method=args.f0_method,
                pitch_shift=args.pitch_shift,
                index_rate=float(args.index_rate),
                filter_radius=args.filter_radius,
                rms_mix_rate=args.rms_mix_rate,
                protect=args.protect,
                speaker_id=args.speaker_id,
                repair_profile=True,
            )
        finally:
            sys.argv = saved_argv
    else:
        return {"name": variant, "skipped": f"unknown variant: {variant}"}

    clips_dir = variant_dir / "clips"
    result["clips"] = _split_bundle(bundle_path, manifest, clips_dir, "candidate")
    return result


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir)
    if not session_dir.is_absolute():
        session_dir = (REPO_ROOT / session_dir).resolve()
    if not session_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {session_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else session_dir / "diagnostic_matrix"
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.times:
        times_sec = _dedupe_times(float(item.strip()) for item in args.times.split(",") if item.strip())
        payload = _load_quality_debug(session_dir)
    else:
        times_sec, payload = _collect_suspect_times(session_dir, args.stage, args.max_clips)

    if not times_sec:
        raise ValueError(
            "No suspect times found. Pass --times explicitly, for example "
            '--times "22.48,23.04,24.17".'
        )

    inferred_model_key = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = (REPO_ROOT / model_path).resolve()
    else:
        model_path, inferred_index_path, inferred_model_key = _infer_model_from_session(session_dir)
        if not args.index_path and inferred_index_path is not None:
            args.index_path = str(inferred_index_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    index_path = _resolve_index_for_model(model_path, args.index_path)
    preprocess_mode = _preprocess_mode_from_payload(payload)

    source_full_path = session_dir / "vocals_for_vc.wav"
    if not source_full_path.exists():
        raise FileNotFoundError(f"Missing source vocals for VC: {source_full_path}")

    lead_full_path = session_dir / "karaoke" / "lead_vocals.wav"
    if not lead_full_path.exists():
        fallback_lead = session_dir / "lead_vocals.wav"
        lead_full_path = fallback_lead if fallback_lead.exists() else source_full_path

    reference_dir = output_dir / "reference"
    source_bundle = reference_dir / "vocals_for_vc_bundle.wav"
    lead_bundle = reference_dir / "lead_vocals_bundle.wav"
    manifest = _extract_bundle(source_full_path, source_bundle, times_sec, args.window_sec, args.gap_sec)
    _extract_bundle(lead_full_path, lead_bundle, times_sec, args.window_sec, args.gap_sec)
    reference_clips = _split_bundle(source_bundle, manifest, reference_dir / "clips", "reference")

    pipeline = CoverPipeline(device="cuda")
    pipeline._last_vc_preprocess_mode = preprocess_mode

    requested_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    variant_results: Dict[str, Dict[str, object]] = {}
    per_clip_results: List[Dict[str, object]] = []
    aggregate_results: Dict[str, Dict[str, object]] = {}

    print(f"Session: {session_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_path}")
    print(f"Index: {index_path if index_path else '(none)'}")
    print(f"Preprocess mode from session: {preprocess_mode}")
    print("Suspect times:", ", ".join(f"{time_sec:.3f}s" for time_sec in times_sec))

    for variant in requested_variants:
        print(f"[diagnose] running {variant} ...")
        variant_result = _run_variant(
            variant=variant,
            output_dir=output_dir,
            manifest=manifest,
            source_bundle=source_bundle,
            lead_bundle=lead_bundle,
            session_dir=session_dir,
            model_path=model_path,
            index_path=index_path,
            args=args,
            pipeline=pipeline,
        )
        variant_results[variant] = variant_result

    for segment in manifest:
        clip_key = f"{float(segment['source_time_sec']):07.3f}s"
        clip_summary: Dict[str, object] = {
            "time_sec": float(segment["source_time_sec"]),
            "reference_clip": reference_clips.get(clip_key),
            "variants": {},
        }
        for variant in requested_variants:
            variant_result = variant_results.get(variant, {})
            variant_clips = variant_result.get("clips", {}) if isinstance(variant_result, dict) else {}
            if not isinstance(variant_clips, dict):
                continue
            candidate_path = variant_clips.get(clip_key)
            reference_path = reference_clips.get(clip_key)
            if not candidate_path or not reference_path:
                continue
            analysis = CoverPipeline._analyze_quality_stage(candidate_path=candidate_path, reference_path=reference_path)
            analysis["artifact_score_estimate"] = _estimate_artifact_score(analysis)
            clip_summary["variants"][variant] = {
                "path": candidate_path,
                "analysis": analysis,
            }
        ranking = []
        variants_dict = clip_summary.get("variants", {})
        if isinstance(variants_dict, dict):
            for variant_name, data in variants_dict.items():
                if not isinstance(data, dict):
                    continue
                analysis = data.get("analysis", {})
                if not isinstance(analysis, dict):
                    continue
                ranking.append(
                    {
                        "variant": variant_name,
                        "artifact_score_estimate": float(analysis.get("artifact_score_estimate", 0.0) or 0.0),
                    }
                )
        ranking.sort(key=lambda item: item["artifact_score_estimate"])
        clip_summary["ranking"] = ranking
        per_clip_results.append(clip_summary)

    for variant in requested_variants:
        analyses = []
        for clip_summary in per_clip_results:
            variants_dict = clip_summary.get("variants", {})
            if not isinstance(variants_dict, dict):
                continue
            data = variants_dict.get(variant)
            if not isinstance(data, dict):
                continue
            analysis = data.get("analysis")
            if isinstance(analysis, dict):
                analyses.append(analysis)
        aggregate = _aggregate_variant_metrics(analyses)
        variant_result = variant_results.get(variant, {})
        if isinstance(variant_result, dict) and "skipped" in variant_result:
            aggregate["skipped"] = variant_result["skipped"]
        aggregate_results[variant] = aggregate

    ranking = [
        {
            "variant": variant,
            "avg_artifact_score_estimate": float(data.get("avg_artifact_score_estimate", 9999.0) or 9999.0),
            "clips": int(data.get("clips", 0) or 0),
        }
        for variant, data in aggregate_results.items()
        if not data.get("skipped")
    ]
    ranking.sort(key=lambda item: item["avg_artifact_score_estimate"])

    summary = {
        "session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "index_path": str(index_path) if index_path else None,
        "model_inferred_from_session": inferred_model_key,
        "preprocess_mode": preprocess_mode,
        "source_constraint_mode": args.source_constraint_mode,
        "times_sec": [float(time_sec) for time_sec in times_sec],
        "window_sec": float(args.window_sec),
        "gap_sec": float(args.gap_sec),
        "manifest": manifest,
        "variants": variant_results,
        "per_clip": per_clip_results,
        "aggregate": aggregate_results,
        "ranking": ranking,
        "score_note": (
            "artifact_score_estimate is a heuristic based on quiet-region energy, "
            "high-frequency excess, transition spikes, and synthetic breath count. Lower is better."
        ),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nAggregate ranking (lower artifact score is better):")
    if ranking:
        for item in ranking:
            print(
                f"  {item['variant']}: score={item['avg_artifact_score_estimate']:.4f}, "
                f"clips={item['clips']}"
            )
    else:
        print("  No completed variants.")
    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
