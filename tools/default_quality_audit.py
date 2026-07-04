#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Default-parameter quality audit for AI cover output.

This tool is intentionally conservative:
- it locks conversion to the repository defaults;
- metrics are guardrails only, so they can block a case but cannot pass it;
- human listening review is required for the final verdict.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_SONG_TAGS = (
    "high_note",
    "low_note",
    "fast_song",
    "slow_song",
    "breathy",
    "long_tail",
    "complex_harmony",
)
REQUIRED_ROLE_COUNT = 3
REQUIRED_CASE_COUNT = 7

LISTENING_REVIEW_FIELDS = (
    "default_one_click_quality",
    "character_identity",
    "source_distinctness",
    "artifact_absence",
    "lyrics_and_performance",
    "mix_integration",
)
MIN_LISTENING_SCORE = 4

FORBIDDEN_CASE_OVERRIDE_KEYS = {
    "parameters",
    "params",
    "overrides",
    "pitch_shift",
    "index_ratio",
    "index_rate",
    "filter_radius",
    "rms_mix_rate",
    "protect",
    "speaker_id",
    "f0_method",
    "separator",
    "roformer_model",
    "karaoke_separation",
    "karaoke_model",
    "uvr5_model",
    "uvr5_agg",
    "uvr5_format",
    "use_official",
    "demucs_model",
    "demucs_shifts",
    "demucs_overlap",
    "demucs_split",
    "hubert_layer",
    "vocals_volume",
    "accompaniment_volume",
    "reverb_amount",
    "backing_mix",
    "vc_preprocess_mode",
    "source_constraint_mode",
    "vc_pipeline_mode",
    "singing_repair",
}

TECHNICAL_ALERT_LIMITS = {
    "peak": 0.985,
    "clip_ratio": 0.00001,
    "quiet_rms_ratio": 3.0,
    "quiet_hf_ratio": 2.5,
    "midquiet_hf_ratio": 2.5,
    "transition_spike_ratio": 1.6,
    "synthetic_breath_frames": 350,
}

QUALITY_PRINCIPLE = (
    "Technical metrics are guardrails: they can block a case, but they cannot pass audio "
    "by themselves. Listening review is required so the audit does not chase numbers at "
    "the expense of character identity, lyrics, or musical quality."
)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_json(config_path or (REPO_ROOT / "configs" / "config.json"))


def validate_manifest_schema(
    manifest: Mapping[str, Any],
    *,
    require_files: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        raise ValueError("Quality audit manifest must be a JSON object.")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Quality audit manifest must contain a non-empty cases list.")

    validated: list[dict[str, Any]] = []
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"Case #{index + 1} must be an object.")
        case = dict(raw_case)
        case_name = str(case.get("name") or f"case_{index + 1}")

        forbidden = sorted(FORBIDDEN_CASE_OVERRIDE_KEYS & set(case.keys()))
        if forbidden:
            joined = ", ".join(forbidden)
            raise ValueError(
                f"Case '{case_name}' must use default parameters; remove override field(s): {joined}."
            )

        for required_key in ("name", "input_audio", "model_path", "role_id", "tags"):
            if required_key not in case:
                raise ValueError(f"Case '{case_name}' is missing required field: {required_key}.")

        tags = case.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags):
            raise ValueError(f"Case '{case_name}' tags must be a non-empty list of strings.")

        if require_files:
            for path_key in ("input_audio", "model_path"):
                path = _resolve_path(str(case[path_key]))
                if not path.exists():
                    raise FileNotFoundError(f"Case '{case_name}' {path_key} does not exist: {path}")
            index_path = case.get("index_path")
            if index_path:
                resolved_index = _resolve_path(str(index_path))
                if not resolved_index.exists():
                    raise FileNotFoundError(f"Case '{case_name}' index_path does not exist: {resolved_index}")

        validated.append(case)
    return validated


def analyze_manifest_coverage(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cases = validate_manifest_schema(manifest)
    seen_song_tags = {
        tag.strip()
        for case in cases
        for tag in case.get("tags", [])
        if isinstance(tag, str) and tag.strip()
    }
    role_ids = {
        str(case.get("role_id", "")).strip()
        for case in cases
        if str(case.get("role_id", "")).strip()
    }
    missing_song_tags = [tag for tag in REQUIRED_SONG_TAGS if tag not in seen_song_tags]
    case_count = len(cases)
    role_count = len(role_ids)
    missing_requirements = []
    if missing_song_tags:
        missing_requirements.append("song_tag_coverage")
    if role_count < REQUIRED_ROLE_COUNT:
        missing_requirements.append("role_count")
    if case_count < REQUIRED_CASE_COUNT:
        missing_requirements.append("case_count")

    return {
        "met": not missing_requirements,
        "case_count": case_count,
        "required_case_count": REQUIRED_CASE_COUNT,
        "role_count": role_count,
        "required_role_count": REQUIRED_ROLE_COUNT,
        "song_tags": sorted(seen_song_tags),
        "required_song_tags": list(REQUIRED_SONG_TAGS),
        "missing_song_tags": missing_song_tags,
        "missing_requirements": missing_requirements,
    }


def build_default_cover_kwargs(
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    validate_manifest_schema({"cases": [dict(case)]})
    cover_cfg = config.get("cover")
    if not isinstance(cover_cfg, Mapping):
        raise KeyError("configs/config.json is missing the cover section.")

    def cover_value(key: str) -> Any:
        if key not in cover_cfg:
            raise KeyError(f"configs/config.json cover section is missing required key: {key}")
        return cover_cfg[key]

    output_dir = Path(output_root) / _safe_name(str(case["name"]))
    return {
        "input_audio": str(_resolve_path(str(case["input_audio"]))),
        "model_path": str(_resolve_path(str(case["model_path"]))),
        "index_path": str(_resolve_path(str(case["index_path"]))) if case.get("index_path") else None,
        "pitch_shift": 0,
        "index_ratio": float(cover_value("index_rate")),
        "filter_radius": int(cover_value("filter_radius")),
        "rms_mix_rate": float(cover_value("rms_mix_rate")),
        "protect": float(cover_value("protect")),
        "speaker_id": int(cover_value("speaker_id")),
        "f0_method": str(cover_value("f0_method")),
        "demucs_model": str(cover_value("demucs_model")),
        "demucs_shifts": int(cover_value("demucs_shifts")),
        "demucs_overlap": float(cover_value("demucs_overlap")),
        "demucs_split": bool(cover_value("demucs_split")),
        "roformer_model": str(cover_value("roformer_model")),
        "separator": str(cover_value("separator")),
        "uvr5_model": str(cover_value("uvr5_model")),
        "uvr5_agg": int(cover_value("uvr5_agg")),
        "uvr5_format": str(cover_value("uvr5_format")),
        "use_official": bool(cover_value("use_official")),
        "hubert_layer": int(cover_value("hubert_layer")),
        "vocals_volume": float(cover_value("default_vocals_volume")) / 100.0,
        "accompaniment_volume": float(cover_value("default_accompaniment_volume")) / 100.0,
        "reverb_amount": float(cover_value("default_reverb")) / 100.0,
        "backing_mix": float(cover_value("backing_mix")),
        "karaoke_separation": bool(cover_value("karaoke_separation")),
        "karaoke_model": str(cover_value("karaoke_model")),
        "karaoke_merge_backing_into_accompaniment": bool(cover_value("karaoke_merge_backing_into_accompaniment")),
        "vc_preprocess_mode": str(cover_value("vc_preprocess_mode")),
        "source_constraint_mode": str(cover_value("source_constraint_mode")),
        "vc_pipeline_mode": str(cover_value("vc_pipeline_mode")),
        "output_dir": str(output_dir),
        "model_display_name": str(case["role_id"]),
    }


def build_quality_verdict(
    manifest: Mapping[str, Any],
    technical_results: Mapping[str, Any] | None = None,
    listening_reviews: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = validate_manifest_schema(manifest)
    coverage = analyze_manifest_coverage(manifest)
    technical_results = technical_results or {}
    listening_reviews = listening_reviews or {}

    case_reports = []
    blocking_cases: list[str] = []
    needs_listening: list[str] = []
    needs_quality_work: list[str] = []
    needs_technical_evidence: list[str] = []

    for case in cases:
        case_name = str(case["name"])
        technical = technical_results.get(case_name, {})
        if not isinstance(technical, Mapping):
            technical = {}
        runtime_error = technical.get("error")
        analysis = _extract_quality_analysis(technical)
        alerts = _technical_alerts(analysis)

        review = listening_reviews.get(case_name)
        review_status = _listening_review_status(review)
        status = "pass"
        if runtime_error:
            status = "blocked_by_runtime_failure"
            blocking_cases.append(case_name)
        elif alerts:
            status = "blocked_by_technical_alerts"
            blocking_cases.append(case_name)
        elif review_status["missing_fields"]:
            status = "needs_listening_review"
            needs_listening.append(case_name)
        elif review_status["low_fields"]:
            status = "needs_quality_work"
            needs_quality_work.append(case_name)
        elif not analysis:
            status = "needs_technical_evidence"
            needs_technical_evidence.append(case_name)

        case_reports.append(
            {
                "name": case_name,
                "role_id": case["role_id"],
                "tags": case["tags"],
                "status": status,
                "technical_alerts": alerts,
                "listening_review_status": review_status,
                "has_technical_evidence": bool(analysis),
                "runtime_error": str(runtime_error) if runtime_error else None,
            }
        )

    if not coverage["met"]:
        verdict = "blocked_by_matrix_coverage"
    elif blocking_cases:
        verdict = "blocked_by_technical_alerts"
    elif needs_listening:
        verdict = "needs_listening_review"
    elif needs_quality_work:
        verdict = "needs_quality_work"
    elif needs_technical_evidence:
        verdict = "needs_technical_evidence"
    else:
        verdict = "pass"

    return {
        "verdict": verdict,
        "quality_principle": QUALITY_PRINCIPLE,
        "coverage": coverage,
        "blocking_cases": blocking_cases,
        "needs_listening_review": needs_listening,
        "needs_quality_work": needs_quality_work,
        "needs_technical_evidence": needs_technical_evidence,
        "cases": case_reports,
    }


def run_default_matrix(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    *,
    device: str,
) -> dict[str, Any]:
    from infer.cover_pipeline import CoverPipeline

    cases = validate_manifest_schema(manifest, require_files=True)
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for case in cases:
        case_name = str(case["name"])
        kwargs = build_default_cover_kwargs(case, config, output_root)
        started = time.time()
        pipeline = CoverPipeline(device=device)
        try:
            result = pipeline.process(**kwargs)
            quality_debug = _load_case_quality_debug(result)
            results[case_name] = {
                "status": "pass",
                "seconds": round(time.time() - started, 3),
                "result": result,
                "default_parameters": _public_default_parameter_record(kwargs),
                "quality_debug": quality_debug,
            }
        except Exception as exc:
            results[case_name] = {
                "status": "fail",
                "seconds": round(time.time() - started, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "default_parameters": _public_default_parameter_record(kwargs),
            }
        finally:
            pipeline.cleanup_all()
    return results


def write_report_files(report: Mapping[str, Any], output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "default_quality_audit.json"
    markdown_path = output_root / "default_quality_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON file describing default-parameter audit cases.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "config.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "default_quality_audit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true", help="Run real default-parameter conversions.")
    parser.add_argument("--reviews", help="Optional JSON file with human listening review scores.")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest)
    config_path = _resolve_path(args.config)
    output_root = _resolve_path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")

    manifest = load_json(manifest_path)
    config = load_config(config_path)
    validate_manifest_schema(manifest, require_files=bool(args.run))

    technical_results: dict[str, Any] = {}
    if args.run:
        technical_results = run_default_matrix(
            manifest,
            config,
            output_root / "runs",
            device=str(args.device),
        )

    reviews = load_json(_resolve_path(args.reviews)) if args.reviews else {}
    report = build_quality_verdict(manifest, technical_results, reviews)
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
    }
    report["manifest_path"] = str(manifest_path)
    report["config_path"] = str(config_path)
    report["technical_results"] = technical_results

    written = write_report_files(report, output_root)
    print(f"Verdict: {report['verdict']}")
    print(f"Report JSON: {written['json']}")
    print(f"Report MD: {written['markdown']}")
    return 0 if report["verdict"] == "pass" else 1


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._") or "case"


def _extract_quality_analysis(technical: Mapping[str, Any]) -> dict[str, Any]:
    direct = technical.get("quality_debug")
    if isinstance(direct, Mapping):
        if "stages" in direct:
            return _select_quality_stage_analysis(direct)
        return dict(direct)
    return {}


def _select_quality_stage_analysis(payload: Mapping[str, Any]) -> dict[str, Any]:
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return {}
    preferred = (
        "vc_final_state",
        "vc_quality_selected_current",
        "vc_quality_selected",
        "vc_refined",
        "cover_mix",
    )
    for stage_name in preferred:
        for entry in reversed(stages):
            if not isinstance(entry, Mapping):
                continue
            if entry.get("stage") != stage_name:
                continue
            analysis = entry.get("analysis")
            return dict(analysis) if isinstance(analysis, Mapping) else {}
    return {}


def _technical_alerts(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    alerts = []
    for metric, limit in TECHNICAL_ALERT_LIMITS.items():
        if metric not in analysis:
            continue
        value = analysis[metric]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > float(limit):
            alerts.append(
                {
                    "metric": metric,
                    "value": numeric,
                    "limit": float(limit),
                    "reason": "above_guardrail",
                }
            )
    return alerts


def _listening_review_status(review: Any) -> dict[str, Any]:
    if not isinstance(review, Mapping):
        return {
            "missing_fields": list(LISTENING_REVIEW_FIELDS),
            "low_fields": [],
        }
    missing = []
    low = []
    for field in LISTENING_REVIEW_FIELDS:
        if field not in review:
            missing.append(field)
            continue
        try:
            score = int(review[field])
        except (TypeError, ValueError):
            missing.append(field)
            continue
        if score < MIN_LISTENING_SCORE:
            low.append({"field": field, "score": score, "minimum": MIN_LISTENING_SCORE})
    return {"missing_fields": missing, "low_fields": low}


def _load_case_quality_debug(result: Mapping[str, Any]) -> dict[str, Any]:
    all_files_dir = result.get("all_files_dir")
    if not all_files_dir:
        raise RuntimeError("Cover result is missing all_files_dir; cannot collect quality evidence.")
    report_path = Path(str(all_files_dir)) / "quality_debug.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing quality_debug.json for audit case: {report_path}")
    return load_json(report_path)


def _public_default_parameter_record(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    hidden_paths = {"input_audio", "model_path", "index_path", "output_dir", "model_display_name"}
    return {
        key: value
        for key, value in kwargs.items()
        if key not in hidden_paths
    }


def _render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Default Quality Audit",
        "",
        f"- verdict: `{report.get('verdict')}`",
        f"- principle: {report.get('quality_principle', QUALITY_PRINCIPLE)}",
        "",
        "## Coverage",
    ]
    coverage = report.get("coverage", {})
    if isinstance(coverage, Mapping):
        lines.extend(
            [
                f"- cases: {coverage.get('case_count')} / {coverage.get('required_case_count')}",
                f"- roles: {coverage.get('role_count')} / {coverage.get('required_role_count')}",
                f"- missing song tags: {', '.join(coverage.get('missing_song_tags', [])) or 'none'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Cases",
            "",
            "| case | status | technical alerts | listening gaps |",
            "|---|---|---:|---:|",
        ]
    )
    for item in report.get("cases", []):
        if not isinstance(item, Mapping):
            continue
        review = item.get("listening_review_status", {})
        missing = review.get("missing_fields", []) if isinstance(review, Mapping) else []
        lines.append(
            f"| `{item.get('name')}` | `{item.get('status')}` | "
            f"{len(item.get('technical_alerts', []))} | {len(missing)} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
