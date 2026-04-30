import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_quality_policy_module():
    module_path = REPO_ROOT / "infer" / "quality_policy.py"
    spec = importlib.util.spec_from_file_location("quality_policy", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


quality_policy = _load_quality_policy_module()
build_conservative_crepe_fill_mask = quality_policy.build_conservative_crepe_fill_mask
build_conservative_harvest_fill_mask = quality_policy.build_conservative_harvest_fill_mask
compute_chunk_crossfade_samples = quality_policy.compute_chunk_crossfade_samples
compute_active_source_replace = quality_policy.compute_active_source_replace
compute_breath_preserving_energy_gates = quality_policy.compute_breath_preserving_energy_gates
compute_source_cleanup_budget = quality_policy.compute_source_cleanup_budget
compute_residual_quiet_hf_blend_curve = quality_policy.compute_residual_quiet_hf_blend_curve
compute_midquiet_transition_hf_blend_curve = quality_policy.compute_midquiet_transition_hf_blend_curve
compute_accompaniment_leakage_metrics = quality_policy.compute_accompaniment_leakage_metrics
compute_karaoke_stem_separation_metrics = quality_policy.compute_karaoke_stem_separation_metrics
compute_mix_fusion_metrics = quality_policy.compute_mix_fusion_metrics
resolve_cover_f0_policy = quality_policy.resolve_cover_f0_policy


class F0RoutingPolicyTests(unittest.TestCase):
    def test_hybrid_routes_to_conservative_rmvpe_fallback(self):
        policy = resolve_cover_f0_policy("hybrid", "off")

        self.assertEqual(policy.requested_method, "hybrid")
        self.assertEqual(policy.vc_method, "rmvpe")
        self.assertEqual(policy.hybrid_mode, "fallback")
        self.assertEqual(policy.gate_method, "rmvpe")

    def test_official_cover_profile_matches_upstream_single_infer_defaults(self):
        profile = quality_policy.get_official_cover_vc_profile()

        self.assertEqual(profile["separator"], "uvr5")
        self.assertFalse(profile["karaoke_separation"])
        self.assertFalse(profile["karaoke_merge_backing_into_accompaniment"])
        self.assertEqual(profile["vc_preprocess_mode"], "direct")
        self.assertEqual(profile["source_constraint_mode"], "off")
        self.assertEqual(profile["f0_method"], "rmvpe")
        self.assertAlmostEqual(profile["index_rate"], 0.75, places=6)
        self.assertEqual(profile["filter_radius"], 3)
        self.assertAlmostEqual(profile["rms_mix_rate"], 0.75, places=6)
        self.assertAlmostEqual(profile["official_rms_mix_rate"], 0.25, places=6)
        self.assertAlmostEqual(profile["protect"], 0.33, places=6)
        self.assertFalse(profile["singing_repair"])


class ConservativeCrepeFillTests(unittest.TestCase):
    def test_only_short_internal_gaps_are_filled(self):
        f0_rmvpe = np.array([0.0, 0.0, 120.0, 120.0, 0.0, 0.0, 120.0, 120.0, 0.0, 0.0], dtype=np.float32)
        f0_crepe = np.full_like(f0_rmvpe, 121.0)
        confidence = np.full_like(f0_rmvpe, 0.95)

        fill_mask = build_conservative_crepe_fill_mask(
            f0_rmvpe,
            f0_crepe,
            confidence,
            confidence_threshold=0.6,
            max_ratio=0.5,
            max_frames=4,
            context_radius=2,
        )

        expected = np.array([False, False, False, False, True, True, False, False, False, False])
        np.testing.assert_array_equal(fill_mask, expected)

    def test_harvest_fill_rejects_long_gap_and_accepts_short_consistent_gap(self):
        reference_f0 = np.array(
            [120.0, 121.0, 122.0, 0.0, 0.0, 121.5, 122.0, 121.0, 0.0, 0.0, 0.0, 0.0, 122.0, 121.0],
            dtype=np.float32,
        )
        fallback_f0 = np.array(
            [0.0, 0.0, 0.0, 121.0, 121.5, 0.0, 0.0, 0.0, 122.0, 122.0, 122.0, 122.0, 0.0, 0.0],
            dtype=np.float32,
        )
        dropout_mask = reference_f0 <= 0

        fill_mask = build_conservative_harvest_fill_mask(
            reference_f0=reference_f0,
            fallback_f0=fallback_f0,
            dropout_mask=dropout_mask,
            max_run=3,
            local_radius=2,
            max_semitones=2.0,
        )

        expected = np.array(
            [False, False, False, True, True, False, False, False, False, False, False, False, False, False],
            dtype=bool,
        )
        np.testing.assert_array_equal(fill_mask, expected)

    def test_chunk_crossfade_scales_above_legacy_floor_for_multi_segment_audio(self):
        self.assertEqual(
            compute_chunk_crossfade_samples(tgt_sr=48000, t_pad_tgt=144000, segment_count=1),
            0,
        )
        self.assertEqual(
            compute_chunk_crossfade_samples(tgt_sr=48000, t_pad_tgt=144000, segment_count=2),
            864,
        )
        self.assertEqual(
            compute_chunk_crossfade_samples(tgt_sr=48000, t_pad_tgt=144000, segment_count=5),
            1152,
        )


class SourceConstraintPolicyTests(unittest.TestCase):
    def test_active_echo_frames_keep_nonzero_replace_pressure(self):
        activity = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        soft_mask = np.array([[0.10, 0.55, 0.10]], dtype=np.float32)
        echo_ratio = np.array([[0.90, 0.85, 0.40]], dtype=np.float32)
        direct_ratio = np.array([0.10, 0.25, 0.15], dtype=np.float32)

        replace = compute_active_source_replace(activity, soft_mask, echo_ratio, direct_ratio)

        self.assertGreater(replace[0, 0], 0.40)
        self.assertGreater(replace[0, 1], 0.10)
        self.assertLess(replace[0, 1], replace[0, 0])
        self.assertLessEqual(float(np.max(replace)), 0.82)

    def test_cleanup_budget_caps_active_boost_below_two_x(self):
        energy_guard = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        phrase_activity = np.array([0.0, 0.5, 1.0], dtype=np.float32)

        allowed_boost, cleanup_floor = compute_source_cleanup_budget(
            energy_guard,
            phrase_activity,
        )

        np.testing.assert_allclose(
            allowed_boost,
            np.array([0.35, 0.85, 1.35], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cleanup_floor,
            np.array([0.62, 0.70, 0.78], dtype=np.float32),
            atol=1e-6,
        )
        self.assertLess(float(np.max(allowed_boost)), 1.5)

    def test_residual_quiet_hf_cleanup_targets_only_low_energy_excess(self):
        source_rms = np.array([0.12, 0.07, 0.012, 0.006, 0.04], dtype=np.float32)
        converted_rms = np.array([0.14, 0.08, 0.030, 0.006, 0.045], dtype=np.float32)
        source_hf = np.array([0.020, 0.014, 0.003, 0.0015, 0.010], dtype=np.float32)
        converted_hf = np.array([0.060, 0.040, 0.018, 0.0016, 0.011], dtype=np.float32)

        blend = compute_residual_quiet_hf_blend_curve(
            source_rms=source_rms,
            converted_rms=converted_rms,
            source_hf=source_hf,
            converted_hf=converted_hf,
        )

        self.assertLess(float(blend[0]), 0.08)
        self.assertLess(float(blend[1]), 0.16)
        self.assertGreater(float(blend[2]), 0.30)
        self.assertLess(float(blend[3]), 0.05)
        self.assertLess(float(blend[4]), 0.05)
        self.assertLessEqual(float(np.max(blend)), 0.68)

    def test_midquiet_transition_hf_cleanup_ignores_body_and_deep_gaps(self):
        source_rms = np.array([0.12, 0.075, 0.042, 0.018, 0.002, 0.045], dtype=np.float32)
        converted_rms = np.array([0.13, 0.090, 0.070, 0.021, 0.002, 0.047], dtype=np.float32)
        source_hf = np.array([0.030, 0.020, 0.011, 0.004, 0.0008, 0.012], dtype=np.float32)
        converted_hf = np.array([0.080, 0.055, 0.040, 0.0043, 0.0010, 0.0122], dtype=np.float32)

        blend = compute_midquiet_transition_hf_blend_curve(
            source_rms=source_rms,
            converted_rms=converted_rms,
            source_hf=source_hf,
            converted_hf=converted_hf,
        )

        self.assertLess(float(blend[0]), 0.08)
        self.assertGreater(float(blend[1]), 0.12)
        self.assertGreater(float(blend[2]), 0.20)
        self.assertLess(float(blend[3]), 0.06)
        self.assertLess(float(blend[4]), 0.03)
        self.assertLess(float(blend[5]), 0.05)
        self.assertLessEqual(float(np.max(blend)), 0.46)


class StemQualityMetricTests(unittest.TestCase):
    def test_accompaniment_leakage_metrics_rank_vocal_shaped_leakage_higher(self):
        vocal_voiceband = np.array(
            [0.010, 0.095, 0.110, 0.100, 0.012, 0.090, 0.105, 0.011],
            dtype=np.float32,
        )
        clean_accompaniment = np.full_like(vocal_voiceband, 0.018)
        leaked_accompaniment = clean_accompaniment + 0.38 * vocal_voiceband

        clean = compute_accompaniment_leakage_metrics(
            accompaniment_voiceband_rms=clean_accompaniment,
            vocal_voiceband_rms=vocal_voiceband,
        )
        leaked = compute_accompaniment_leakage_metrics(
            accompaniment_voiceband_rms=leaked_accompaniment,
            vocal_voiceband_rms=vocal_voiceband,
        )

        self.assertLess(clean["leakage_risk_score"], 0.25)
        self.assertGreater(leaked["leakage_risk_score"], clean["leakage_risk_score"] + 0.35)
        self.assertGreater(leaked["vocal_activity_correlation"], 0.85)
        self.assertGreater(leaked["active_to_quiet_voiceband_db"], clean["active_to_quiet_voiceband_db"] + 5.0)

    def test_karaoke_stem_metrics_flag_duplicated_lead_in_backing(self):
        lead_rms = np.array([0.010, 0.110, 0.120, 0.100, 0.012, 0.090, 0.100, 0.010], dtype=np.float32)
        separated_backing = np.array([0.008, 0.018, 0.020, 0.017, 0.070, 0.020, 0.018, 0.065], dtype=np.float32)
        duplicated_backing = 0.72 * lead_rms

        separated = compute_karaoke_stem_separation_metrics(
            lead_rms=lead_rms,
            backing_rms=separated_backing,
        )
        duplicated = compute_karaoke_stem_separation_metrics(
            lead_rms=lead_rms,
            backing_rms=duplicated_backing,
        )

        self.assertLess(separated["duplication_risk_score"], 0.45)
        self.assertGreater(duplicated["duplication_risk_score"], separated["duplication_risk_score"] + 0.35)
        self.assertGreater(duplicated["envelope_correlation"], 0.95)
        self.assertGreater(duplicated["mutual_active_ratio"], separated["mutual_active_ratio"])

    def test_mix_fusion_metrics_detect_bed_ducking_and_overloud_backing(self):
        lead_rms = np.array([0.010, 0.100, 0.110, 0.095, 0.012, 0.090, 0.105, 0.010], dtype=np.float32)
        backing_ok = np.array([0.004, 0.020, 0.025, 0.020, 0.006, 0.018, 0.022, 0.004], dtype=np.float32)
        backing_loud = np.array([0.004, 0.085, 0.090, 0.080, 0.006, 0.075, 0.085, 0.004], dtype=np.float32)
        stable_bed = np.full_like(lead_rms, 0.080)
        ducked_bed = np.array([0.085, 0.048, 0.045, 0.050, 0.082, 0.050, 0.047, 0.084], dtype=np.float32)

        stable = compute_mix_fusion_metrics(
            lead_rms=lead_rms,
            backing_rms=backing_ok,
            bed_rms=stable_bed,
        )
        ducked = compute_mix_fusion_metrics(
            lead_rms=lead_rms,
            backing_rms=backing_loud,
            bed_rms=ducked_bed,
        )

        self.assertLess(stable["ducking_risk_score"], 0.20)
        self.assertGreater(ducked["ducking_risk_score"], stable["ducking_risk_score"] + 0.45)
        self.assertLess(stable["backing_excess_frame_ratio"], 0.10)
        self.assertGreater(ducked["backing_excess_frame_ratio"], 0.70)
        self.assertLess(ducked["bed_active_vs_quiet_db"], -3.0)


class BreathEnergyGateTests(unittest.TestCase):
    def test_marginal_quiet_unvoiced_frames_keep_more_feature_than_pitch(self):
        energy_db = np.array([-70.0, -60.0, -47.0, -38.0], dtype=np.float32)
        unvoiced_mask = np.array([True, True, True, False], dtype=bool)

        feature_gate, pitch_gate = compute_breath_preserving_energy_gates(
            energy_db=energy_db,
            ref_db=-12.0,
            unvoiced_mask=unvoiced_mask,
            quiet_floor=0.05,
            breath_floor=0.28,
            breath_active_margin_db=52.0,
            transition_width_db=6.0,
        )

        self.assertAlmostEqual(float(feature_gate[0]), 0.05, places=5)
        self.assertAlmostEqual(float(pitch_gate[0]), 0.05, places=5)
        self.assertGreater(float(feature_gate[1]), float(pitch_gate[1]))
        self.assertGreater(float(feature_gate[1]), 0.13)
        self.assertLess(float(pitch_gate[1]), 0.13)
        self.assertGreater(float(feature_gate[1]), float(pitch_gate[1]))
        self.assertAlmostEqual(float(feature_gate[2]), float(pitch_gate[2]), places=5)
        self.assertAlmostEqual(float(feature_gate[3]), float(pitch_gate[3]), places=5)


class SourceRegressionTests(unittest.TestCase):
    def test_cover_pipeline_uses_single_conservative_cleanup_budget(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertNotIn("allowed_boost = 0.50 + 1.50 * energy_guard", source)
        self.assertNotRegex(
            source,
            r"cleanup_gain\s*=\s*np\.clip\(\s*frame_budget\s*/\s*\(constrained_frame_rms \+ eps\),\s*0\.75 \+ 0\.20 \* phrase_activity",
        )

    def test_cover_pipeline_gain_parameters_match_runtime_clip(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertIn("min_gain=0.85", source)
        self.assertIn("max_gain=1.12", source)
        self.assertNotIn("min_gain=0.95", source)
        self.assertNotIn("max_gain=1.30", source)
        self.assertIn("reduction_ratio", source)
        self.assertIn("uvr_deecho_plus", source)
        self.assertIn("_apply_source_breath_cleanup", source)
        self.assertIn("_apply_source_transition_cleanup", source)
        self.assertIn("Source breath cleanup:", source)
        self.assertIn("Source transition cleanup:", source)
        self.assertIn("_record_quality_debug", source)
        self.assertIn("quality_debug.json", source)
        self.assertIn("debug_clips", source)
        self.assertIn("VC backend: upstream_official_raw + current postprocess", source)
        self.assertIn("convert_vocals_official_upstream(", source)
        self.assertNotIn('log.detail("使用当前项目官方封装VC进行转换")\n                convert_vocals_official(', source)

    def test_cover_pipeline_has_final_residual_quiet_hf_cleanup(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertIn("compute_residual_quiet_hf_blend_curve", source)
        self.assertIn("def _apply_residual_quiet_hf_cleanup(", source)
        self.assertIn("Residual quiet-HF cleanup:", source)
        self.assertIn("_apply_residual_quiet_hf_cleanup(", source.split("def _refine_source_constrained_output(", 1)[1])

    def test_cover_pipeline_has_midquiet_transition_hf_cleanup(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertIn("compute_midquiet_transition_hf_blend_curve", source)
        self.assertIn("def _apply_midquiet_transition_hf_cleanup(", source)
        self.assertIn("Midquiet transition-HF cleanup:", source)
        refine_body = source.split("def _refine_source_constrained_output(", 1)[1]
        self.assertIn("_apply_midquiet_transition_hf_cleanup(", refine_body)
        self.assertLess(
            refine_body.index("_apply_midquiet_transition_hf_cleanup("),
            refine_body.index("_apply_residual_quiet_hf_cleanup("),
        )

    def test_cover_pipeline_records_full_stem_quality_metrics(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertIn("compute_accompaniment_leakage_metrics", source)
        self.assertIn("compute_karaoke_stem_separation_metrics", source)
        self.assertIn("compute_mix_fusion_metrics", source)
        self.assertIn('stage="accompaniment_purity"', source)
        self.assertIn('stage="karaoke_split"', source)
        self.assertIn('stage="mix_fusion"', source)
        self.assertIn("_record_accompaniment_purity_debug(", source)
        self.assertIn("_record_karaoke_split_debug(", source)
        self.assertIn("_record_mix_fusion_debug(", source)

    def test_official_vc_pipeline_no_longer_forces_hybrid_to_aggressive_crepe(self):
        source = (REPO_ROOT / "infer" / "modules" / "vc" / "pipeline.py").read_text(encoding="utf-8")

        self.assertNotIn('self.f0_hybrid_mode = "rmvpe+crepe"', source)

    def test_config_defaults_match_conservative_hybrid_policy(self):
        source = (REPO_ROOT / "configs" / "config.json").read_text(encoding="utf-8")

        self.assertRegex(source, r'"f0_hybrid_mode"\s*:\s*"fallback"')
        self.assertNotRegex(source, r'"crepe_force_ratio"\s*:\s*0\.0')

    def test_official_vc_pipeline_uses_breath_preserving_energy_gates(self):
        source = (REPO_ROOT / "infer" / "modules" / "vc" / "pipeline.py").read_text(encoding="utf-8")

        self.assertIn("compute_breath_preserving_energy_gates", source)
        self.assertIn("self.unvoiced_feature_gate_floor", source)
        self.assertIn("compute_chunk_crossfade_samples", source)
        self.assertIn("build_conservative_harvest_fill_mask", source)
        self.assertIn("分段边界(秒)", source)

    def test_config_includes_breath_gate_defaults(self):
        source = (REPO_ROOT / "configs" / "config.json").read_text(encoding="utf-8")

        self.assertRegex(source, r'"unvoiced_feature_gate_floor"\s*:\s*0\.28')
        self.assertRegex(source, r'"breath_active_margin_db"\s*:\s*52\.0')

    def test_cover_default_rms_mix_preserves_fb61a20_vocal_body(self):
        config = json.loads((REPO_ROOT / "configs" / "config.json").read_text(encoding="utf-8"))
        cover_source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertAlmostEqual(float(config["cover"]["rms_mix_rate"]), 0.0, places=6)
        self.assertIn("rms_mix_rate: float = 0.0", cover_source)

    def test_cover_default_protect_uses_fb61a20_artifact_guard(self):
        config = json.loads((REPO_ROOT / "configs" / "config.json").read_text(encoding="utf-8"))
        cover_source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertAlmostEqual(float(config["cover"]["protect"]), 0.33, places=6)
        self.assertIn("protect: float = 0.33", cover_source)
        self.assertIn("| 保护系数 | 防止撕裂伪影，越小保护越强 | 0.33 |", readme)

    def test_readme_cover_config_snippet_matches_measured_defaults(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn('"rms_mix_rate": 0.0', readme)
        self.assertIn('"protect": 0.33', readme)
        self.assertNotIn('"protect": 0.50', readme)

    def test_karaoke_backing_merge_uses_bed_blend_without_dynamic_ducking(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        merge_body = source.split("def _merge_backing_into_accompaniment(", 1)[1].split("def _init_rvc_pipeline(", 1)[0]
        self.assertIn("_blend_backing_into_accompaniment_bed(", merge_body)
        self.assertNotIn("_duck_backing_under_lead(", merge_body)
        self.assertIn("和声融入伴奏", source)

    def test_default_final_mix_does_not_adaptively_duck_accompaniment(self):
        spec = importlib.util.spec_from_file_location("mixer_under_test", REPO_ROOT / "lib" / "mixer.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mixer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mixer
        spec.loader.exec_module(mixer)

        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        accompaniment = np.stack([
            0.12 * np.sin(2 * np.pi * 220 * t),
            0.12 * np.sin(2 * np.pi * 224 * t),
        ], axis=1).astype(np.float32)
        vocals = np.zeros((sr, 2), dtype=np.float32)
        vocals[sr // 4 : sr // 2, 0] = 0.08 * np.sin(2 * np.pi * 440 * t[: sr // 4])
        vocals[sr // 4 : sr // 2, 1] = vocals[sr // 4 : sr // 2, 0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            vocals_path = tmp_path / "vocals.wav"
            accompaniment_path = tmp_path / "accompaniment.wav"
            output_path = tmp_path / "mix.wav"
            sf.write(vocals_path, vocals, sr)
            sf.write(accompaniment_path, accompaniment, sr)

            with patch.object(
                mixer,
                "_apply_adaptive_vocal_ducking",
                side_effect=AssertionError("default final mix must not pump the accompaniment"),
            ):
                mixer.mix_vocals_and_accompaniment(
                    str(vocals_path),
                    str(accompaniment_path),
                    str(output_path),
                    target_sr=sr,
                )

            self.assertTrue(output_path.exists())

    def test_karaoke_backing_merge_does_not_pollute_exported_accompaniment(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertIn("pure_accompaniment_path = accompaniment_path", source)
        self.assertIn("mix_accompaniment_path = accompaniment_path", source)
        self.assertIn("mix_accompaniment_path = self._merge_backing_into_accompaniment(", source)
        self.assertIn("accompaniment_path=mix_accompaniment_path", source)
        self.assertIn("shutil.copy(pure_accompaniment_path, final_accompaniment)", source)
        self.assertIn('"accompaniment": pure_accompaniment_path', source)
        self.assertNotRegex(source, r"(?m)^\s*accompaniment_path\s*=\s*self\._merge_backing_into_accompaniment\(")

    def test_ui_and_docs_match_cover_rms_mix_default(self):
        ui_source = (REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn('config.get("rms_mix_rate", 0.0)', ui_source)
        self.assertIn("0.0 (0%)", readme)

    def test_upstream_official_adapter_routes_hybrid_to_conservative_vc_method(self):
        source = (REPO_ROOT / "infer" / "official_adapter.py").read_text(encoding="utf-8")

        self.assertIn("effective_f0_method = f0_policy.vc_method", source)
        self.assertIn("官方F0路由解析", source)


    def test_upstream_official_adapter_retries_without_index_on_subprocess_failure(self):
        source = (REPO_ROOT / "infer" / "official_adapter.py").read_text(encoding="utf-8")

        self.assertIn("should_retry_without_index = bool(official_index) and float(index_rate) > 0.0", source)
        self.assertIn("retry_command = _build_command(None, 0.0)", source)
        self.assertIn("used_index_fallback = True", source)

    def test_cover_official_mode_skips_custom_vc_preprocess_and_source_cleanup(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(encoding="utf-8")

        self.assertIn("get_official_cover_vc_profile", source)
        self.assertIn('effective_preprocess_mode = "direct"', source)
        self.assertIn("vc_preprocessed = False", source)
        self.assertNotIn("官方模式：direct预处理已提升为auto", source)
        self.assertNotIn("官方模式也必须经过去混响预处理", source)

    def test_ui_official_mode_switch_applies_full_profile(self):
        source = (REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8")

        self.assertIn("get_cover_vc_pipeline_profile_updates", source)
        self.assertIn("get_official_cover_vc_profile", source)
        self.assertIn("cover_vc_pipeline_mode.change(", source)
        self.assertIn("cover_index_rate", source)
        self.assertIn("cover_rms_mix_rate", source)
        self.assertIn("cover_karaoke", source)
        self.assertIn("cover_source_constraint_mode", source)


if __name__ == "__main__":
    unittest.main()
