import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


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
compute_active_source_replace = quality_policy.compute_active_source_replace
compute_source_cleanup_budget = quality_policy.compute_source_cleanup_budget
resolve_cover_f0_policy = quality_policy.resolve_cover_f0_policy


class F0RoutingPolicyTests(unittest.TestCase):
    def test_hybrid_routes_to_conservative_rmvpe_fallback(self):
        policy = resolve_cover_f0_policy("hybrid", "off")

        self.assertEqual(policy.requested_method, "hybrid")
        self.assertEqual(policy.vc_method, "rmvpe")
        self.assertEqual(policy.hybrid_mode, "fallback")
        self.assertEqual(policy.gate_method, "rmvpe")


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


class SourceConstraintPolicyTests(unittest.TestCase):
    def test_active_echo_frames_keep_nonzero_replace_pressure(self):
        activity = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        soft_mask = np.array([[0.10, 0.55, 0.10]], dtype=np.float32)
        echo_ratio = np.array([[0.90, 0.85, 0.40]], dtype=np.float32)
        direct_ratio = np.array([0.10, 0.25, 0.15], dtype=np.float32)

        replace = compute_active_source_replace(activity, soft_mask, echo_ratio, direct_ratio)

        self.assertGreater(replace[0, 0], 0.25)
        self.assertGreater(replace[0, 1], 0.10)
        self.assertLess(replace[0, 1], replace[0, 0])
        self.assertLessEqual(float(np.max(replace)), 0.70)

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

    def test_official_vc_pipeline_no_longer_forces_hybrid_to_aggressive_crepe(self):
        source = (REPO_ROOT / "infer" / "modules" / "vc" / "pipeline.py").read_text(encoding="utf-8")

        self.assertNotIn('self.f0_hybrid_mode = "rmvpe+crepe"', source)

    def test_config_defaults_match_conservative_hybrid_policy(self):
        source = (REPO_ROOT / "configs" / "config.json").read_text(encoding="utf-8")

        self.assertRegex(source, r'"f0_hybrid_mode"\s*:\s*"fallback"')
        self.assertNotRegex(source, r'"crepe_force_ratio"\s*:\s*0\.0')


if __name__ == "__main__":
    unittest.main()
