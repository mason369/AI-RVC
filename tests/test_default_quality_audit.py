import unittest
from pathlib import Path


class DefaultQualityAuditTests(unittest.TestCase):
    def test_manifest_rejects_hidden_parameter_overrides(self):
        from tools.default_quality_audit import validate_manifest_schema

        manifest = {
            "cases": [
                {
                    "name": "high_note_rin",
                    "input_audio": "samples/high.wav",
                    "model_path": "assets/weights/characters/rin/Rin.pth",
                    "role_id": "rin",
                    "tags": ["high_note"],
                    "pitch_shift": 3,
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "default parameters"):
            validate_manifest_schema(manifest)

    def test_coverage_requires_song_and_role_matrix(self):
        from tools.default_quality_audit import analyze_manifest_coverage

        manifest = {
            "cases": [
                {"name": "high", "input_audio": "a.wav", "model_path": "m1.pth", "role_id": "role_a", "tags": ["high_note"]},
                {"name": "low", "input_audio": "b.wav", "model_path": "m2.pth", "role_id": "role_b", "tags": ["low_note"]},
            ]
        }

        coverage = analyze_manifest_coverage(manifest)

        self.assertFalse(coverage["met"])
        self.assertIn("fast_song", coverage["missing_song_tags"])
        self.assertIn("slow_song", coverage["missing_song_tags"])
        self.assertLess(coverage["role_count"], coverage["required_role_count"])

    def test_metrics_can_block_but_cannot_pass_without_listening_review(self):
        from tools.default_quality_audit import build_quality_verdict

        report = build_quality_verdict(
            manifest={
                "cases": [
                    {
                        "name": "high",
                        "input_audio": "a.wav",
                        "model_path": "m1.pth",
                        "role_id": "role_a",
                        "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                    },
                    {
                        "name": "role_b",
                        "input_audio": "b.wav",
                        "model_path": "m2.pth",
                        "role_id": "role_b",
                        "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                    },
                    {
                        "name": "role_c",
                        "input_audio": "c.wav",
                        "model_path": "m3.pth",
                        "role_id": "role_c",
                        "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                    },
                    {
                        "name": "role_a_2",
                        "input_audio": "d.wav",
                        "model_path": "m1.pth",
                        "role_id": "role_a",
                        "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                    },
                    {
                        "name": "role_b_2",
                        "input_audio": "e.wav",
                        "model_path": "m2.pth",
                        "role_id": "role_b",
                        "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                    },
                    {
                        "name": "role_c_2",
                        "input_audio": "f.wav",
                        "model_path": "m3.pth",
                        "role_id": "role_c",
                        "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                    },
                    {
                        "name": "role_a_3",
                        "input_audio": "g.wav",
                        "model_path": "m1.pth",
                        "role_id": "role_a",
                        "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                    },
                ]
            },
            technical_results={
                "high": {
                    "quality_debug": {
                        "peak": 0.83,
                        "clip_ratio": 0.0,
                        "quiet_rms_ratio": 1.2,
                        "quiet_hf_ratio": 1.1,
                        "transition_spike_ratio": 1.0,
                        "synthetic_breath_frames": 0,
                    }
                }
            },
            listening_reviews={},
        )

        self.assertEqual(report["verdict"], "needs_listening_review")
        self.assertIn("cannot pass", report["quality_principle"].lower())

    def test_single_case_alert_blocks_even_when_other_cases_are_clean(self):
        from tools.default_quality_audit import build_quality_verdict

        manifest = {
            "cases": [
                {
                    "name": f"case_{i}",
                    "input_audio": f"{i}.wav",
                    "model_path": f"m{i % 3}.pth",
                    "role_id": f"role_{i % 3}",
                    "tags": ["high_note", "low_note", "fast_song", "slow_song", "breathy", "long_tail", "complex_harmony"],
                }
                for i in range(7)
            ]
        }
        reviews = {
            f"case_{i}": {
                "default_one_click_quality": 5,
                "character_identity": 5,
                "source_distinctness": 5,
                "artifact_absence": 5,
                "lyrics_and_performance": 5,
                "mix_integration": 5,
            }
            for i in range(7)
        }
        technical = {
            f"case_{i}": {
                "quality_debug": {
                    "peak": 0.80,
                    "clip_ratio": 0.0,
                    "quiet_rms_ratio": 1.1,
                    "quiet_hf_ratio": 1.1,
                    "transition_spike_ratio": 1.0,
                    "synthetic_breath_frames": 0,
                }
            }
            for i in range(7)
        }
        technical["case_3"]["quality_debug"]["quiet_hf_ratio"] = 3.2

        report = build_quality_verdict(manifest, technical, reviews)

        self.assertEqual(report["verdict"], "blocked_by_technical_alerts")
        self.assertIn("case_3", report["blocking_cases"])
        self.assertNotIn("average", report["quality_principle"].lower())

    def test_default_cover_kwargs_use_config_values_only(self):
        from tools.default_quality_audit import build_default_cover_kwargs

        config = {
            "cover": {
                "separator": "roformer",
                "roformer_model": "ensemble:vocal_rvc",
                "karaoke_separation": True,
                "karaoke_model": "ensemble:karaoke",
                "karaoke_merge_backing_into_accompaniment": True,
                "uvr5_model": "HP2_all_vocals",
                "uvr5_agg": 10,
                "uvr5_format": "wav",
                "use_official": True,
                "demucs_model": "htdemucs_ft",
                "demucs_shifts": 10,
                "demucs_overlap": 0.5,
                "demucs_split": True,
                "f0_method": "rmvpe",
                "f0_hybrid_mode": "off",
                "index_rate": 0.5,
                "filter_radius": 3,
                "rms_mix_rate": 0.0,
                "protect": 0.33,
                "speaker_id": 0,
                "hubert_layer": 12,
                "default_vocals_volume": 100,
                "default_accompaniment_volume": 100,
                "default_reverb": 0,
                "backing_mix": 0.0,
                "vc_preprocess_mode": "auto",
                "source_constraint_mode": "auto",
                "vc_pipeline_mode": "current",
            }
        }
        case = {
            "name": "sample",
            "input_audio": "song.wav",
            "model_path": "model.pth",
            "index_path": "model.index",
            "role_id": "role",
            "tags": ["high_note"],
        }

        kwargs = build_default_cover_kwargs(case, config, Path("out"))

        self.assertEqual(kwargs["pitch_shift"], 0)
        self.assertEqual(kwargs["index_ratio"], 0.5)
        self.assertEqual(kwargs["f0_method"], "rmvpe")
        self.assertEqual(kwargs["vc_preprocess_mode"], "auto")
        self.assertEqual(kwargs["source_constraint_mode"], "auto")
        self.assertEqual(kwargs["vc_pipeline_mode"], "current")
        self.assertNotIn("singing_repair", kwargs)
        self.assertEqual(kwargs["vocals_volume"], 1.0)
        self.assertEqual(kwargs["accompaniment_volume"], 1.0)
        self.assertEqual(kwargs["reverb_amount"], 0.0)


if __name__ == "__main__":
    unittest.main()
