import inspect
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ModelDefaultTests(unittest.TestCase):
    def test_explicit_cuda_device_does_not_fall_back_to_cpu(self):
        from lib import device

        with (
            patch.object(device.torch.cuda, "is_available", return_value=False),
            patch.object(device, "_has_xpu", return_value=False),
            patch.object(device, "_has_directml", return_value=False),
            patch.object(device, "_has_mps", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "已选择 CUDA"):
                device.get_device("cuda")
            self.assertEqual(device.get_device("auto").type, "cpu")

    def test_roformer_default_uses_public_deployable_hybrid_sota_route(self):
        from infer import separator

        self.assertEqual(
            separator.ROFORMER_DEFAULT_MODEL,
            "hybrid:leap_xe90_vocals+leap62_instrumental",
        )
        self.assertEqual(
            separator.ROFORMER_SOTA_MODELS,
            [
                "bs_roformer_leap_xe_voc.ckpt",
                "Xe/leap_xe_config_voc.yaml",
                "Xe/bs_roformer_leap_xe_config_voc.yaml",
                "leap_instrumental/bs_roformer_leap_inst.ckpt",
                "leap_instrumental/bs_leap_inst_conf.yaml",
                "leap_instrumental/bs_roformer_leap_inst_runtime.yaml",
            ],
        )
        self.assertIn(
            "vocals_mel_band_roformer.ckpt",
            separator.ROFORMER_LEGACY_SINGLE_MODEL,
        )

    def test_karaoke_default_uses_public_sota_ensemble(self):
        from infer import separator

        self.assertEqual(
            separator.KARAOKE_DEFAULT_MODEL,
            "ensemble:mvsep_9205_avg",
        )
        self.assertEqual(
            separator.KARAOKE_SOTA_MODEL,
            "ensemble:mvsep_9205_avg",
        )
        self.assertEqual(
            separator.KARAOKE_SOTA_MODELS,
            [
                "bs_karaoke_gabox_IS.ckpt",
                "bs_roformer_karaoke_frazer_becruily.ckpt",
                "karaoke_bs_roformer_anvuew.ckpt",
            ],
        )
        self.assertEqual(
            separator.KARAOKE_LEGACY_SINGLE_MODEL,
            "mel_band_roformer_karaoke_gabox.ckpt",
        )

    def test_telknet_separator_chain_uses_exact_public_model_names(self):
        from infer import separator

        self.assertEqual(
            separator._model_spec_label("hybrid:leap_xe90_vocals+leap62_instrumental"),
            (
                "BS-RoFormer Leap XE 90 bands (pcunwa) + "
                "BS-RoFormer Leap Instrumental 62 bands (pcunwa)"
            ),
        )
        self.assertEqual(
            separator._model_spec_label(separator.KARAOKE_DEFAULT_MODEL),
            (
                "BS-Kar-Gabox_IS + BS-Kar-Frazer&Becruily + "
                "BS-Kar-Anvuew (AVG)"
            ),
        )
        self.assertEqual(
            separator.get_separator_chain_labels(
                separator_name="roformer",
                roformer_model="hybrid:leap_xe90_vocals+leap62_instrumental",
                karaoke_enabled=True,
                karaoke_model=separator.KARAOKE_DEFAULT_MODEL,
            ),
            [
                "输入: 两路 Leap 共用 44.1kHz 双声道 PCM（非WAV预解码）",
                "人声: BS-RoFormer Leap XE 90 bands (pcunwa)",
                (
                    "纯伴奏: BS-RoFormer Leap Instrumental 62 bands (pcunwa)"
                ),
                (
                    "主唱/带和声伴奏: BS-Kar-Gabox_IS + "
                    "BS-Kar-Frazer&Becruily + BS-Kar-Anvuew (AVG)"
                ),
                "纯和声: Leap 人声 - MVSep 主唱",
            ],
        )

    def test_cover_pipeline_logs_telknet_chain_before_separation(self):
        from infer.cover_pipeline import CoverPipeline

        source = inspect.getsource(CoverPipeline.process)
        manifest_position = source.index("本次实际分离链路")
        separation_position = source.index(
            '"正在分离人声和纯伴奏..."'
        )

        self.assertIn("get_separator_chain_labels", source)
        self.assertLess(manifest_position, separation_position)

    def test_mvsep_9205_custom_preset_is_explicit(self):
        from infer import separator

        preset = separator._CUSTOM_ENSEMBLE_PRESETS["mvsep_9205_avg"]

        self.assertEqual(preset["algorithm"], "avg_wave")
        self.assertEqual(preset["models"], separator.KARAOKE_SOTA_MODELS)
        self.assertTrue(
            set(separator.KARAOKE_SOTA_MODELS).issubset(
                set(separator._CUSTOM_AUDIO_SEPARATOR_MODELS)
            )
        )






    def test_audio_separator_runtime_honors_explicit_cpu_selection(self):
        from infer import separator

        runtime = type("Runtime", (), {})()
        runtime.torch_device = separator.torch.device("cuda")
        runtime.onnx_execution_provider = ["CUDAExecutionProvider"]

        separator._configure_audio_separator_runtime(runtime, "cpu")

        self.assertEqual(str(runtime.torch_device), "cpu")
        self.assertEqual(runtime.onnx_execution_provider, ["CPUExecutionProvider"])

    def test_audio_separator_runtime_routes_non_onnx_accelerators_to_cpu_ort(self):
        from infer import separator
        import onnxruntime

        class Probe:
            @staticmethod
            def to(_device):
                return Probe()

        for device in ("xpu", "mps"):
            runtime = type("Runtime", (), {})()
            with self.subTest(device=device), patch.object(
                separator.torch,
                "zeros",
                return_value=Probe(),
            ), patch.object(
                onnxruntime,
                "get_available_providers",
                return_value=["CPUExecutionProvider"],
            ):
                separator._configure_audio_separator_runtime(runtime, device)

            self.assertEqual(str(runtime.torch_device), device)
            self.assertEqual(
                runtime.onnx_execution_provider,
                ["CPUExecutionProvider"],
            )



    def test_deecho_default_uses_public_roformer_dereverb_model(self):
        from infer import separator

        self.assertEqual(
            separator.ROFORMER_DEREVERB_DEFAULT_MODEL,
            "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
        )

    def test_download_models_tracks_default_separator_assets(self):
        from tools import download_models

        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_paths = download_models.get_default_separator_asset_paths(Path(tmp_dir))
            flat_paths = [
                str(path).replace("\\", "/")
                for paths in asset_paths.values()
                for path in paths
            ]

            self.assertIn(
                "assets/separator_models/Xe/bs_leap_xe_voc.ckpt",
                "\n".join(flat_paths),
            )
            self.assertIn(
                "assets/separator_models/Xe/bs_roformer_leap_xe_config_voc.yaml",
                "\n".join(flat_paths),
            )
            self.assertIn(
                "assets/separator_models/leap_instrumental/bs_roformer_leap_inst.ckpt",
                "\n".join(flat_paths),
            )
            self.assertIn(
                "assets/separator_models/bsroformers/bs_karaoke_gabox_IS.ckpt",
                "\n".join(flat_paths),
            )
            self.assertIn(
                "assets/separator_models/dereverb_stereo/dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
                "\n".join(flat_paths),
            )
            self.assertFalse(
                download_models.check_required_default_separator_models(Path(tmp_dir))
            )

    def test_leap_runtime_config_declares_bs_roformer_model_type(self):
        import yaml

        from infer import separator

        class FakeSeparator:
            def __init__(self, model_file_dir: str):
                self.model_file_dir = model_file_dir

            @staticmethod
            def download_model_files(_model_filename):
                raise AssertionError("custom Leap model must not use the default downloader")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            upstream_model = tmp_path / "upstream.ckpt"
            upstream_model.write_bytes(b"model")
            upstream_config = tmp_path / "upstream.yaml"
            upstream_config.write_text(
                "audio:\n  sample_rate: 44100\nmodel:\n  freqs_per_bands: [2, 2]\n",
                encoding="utf-8",
            )

            def fake_download(_repo_id, filename, _model_dir, *_pins):
                if str(filename).endswith(".ckpt"):
                    return str(upstream_model)
                return str(upstream_config)

            fake_separator = FakeSeparator(str(tmp_path / "models"))
            with patch("tools.model_assets.download_pinned_asset", side_effect=fake_download):
                separator._install_custom_audio_separator_models(fake_separator)
                result = fake_separator.download_model_files(
                    separator.LEAP_XE_VOCALS_MODEL
                )

            runtime_config = Path(result[4])
            runtime_data = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
            self.assertEqual(runtime_data["model_type"], "bs_roformer")




    def test_mvsep_9205_runtime_pads_short_input_and_trims_outputs(self):
        from infer import separator

        class FakeKaraokeEnsemble:
            def __init__(self):
                self.output_dir = ""
                self.last_audio_path = None

            def separate(self, audio_path: str):
                self.last_audio_path = audio_path
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                data, sample_rate = sf.read(audio_path, always_2d=True)
                lead = out_dir / "song_(Vocals)_mvsep.wav"
                backing = out_dir / "song_(Instrumental)_mvsep.wav"
                sf.write(lead, data, sample_rate)
                sf.write(backing, data * 0.25, sample_rate)
                return [str(lead), str(backing)]

        original_min_duration = getattr(
            separator,
            "_get_karaoke_min_duration_seconds",
            None,
        )
        separator._get_karaoke_min_duration_seconds = lambda model_dir, model_spec: 2.0
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                audio_path = tmp_path / "short.wav"
                sf.write(audio_path, np.ones(8000, dtype=np.float32), 16000)

                karaoke = separator.KaraokeSeparator(
                    model_filename=separator.KARAOKE_DEFAULT_MODEL,
                    device="cpu",
                )
                fake_ensemble = FakeKaraokeEnsemble()
                karaoke.separator = fake_ensemble
                karaoke._init_output_dir = str(tmp_path / "out")
                karaoke.load_model = lambda output_dir="": None

                lead_path, backing_path = karaoke.separate(
                    str(audio_path),
                    str(tmp_path / "out"),
                )

                self.assertNotEqual(fake_ensemble.last_audio_path, str(audio_path))
                self.assertGreaterEqual(
                    sf.info(fake_ensemble.last_audio_path).frames,
                    32001,
                )
                self.assertEqual(sf.info(lead_path).samplerate, 44100)
                self.assertEqual(sf.info(lead_path).frames, 22050)
                self.assertEqual(sf.info(backing_path).frames, 22050)
                self.assertEqual(Path(lead_path).name, "lead_vocals.wav")
                self.assertEqual(Path(backing_path).name, "accompaniment.wav")
        finally:
            if original_min_duration is None:
                delattr(separator, "_get_karaoke_min_duration_seconds")
            else:
                separator._get_karaoke_min_duration_seconds = original_min_duration

    def test_mvsep_9205_min_duration_matches_audio_separator_roformer_window(self):
        from infer import separator

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            for index, model_name in enumerate(separator.KARAOKE_SOTA_MODELS, start=1):
                config_name = separator._CUSTOM_AUDIO_SEPARATOR_MODELS[model_name][
                    "config_filename"
                ]
                config_path = model_dir / config_name
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    "\n".join(
                        [
                            "audio:",
                            "  chunk_size: 100",
                            "  sample_rate: 10",
                            "model:",
                            f"  stft_hop_length: {index + 1}",
                            "inference:",
                            f"  dim_t: {index + 4}",
                        ]
                    ),
                    encoding="utf-8",
                )

            duration = separator._get_karaoke_min_duration_seconds(
                str(model_dir),
                separator.KARAOKE_DEFAULT_MODEL,
            )

            self.assertEqual(duration, 2.4)

    def test_mvsep_9205_karaoke_uses_original_full_mix(self):
        from infer.cover_pipeline import CoverPipeline
        from infer.separator import KARAOKE_DEFAULT_MODEL

        class FakeKaraokeSeparator:
            def __init__(self):
                self.received_audio_path = None

            def separate(self, audio_path: str, output_dir: str):
                self.received_audio_path = audio_path
                return str(Path(output_dir) / "lead.wav"), str(Path(output_dir) / "backing.wav")

            def unload_model(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pipeline = CoverPipeline(device="cpu")
            fake_separator = FakeKaraokeSeparator()
            pipeline.karaoke_separator = fake_separator
            pipeline._init_karaoke_separator = lambda model_name: None

            pipeline._separate_karaoke(
                source_audio_path=str(tmp_path / "original_mix.wav"),
                session_dir=tmp_path,
                karaoke_model=KARAOKE_DEFAULT_MODEL,
            )

            self.assertEqual(
                fake_separator.received_audio_path,
                str(tmp_path / "original_mix.wav"),
            )

    def test_legacy_karaoke_model_also_uses_original_full_mix(self):
        from infer.cover_pipeline import CoverPipeline
        from infer.separator import KARAOKE_LEGACY_SINGLE_MODEL

        class FakeKaraokeSeparator:
            def __init__(self):
                self.received_audio_path = None

            def separate(self, audio_path: str, output_dir: str):
                self.received_audio_path = audio_path
                return str(Path(output_dir) / "lead.wav"), str(Path(output_dir) / "backing.wav")

            def unload_model(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pipeline = CoverPipeline(device="cpu")
            fake_separator = FakeKaraokeSeparator()
            pipeline.karaoke_separator = fake_separator
            pipeline._init_karaoke_separator = lambda model_name: None

            pipeline._separate_karaoke(
                source_audio_path=str(tmp_path / "original_mix.wav"),
                session_dir=tmp_path,
                karaoke_model=KARAOKE_LEGACY_SINGLE_MODEL,
            )

            self.assertEqual(
                fake_separator.received_audio_path,
                str(tmp_path / "original_mix.wav"),
            )

    def test_primary_separator_is_unloaded_before_next_gpu_stage(self):
        from infer.cover_pipeline import CoverPipeline

        class FakePrimarySeparator:
            def __init__(self):
                self.unloaded = False

            def unload_model(self):
                self.unloaded = True

        pipeline = CoverPipeline(device="cpu")
        primary_separator = FakePrimarySeparator()
        pipeline.separator = primary_separator

        pipeline._release_primary_separator()

        self.assertTrue(primary_separator.unloaded)
        self.assertIsNone(pipeline.separator)

    def test_strict_sota_defaults_do_not_expose_model_fallback_lists(self):
        from infer import separator

        self.assertFalse(hasattr(separator, "ROFORMER_FALLBACK_MODELS"))
        self.assertFalse(hasattr(separator, "KARAOKE_FALLBACK_MODELS"))
        self.assertFalse(hasattr(separator, "ROFORMER_DEREVERB_FALLBACK_MODELS"))

    def test_separator_import_survives_missing_audio_separator(self):
        script = """
import importlib.abc
import sys

class BlockAudioSeparator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "audio_separator" or fullname.startswith("audio_separator."):
            raise ImportError("blocked audio_separator")
        return None

sys.meta_path.insert(0, BlockAudioSeparator())
from infer import separator

assert separator.AUDIO_SEPARATOR_AVAILABLE is False
try:
    separator.RoformerSeparator()
except ImportError as exc:
    assert "audio-separator" in str(exc)
else:
    raise AssertionError("RoformerSeparator should fail when audio_separator is missing")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


class KaraokeCandidateScoringTests(unittest.TestCase):
    def test_karaoke_candidate_score_rewards_reconstruction_and_low_correlation(self):
        from tools.evaluate_karaoke_models import score_karaoke_stems

        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        lead_good = 0.18 * np.sin(2 * np.pi * 220 * t)
        backing_good = 0.05 * np.sin(2 * np.pi * 330 * t + 0.4)
        input_vocals = lead_good + backing_good

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.wav"
            lead_good_path = tmp_path / "lead_good.wav"
            backing_good_path = tmp_path / "backing_good.wav"
            lead_bad_path = tmp_path / "lead_bad.wav"
            backing_bad_path = tmp_path / "backing_bad.wav"

            sf.write(input_path, input_vocals, sr)
            sf.write(lead_good_path, lead_good, sr)
            sf.write(backing_good_path, backing_good, sr)
            sf.write(lead_bad_path, input_vocals, sr)
            sf.write(backing_bad_path, 0.7 * input_vocals, sr)

            good = score_karaoke_stems(input_path, lead_good_path, backing_good_path)
            bad = score_karaoke_stems(input_path, lead_bad_path, backing_bad_path)

        self.assertGreater(good["score"], bad["score"])
        self.assertLess(good["reconstruction_error"], bad["reconstruction_error"])
        self.assertLess(good["lead_backing_abs_corr"], bad["lead_backing_abs_corr"])

    def test_karaoke_candidate_score_penalizes_truncated_stems(self):
        from tools.evaluate_karaoke_models import score_karaoke_stems

        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        lead_good = 0.18 * np.sin(2 * np.pi * 220 * t)
        backing_good = 0.04 * np.sin(2 * np.pi * 330 * t + 0.4)
        input_vocals = lead_good + backing_good
        short_len = sr // 4

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.wav"
            lead_short_path = tmp_path / "lead_short.wav"
            backing_short_path = tmp_path / "backing_short.wav"
            lead_full_path = tmp_path / "lead_full.wav"
            backing_full_path = tmp_path / "backing_full.wav"

            sf.write(input_path, input_vocals, sr)
            sf.write(lead_short_path, lead_good[:short_len], sr)
            sf.write(backing_short_path, backing_good[:short_len], sr)
            sf.write(lead_full_path, 0.97 * lead_good, sr)
            sf.write(backing_full_path, 0.97 * backing_good, sr)

            short = score_karaoke_stems(input_path, lead_short_path, backing_short_path)
            full = score_karaoke_stems(input_path, lead_full_path, backing_full_path)

        self.assertIn("length_coverage", short)
        self.assertLess(short["length_coverage"], 0.999)
        self.assertGreaterEqual(full["length_coverage"], 0.999)
        self.assertGreater(full["score"], short["score"])

    def test_reference_karaoke_score_uses_true_si_sdr_when_refs_exist(self):
        from tools.evaluate_karaoke_models import score_reference_stems

        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        lead = 0.18 * np.sin(2 * np.pi * 220 * t)
        backing = 0.04 * np.sin(2 * np.pi * 330 * t + 0.4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            reference_lead_path = tmp_path / "reference_lead.wav"
            reference_backing_path = tmp_path / "reference_backing.wav"
            lead_path = tmp_path / "lead.wav"
            backing_path = tmp_path / "backing.wav"

            sf.write(reference_lead_path, lead, sr)
            sf.write(reference_backing_path, backing, sr)
            sf.write(lead_path, lead, sr)
            sf.write(backing_path, backing, sr)

            metrics = score_reference_stems(
                reference_lead_path,
                reference_backing_path,
                lead_path,
                backing_path,
            )

        self.assertGreater(metrics["mean_si_sdr"], 100.0)
        self.assertIn("lead", metrics["stems"])
        self.assertIn("backing", metrics["stems"])


if __name__ == "__main__":
    unittest.main()
