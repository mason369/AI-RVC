"""Exercise upstream configuration, chunk and file contracts without weights."""
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch
from ml_collections import ConfigDict

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.common_separator import CommonSeparator
from audio_separator.separator.exceptions import AudioExportError, InvalidAudioDataError
from infer import separator as project_separator


def common_config(output_dir, model_data=None):
    return {
        "logger": logging.getLogger("separator_contract_test"),
        "log_level": logging.ERROR,
        "torch_device": torch.device("cpu"),
        "torch_device_cpu": torch.device("cpu"),
        "model_name": "contract",
        "model_path": "contract.ckpt",
        "model_data": model_data or {},
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "sample_rate": 44100,
        "normalization_threshold": .9,
        "amplification_threshold": 0,
        "use_soundfile": True,
    }


class AudioSeparatorUpgradeTests(unittest.TestCase):
    def test_yaml_overlap_and_batch_are_used_and_explicit_override_still_works(self):
        data = {
            "is_roformer": True,
            "inference": {"dim_t": 12, "num_overlap": 4, "batch_size": 6},
            "model": {"stft_hop_length": 4},
            "training": {"target_instrument": "vocals", "instruments": ["vocals", "other"]},
            "audio": {"sample_rate": 44100},
        }

        def load_without_weights(instance):
            instance.model_data_cfgdict = ConfigDict(instance.model_data)

        with patch.object(MDXCSeparator, "load_model", load_without_weights):
            inherited = MDXCSeparator(common_config("unused", data), {"overlap": None, "batch_size": None})
            overridden = MDXCSeparator(common_config("unused", data), {"overlap": 2, "batch_size": 1})
        self.assertEqual((inherited.overlap, inherited.batch_size), (4, 6))
        self.assertEqual((overridden.overlap, overridden.batch_size), (2, 1))

    def test_chunk_schedule_covers_short_exact_and_tail_without_duplicate_windows(self):
        for length, expected in [(5, [0]), (11, [0]), (20, [0, 8, 9]), (99, list(range(0, 89, 8)))]:
            with self.subTest(length=length):
                starts = MDXCSeparator._roformer_chunk_starts(length, 11, 8)
                self.assertEqual(starts, expected)
                covered = np.zeros(length, dtype=bool)
                for start in starts:
                    covered[start:start + 11] = True
                self.assertTrue(covered.all())
                self.assertEqual(len(starts), len(set(starts)))

    def test_silent_and_near_silent_float_stems_are_written_with_exact_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = CommonSeparator(common_config(tmp))
            writer.input_subtype = "FLOAT"
            for value in (0.0, 1e-9):
                samples = np.full((257, 2), value, dtype=np.float32)
                name = f"quiet-{value}.wav"
                writer.write_audio_soundfile(name, samples)
                result, rate = sf.read(Path(tmp) / name, dtype="float32", always_2d=True)
                self.assertEqual(rate, 44100)
                self.assertEqual(sf.info(Path(tmp) / name).subtype, "FLOAT")
                np.testing.assert_array_equal(result, samples)

    def test_export_failure_preserves_previous_output_and_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = CommonSeparator(common_config(tmp))
            writer.input_subtype = "FLOAT"
            target = Path(tmp) / "keep.wav"
            sf.write(target, np.ones((128, 2), dtype=np.float32) * .1, 44100, subtype="FLOAT")
            previous = target.read_bytes()
            with patch("audio_separator.separator.common_separator.sf.write", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(AudioExportError, "disk failure"):
                    writer.write_audio_soundfile(target.name, np.zeros((128, 2), dtype=np.float32))
            self.assertEqual(target.read_bytes(), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [target])

    def test_nonfinite_audio_is_rejected_without_an_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = CommonSeparator(common_config(tmp))
            writer.input_subtype = "FLOAT"
            with self.assertRaises(InvalidAudioDataError):
                writer.write_audio_soundfile("bad.wav", np.full((32, 2), np.nan, dtype=np.float32))
            self.assertFalse((Path(tmp) / "bad.wav").exists())

    def test_directml_custom_roformer_stops_before_download_or_cpu_substitution(self):
        original_download = Mock(side_effect=AssertionError("must not download"))
        instance = SimpleNamespace(torch_device="privateuseone:0", download_model_files=original_download)
        project_separator._install_custom_audio_separator_models(instance)
        with self.assertRaisesRegex(RuntimeError, "DirectML"):
            instance.download_model_files(project_separator.LEAP_XE_VOCALS_MODEL)
        original_download.assert_not_called()

    def test_directml_registry_mdxc_is_rejected_and_other_architectures_pass(self):
        for architecture in ("MDXC", "MDX", "VR"):
            details = ("file", architecture, "model", "file", "config")
            instance = SimpleNamespace(torch_device="privateuseone:0", download_model_files=Mock(return_value=details))
            project_separator._install_custom_audio_separator_models(instance)
            if architecture == "MDXC":
                with self.assertRaisesRegex(RuntimeError, "DirectML"):
                    instance.download_model_files("registry-model")
            else:
                self.assertEqual(instance.download_model_files("registry-model"), details)

    def test_backend_guard_accepts_cpu_cuda_and_other_requested_torch_devices(self):
        for device in ("cpu", "cuda:0", "xpu:0", "mps"):
            project_separator._require_separator_model_backend(SimpleNamespace(torch_device=device), "MDXC")

    def test_packaging_and_notebook_keep_the_same_separator_contract(self):
        import json
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "AI_RVC_Colab.ipynb").read_text(encoding="utf-8"))
        self.assertTrue(any("'audio-separator': '==0.47.0'" in "".join(cell.get("source", [])) for cell in notebook["cells"]))
        workflow = (root / ".github/workflows/build-executables.yml").read_text(encoding="utf-8")
        self.assertIn('audio-separator[${{ matrix.audio_separator_extra }}]==0.47.0', workflow)
        self.assertIn('--collect-submodules audio_separator.separator', workflow)
        self.assertIn('--collect-data audio_separator', workflow)
        self.assertIn('--copy-metadata audio-separator', workflow)
        spec = (root / "AI-RVC.spec").read_text(encoding="utf-8")
        compile(spec, 'AI-RVC.spec', 'exec')
        self.assertIn("collect_submodules('audio_separator.separator')", spec)
        self.assertIn("copy_metadata('audio-separator')", spec)
        ci = (root / ".github/workflows/platform-contracts.yml").read_text(encoding="utf-8")
        self.assertIn('"audio-separator[cpu]==0.47.0"', ci)
        self.assertIn('"test_audio_separator_upgrade.py"', ci)
        self.assertIn('brew install libsamplerate', ci)
        self.assertIn('torch==2.13.0 torchvision==0.28.0 torchaudio==2.13.0', ci)


if __name__ == "__main__":
    unittest.main()
