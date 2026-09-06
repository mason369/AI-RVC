import tempfile
import unittest
from pathlib import Path

from infer.separator import RoformerDereverbSeparator, _resolve_output_files


class SeparatorOutputResolutionTests(unittest.TestCase):
    def test_resolves_existing_file_when_returned_name_has_stale_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            actual = output_dir / "vocal_song.wav_10_(noreverb)_dereverb.wav"
            actual.write_bytes(b"audio")

            resolved = _resolve_output_files(
                ["vocal_song.wav_10_(noreverb)_dereverb.wav"],
                output_dir,
            )

        self.assertEqual([Path(path).name for path in resolved], [actual.name])

    def test_missing_output_is_not_replaced_by_existing_same_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old_(noreverb).wav"
            old.write_bytes(b"previous result")
            with self.assertRaisesRegex(RuntimeError, "不存在或为空"):
                _resolve_output_files(["new_(noreverb).wav"], root)
            self.assertEqual(old.read_bytes(), b"previous result")

    def test_empty_directory_and_duplicate_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.wav"
            valid.write_bytes(b"audio")
            (root / "empty.wav").touch()
            for paths in (["empty.wav"], [str(root)], ["valid.wav", str(valid)]):
                with self.subTest(paths=paths), self.assertRaises(RuntimeError):
                    _resolve_output_files(paths, root)

    def test_classifies_noreverb_as_dry_before_reverb(self):
        self.assertEqual(
            RoformerDereverbSeparator._classify_stem(
                "vocal_song.wav_10_(noreverb)_dereverb_mel_band_roformer.wav"
            ),
            "dry",
        )

    def test_classifies_reverb_as_wet(self):
        self.assertEqual(
            RoformerDereverbSeparator._classify_stem(
                "vocal_song.wav_10_(reverb)_dereverb_mel_band_roformer.wav"
            ),
            "wet",
        )

    def test_dereverb_does_not_accept_wet_output_as_dry(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            wet = output_dir / "song_(reverb)_dereverb.wav"
            wet.write_bytes(b"audio")
            import numpy as np
            import soundfile as sf
            source = output_dir / "input.wav"
            sf.write(source, np.ones((4410, 2), dtype=np.float32) * .1, 44100, subtype="FLOAT")

            separator = RoformerDereverbSeparator.__new__(RoformerDereverbSeparator)
            separator.model_filename = "legacy_test_model.ckpt"
            separator.load_model = lambda output_dir="": None
            separator.separator = type(
                "FakeSeparator",
                (),
                {
                    "output_dir": str(output_dir),
                    "separate": lambda self, audio_path: [str(wet)],
                },
            )()

            with self.assertRaisesRegex(FileNotFoundError, "dry轨未找到"):
                separator.separate_dry(str(source), str(output_dir))


if __name__ == "__main__":
    unittest.main()
