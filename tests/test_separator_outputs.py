import tempfile
import unittest
from pathlib import Path

from infer.separator import RoformerDereverbSeparator, _resolve_output_files


class SeparatorOutputResolutionTests(unittest.TestCase):
    def test_resolves_existing_file_when_returned_name_has_stale_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            actual = output_dir / "song_(noreverb)_dereverb.wav"
            actual.write_bytes(b"audio")

            resolved = _resolve_output_files(
                ["vocal_song.wav_10_(noreverb)_dereverb.wav"],
                output_dir,
            )

        self.assertEqual([Path(path).name for path in resolved], [actual.name])

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

            separator = RoformerDereverbSeparator.__new__(RoformerDereverbSeparator)
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
                separator.separate_dry("input.wav", str(output_dir))


if __name__ == "__main__":
    unittest.main()
