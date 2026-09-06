"""Optional routes must expose implementations, not nominal settings."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch


class OptionalRouteContractTests(unittest.TestCase):
    def test_local_dio_is_rejected_before_inference(self):
        from configs.schema import validate_config
        with self.assertRaises(ValueError):
            validate_config({"cover": {"use_official": False, "f0_method": "dio"}})

    def test_all_public_local_f0_methods_have_a_factory(self):
        from configs.schema import validate_config
        from infer.f0_extractor import get_f0_extractor
        for method in ("rmvpe", "pm", "harvest", "crepe"):
            validate_config({"cover": {"use_official": False, "f0_method": method}})
            self.assertIsNotNone(get_f0_extractor(method, device="cpu", rmvpe_path="rmvpe.pt"))

    def test_demucs_keeps_every_non_vocal_stem(self):
        from infer.separator import VocalSeparator
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            sf.write(source, np.ones((4410, 2), dtype=np.float32) * .1, 44100, subtype="FLOAT")
            separator = VocalSeparator(device="cpu", shifts=2, overlap=.25)
            separator.model = mock.Mock(samplerate=44100, sources=["drums", "bass", "other", "vocals", "guitar", "piano"])
            tracks = torch.stack([torch.full((2, 4410), idx / 100) for idx in range(6)])[None]
            with mock.patch("infer.separator.apply_model", return_value=tracks), mock.patch("infer.separator.torchaudio.load", side_effect=AssertionError("obsolete decoder")):
                vocals, accompaniment = separator.separate(str(source), str(root / "out"))
            self.assertTrue(np.allclose(sf.read(vocals)[0], .03))
            self.assertTrue(np.allclose(sf.read(accompaniment)[0], .12))
            self.assertEqual(sf.info(accompaniment).subtype, "FLOAT")


if __name__ == "__main__":
    unittest.main()
