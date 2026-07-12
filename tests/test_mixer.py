import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from lib.mixer import _probe_sample_rate, mix_vocals_and_accompaniment


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


class MixStabilityTests(unittest.TestCase):
    def test_sample_rate_probe_failure_is_not_replaced_with_a_default(self):
        with mock.patch("lib.mixer.sf.info", side_effect=RuntimeError("broken metadata")):
            with self.assertRaisesRegex(RuntimeError, "无法读取混音输入的采样率"):
                _probe_sample_rate("broken.wav")

    def test_default_mix_does_not_duck_accompaniment_when_vocal_enters(self):
        sr = 48000
        duration = 3.0
        t = np.arange(int(sr * duration), dtype=np.float32) / sr
        accompaniment = 0.07 * np.sin(2.0 * np.pi * 220.0 * t)
        vocals = np.zeros_like(accompaniment)
        active = (t >= 1.10) & (t < 2.00)
        vocals[active] = 0.045 * np.sin(2.0 * np.pi * 440.0 * t[active])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            vocals_path = tmp / "vocals.wav"
            accompaniment_path = tmp / "accompaniment.wav"
            output_path = tmp / "mix.wav"
            sf.write(vocals_path, vocals, sr)
            sf.write(accompaniment_path, accompaniment, sr)

            mix_vocals_and_accompaniment(
                str(vocals_path),
                str(accompaniment_path),
                str(output_path),
                target_sr=sr,
            )

            mixed, out_sr = sf.read(output_path)

        self.assertEqual(out_sr, sr)
        if mixed.ndim > 1:
            mixed = mixed.mean(axis=1)
        residual = mixed.astype(np.float32) - vocals.astype(np.float32)
        before = residual[int(0.40 * sr): int(0.90 * sr)]
        during = residual[int(1.30 * sr): int(1.80 * sr)]
        drop_db = 20.0 * np.log10((_rms(during) + 1e-12) / (_rms(before) + 1e-12))

        self.assertGreater(
            drop_db,
            -0.25,
            f"default mix should keep accompaniment steady, got {drop_db:.2f} dB drop",
        )


if __name__ == "__main__":
    unittest.main()
