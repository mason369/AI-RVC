import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from infer.cover_pipeline import CoverPipeline


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _preemphasis_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    residual = np.empty_like(audio)
    residual[0] = audio[0]
    residual[1:] = audio[1:] - 0.97 * audio[:-1]
    return _rms(residual)


def _fade(length: int) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    ramp = max(1, min(length // 4, 512))
    env[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
    return env


class EchoTailGateTests(unittest.TestCase):
    def test_loud_echo_tail_is_suppressed_when_dereverb_removes_it(self):
        sr = 48000
        t = np.arange(int(2.4 * sr), dtype=np.float32) / sr
        original = np.zeros_like(t)
        dereverbed = np.zeros_like(t)

        direct_mask = (t >= 0.20) & (t < 0.72)
        echo_mask = (t >= 1.16) & (t < 1.68)
        direct = 0.24 * np.sin(2.0 * np.pi * 330.0 * t[direct_mask])
        echo = 0.12 * np.sin(2.0 * np.pi * 330.0 * (t[echo_mask] - 0.96))
        direct *= _fade(direct.size)
        echo *= _fade(echo.size)
        original[direct_mask] = direct
        original[echo_mask] = echo
        dereverbed[direct_mask] = direct

        gain, gated_frames, total_frames = CoverPipeline._compute_echo_tail_sample_gain(
            original=original,
            dereverbed=dereverbed,
            sr=sr,
        )

        direct_gain = gain[int(0.34 * sr): int(0.58 * sr)]
        echo_gain = gain[int(1.28 * sr): int(1.54 * sr)]
        self.assertGreater(total_frames, 0)
        self.assertGreater(gated_frames, 0)
        self.assertGreater(float(np.percentile(direct_gain, 5)), 0.85)
        self.assertLess(
            float(np.mean(echo_gain)),
            0.55,
            "loud echo tails should be reduced even when they are not quiet",
        )


class BreathCleanupTests(unittest.TestCase):
    def test_low_energy_recovery_breath_loses_tonal_hf_without_body_loss(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        converted = np.zeros_like(t)

        body = (t >= 0.25) & (t < 0.90)
        breath = (t >= 1.15) & (t < 1.45)
        source[body] = 0.15 * np.sin(2.0 * np.pi * 260.0 * t[body])
        converted[body] = source[body]

        breath_body = 0.012 * np.sin(2.0 * np.pi * 620.0 * t[breath])
        breath_body *= _fade(breath_body.size)
        source[breath] = breath_body
        converted[breath] = breath_body + 0.009 * np.sin(2.0 * np.pi * 5200.0 * t[breath])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            converted_path = tmp / "converted.wav"
            sf.write(source_path, source, sr)
            sf.write(converted_path, converted, sr)

            pipeline = CoverPipeline(device="cpu")
            before, _ = sf.read(converted_path)
            pipeline._apply_source_breath_cleanup(str(source_path), str(converted_path))
            pipeline._apply_source_transition_cleanup(str(source_path), str(converted_path))
            after, out_sr = sf.read(converted_path)

        self.assertEqual(out_sr, sr)
        before_breath = before[int(1.20 * sr): int(1.40 * sr)]
        after_breath = after[int(1.20 * sr): int(1.40 * sr)]
        before_body = before[int(0.36 * sr): int(0.76 * sr)]
        after_body = after[int(0.36 * sr): int(0.76 * sr)]

        self.assertLess(_preemphasis_rms(after_breath), _preemphasis_rms(before_breath) * 0.80)
        self.assertGreater(_rms(after_body), _rms(before_body) * 0.96)


if __name__ == "__main__":
    unittest.main()
