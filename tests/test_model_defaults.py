import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ModelDefaultTests(unittest.TestCase):
    def test_roformer_default_uses_public_audio_separator_sota_model(self):
        from infer import separator

        self.assertEqual(
            separator.ROFORMER_DEFAULT_MODEL,
            "vocals_mel_band_roformer.ckpt",
        )
        self.assertIn(
            "melband_roformer_big_beta4.ckpt",
            separator.ROFORMER_FALLBACK_MODELS,
        )

    def test_karaoke_default_uses_public_sota_gabox_model(self):
        from infer import separator

        self.assertEqual(
            separator.KARAOKE_DEFAULT_MODEL,
            "mel_band_roformer_karaoke_gabox.ckpt",
        )
        self.assertEqual(
            separator.KARAOKE_SOTA_MODEL,
            "mel_band_roformer_karaoke_gabox.ckpt",
        )
        self.assertEqual(
            separator.KARAOKE_FALLBACK_MODELS[:1],
            ["mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"],
        )
        self.assertNotEqual(
            separator.KARAOKE_DEFAULT_MODEL,
            "sota_karaoke_ensemble",
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
