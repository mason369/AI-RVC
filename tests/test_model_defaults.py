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
    def test_roformer_default_uses_audio_separator_top_local_bs_roformer(self):
        from infer import separator

        self.assertEqual(
            separator.ROFORMER_DEFAULT_MODEL,
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        )
        self.assertIn(
            "vocals_mel_band_roformer.ckpt",
            separator.ROFORMER_FALLBACK_MODELS,
        )

    def test_karaoke_default_uses_public_sdr_model_with_legacy_fallbacks(self):
        from infer import separator

        self.assertEqual(
            separator.KARAOKE_DEFAULT_MODEL,
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        )
        self.assertEqual(
            separator.KARAOKE_FALLBACK_MODELS[:3],
            [
                "mel_band_roformer_karaoke_gabox.ckpt",
                "mel_band_roformer_karaoke_gabox_v2.ckpt",
                "mel_band_roformer_karaoke_becruily.ckpt",
            ],
        )

    def test_vc_preprocess_prefers_local_roformer_deecho_before_uvr_fallback(self):
        from infer import separator

        self.assertEqual(
            separator.ROFORMER_DEECHO_DEFAULT_MODEL,
            "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
        )
        self.assertIn(
            "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            separator.ROFORMER_DEECHO_FALLBACK_MODELS,
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


if __name__ == "__main__":
    unittest.main()
