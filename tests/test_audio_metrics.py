import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ReferenceAudioMetricTests(unittest.TestCase):
    def test_si_sdr_is_scale_invariant_but_sdr_is_not(self):
        from lib.audio_metrics import signal_distortion_ratio, scale_invariant_signal_distortion_ratio

        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        reference = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        scaled = 2.0 * reference

        self.assertGreater(scale_invariant_signal_distortion_ratio(reference, scaled), 100.0)
        self.assertLess(signal_distortion_ratio(reference, scaled), 0.1)

    def test_reference_stem_metrics_rank_clean_estimate_above_noisy_estimate(self):
        from lib.audio_metrics import evaluate_reference_stems

        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        lead = 0.18 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        backing = 0.04 * np.sin(2 * np.pi * 330 * t + 0.4).astype(np.float32)
        noise = 0.03 * np.sin(2 * np.pi * 910 * t).astype(np.float32)

        clean = evaluate_reference_stems(
            references={"lead": lead, "backing": backing},
            estimates={"lead": lead, "backing": backing},
        )
        noisy = evaluate_reference_stems(
            references={"lead": lead, "backing": backing},
            estimates={"lead": lead + noise, "backing": backing - noise},
        )

        self.assertGreater(clean["mean_si_sdr"], noisy["mean_si_sdr"] + 20.0)
        self.assertGreater(clean["stems"]["lead"]["si_sdr"], noisy["stems"]["lead"]["si_sdr"])
        self.assertGreater(clean["stems"]["backing"]["sdr"], noisy["stems"]["backing"]["sdr"])


if __name__ == "__main__":
    unittest.main()
