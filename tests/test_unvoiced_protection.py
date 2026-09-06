"""Reproduce the pinned F0 interpolation bug without downloading runtime assets."""
import unittest
import traceback
from types import SimpleNamespace

import numpy as np

from lib.upstream_audio_output import preserve_unvoiced_f0


class PinnedF0Fixture:
    """The pinned get_f0 interpolation and native pitch conversion blocks."""
    def get_f0(self, x, p_len, f0_up_key, f0_method):
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        try:
            uv = f0 == 0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        except Exception:
            traceback.print_exc()
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel_min = 1127 * np.log(1 + 50 / 700)
        f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int32), f0bak


class UnvoicedProtectionTests(unittest.TestCase):
    def test_pinned_f0_keeps_the_mask_required_by_protect(self):
        pipeline = PinnedF0Fixture()
        raw = np.array([0, 110, 220, 0, 0], dtype=np.float32)
        pipeline.model_rmvpe = SimpleNamespace(infer_from_audio=lambda *a, **kw: raw.copy())
        _, before = pipeline.get_f0(np.zeros(800), 5, 0, 'rmvpe')
        self.assertTrue((before > 0).all())
        preserve_unvoiced_f0(pipeline)
        coarse, after = pipeline.get_f0(np.zeros(800), 5, 12, 'rmvpe')
        np.testing.assert_array_equal(after, raw * 2)
        np.testing.assert_array_equal(coarse[raw == 0], np.ones(3))
        for raw in (np.zeros(5, dtype=np.float32), np.array([np.nan, 1]), np.array([-1, 100])):
            if np.isfinite(raw).all() and np.all(raw >= 0):
                _, silence = pipeline.get_f0(np.zeros(800), len(raw), 0, 'rmvpe')
                np.testing.assert_array_equal(silence, raw)
            else:
                with self.assertRaises(ValueError):
                    pipeline.get_f0(np.zeros(800), len(raw), 0, 'rmvpe')


if __name__ == '__main__':
    unittest.main()
