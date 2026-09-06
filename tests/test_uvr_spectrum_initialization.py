import inspect
from pathlib import Path
import tempfile
import textwrap
from types import FunctionType, SimpleNamespace
import unittest

import numpy as np

from infer.lib.uvr5_pack.lib_v5 import spec_utils
from lib.upstream_audio_output import initialize_uvr_spectrum


class UvrSpectrumInitializationTests(unittest.TestCase):
    def setUp(self):
        self.parameters = SimpleNamespace(param={
            'band': {1: {'n_fft': 16, 'hl': 4, 'crop_start': 1, 'crop_stop': 8, 'hpf_start': 0}},
            'mid_side': False, 'mid_side_b2': False, 'reverse': False,
        })
        self.spectrum = np.ones((2, 7, 8), dtype=np.complex128) * (1 + 0.25j)
        self.dirty_numpy = SimpleNamespace(**{
            **vars(np),
            'ndarray': lambda shape, dtype: np.full(shape, complex(np.nan, np.nan), dtype=dtype),
        })

    def test_local_reconstruction_ignores_uninitialized_memory(self):
        original = spec_utils.cmb_spectrogram_to_wave
        namespace = dict(original.__globals__, np=self.dirty_numpy)
        function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
        actual = function(self.spectrum, self.parameters)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(np.max(np.abs(actual)), 0)
        np.testing.assert_array_equal(actual, original(self.spectrum, self.parameters))
        self.assertEqual(actual.dtype, np.float64)

    def test_pinned_adapter_eliminates_nan_from_unused_bins(self):
        source = textwrap.dedent(inspect.getsource(spec_utils.cmb_spectrogram_to_wave))
        source = source.replace('spec_s = np.zeros(', 'spec_s = np.ndarray(', 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy_spectrum.py'
            path.write_text(source, encoding='utf-8')
            namespace = dict(spec_utils.cmb_spectrogram_to_wave.__globals__, np=self.dirty_numpy)
            exec(compile(source, str(path), 'exec'), namespace)
            module = SimpleNamespace(cmb_spectrogram_to_wave=namespace['cmb_spectrogram_to_wave'])
            before = module.cmb_spectrogram_to_wave(self.spectrum, self.parameters)
            self.assertFalse(np.isfinite(before).all())
            initialize_uvr_spectrum(module)
            after = module.cmb_spectrogram_to_wave(self.spectrum, self.parameters)
            self.assertTrue(np.isfinite(after).all())
            np.testing.assert_array_equal(after, spec_utils.cmb_spectrogram_to_wave(self.spectrum, self.parameters))
            self.assertEqual(path.read_text(encoding='utf-8'), source)

    def test_changed_upstream_contract_stops_before_reconstruction(self):
        module = SimpleNamespace(cmb_spectrogram_to_wave=spec_utils.cmb_spectrogram_to_wave)
        with self.assertRaisesRegex(RuntimeError, '频谱接口发生变化'):
            initialize_uvr_spectrum(module)


if __name__ == '__main__':
    unittest.main()
