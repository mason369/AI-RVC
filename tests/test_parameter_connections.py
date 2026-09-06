"""Regression coverage for settings that previously missed an execution branch."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf


class ParameterConnectionTests(unittest.TestCase):
    def test_uvr5_selection_is_independent_of_vc_engine(self):
        from infer import cover_pipeline as module
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.wav'
            sf.write(source, np.zeros(1600), 16000, subtype='FLOAT')
            for use_official in (False, True):
                with self.subTest(use_official=use_official):
                    pipeline = module.CoverPipeline('cpu')
                    pipeline.temp_dir = root / 'sessions'
                    with (
                        patch.object(module.torch, 'load', return_value={}),
                        patch('infer.contracts.inspect_checkpoint', return_value=SimpleNamespace(speaker_count=1, uses_f0=True)),
                        patch.object(module, 'separate_uvr5_official_upstream', side_effect=RuntimeError('selected UVR5')) as uvr5,
                        patch.object(pipeline, '_init_separator', side_effect=AssertionError('wrong separator')),
                        self.assertRaisesRegex(RuntimeError, 'selected UVR5'),
                    ):
                        pipeline.process(str(source), 'model.pth', index_ratio=0, separator='uvr5',
                                         use_official=use_official, uvr5_model='HP2_all_vocals', uvr5_agg=7,
                                         karaoke_separation=False)
                    self.assertEqual(uvr5.call_args.args[2], 'HP2_all_vocals')
                    self.assertEqual(uvr5.call_args.kwargs['agg'], 7)
                    self.assertEqual(uvr5.call_args.kwargs['fmt'], 'wav')

    def test_configured_speaker_id_reaches_model_control(self):
        from ui import app
        with (
            patch.object(app, 'config', {'cover': {'speaker_id': 3, 'index_rate': .7}}),
            patch('tools.character_models.get_character_model_path', return_value={'model_path': 'model.pth', 'index_path': 'model.index'}),
            patch('torch.load', return_value={}),
            patch('infer.contracts.inspect_checkpoint', return_value=SimpleNamespace(speaker_count=5, uses_f0=True)),
        ):
            speaker, _, index = app.update_model_controls('test')
            self.assertEqual(speaker['value'], 3)
            self.assertEqual(speaker['maximum'], 4)
            self.assertEqual(index['value'], 70)

    def test_invalid_configured_speaker_is_not_reset_to_zero(self):
        from ui import app
        with (
            patch.object(app, 'config', {'cover': {'speaker_id': 3}}),
            patch('tools.character_models.get_character_model_path', return_value={'model_path': 'model.pth'}),
            patch('torch.load', return_value={}),
            patch('infer.contracts.inspect_checkpoint', return_value=SimpleNamespace(speaker_count=1, uses_f0=True)),
            self.assertRaises(ValueError),
        ):
            app.update_model_controls('test')

    def test_pm_silence_is_zero_and_gate_really_attenuates(self):
        from infer.f0_extractor import PMExtractor
        from infer.pipeline import VoiceConversionPipeline
        audio = np.zeros(16000, dtype=np.float32)
        audio[:8000] = .2 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)
        f0 = PMExtractor().extract(audio)
        self.assertTrue(np.isfinite(f0).all())
        self.assertTrue((f0[-20:] == 0).all())
        gated = VoiceConversionPipeline('cpu')._apply_silence_gate(
            np.ones_like(audio), audio, f0, 16000, 16000, 160, -30, 0, 100, .33)
        self.assertLess(float(np.mean(gated[-1600:])), .34)
        self.assertGreater(float(np.mean(gated[1600:4000])), .99)

    def test_fcpe_factory_uses_the_bundled_model_and_unvoiced_frames(self):
        import torch
        from infer.f0_extractor import get_f0_extractor
        model = Mock()
        model.infer.return_value = torch.tensor([[[220.0], [0.0], [330.0]]])
        with patch('torchfcpe.spawn_bundled_infer_model', return_value=model) as spawn:
            extractor = get_f0_extractor('fcpe', device='cpu')
            actual = extractor.extract(np.zeros(320, dtype=np.float32))
        spawn.assert_called_once_with(device='cpu')
        np.testing.assert_array_equal(actual, [220, 0, 330])
        self.assertIs(model.infer.call_args.kwargs['interp_uv'], False)
        self.assertEqual(model.infer.call_args.kwargs['threshold'], .006)
        self.assertEqual(model.infer.call_args.kwargs['output_interp_target_length'], 3)


if __name__ == '__main__':
    unittest.main()
