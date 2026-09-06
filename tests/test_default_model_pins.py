"""Default inference reuses verified assets without querying mutable HF branches."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infer.separator import (
    _CUSTOM_AUDIO_SEPARATOR_MODELS, _install_custom_audio_separator_models,
    LEAP_XE_VOCALS_MODEL, LEAP_INSTRUMENTAL_MODEL, KARAOKE_SOTA_MODELS,
    ROFORMER_DEREVERB_DEFAULT_MODEL,
)


class DefaultModelPinTests(unittest.TestCase):
    def test_all_six_defaults_have_immutable_content_contracts(self):
        names = [LEAP_XE_VOCALS_MODEL, LEAP_INSTRUMENTAL_MODEL,
                 *KARAOKE_SOTA_MODELS, ROFORMER_DEREVERB_DEFAULT_MODEL]
        self.assertEqual(len(names), 6)
        for name in names:
            spec = _CUSTOM_AUDIO_SEPARATOR_MODELS[name]
            self.assertRegex(spec['revision'], r'^[0-9a-f]{40}$')
            for key in ('model_sha256', 'config_sha256'):
                self.assertRegex(spec[key], r'^[0-9a-f]{64}$')

    def test_cached_custom_model_load_is_offline_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = {'model.ckpt': b'verified weight', 'config.yaml': b'model_type: bs_roformer\n'}
            for name, content in data.items():
                (Path(directory) / name).write_bytes(content)
            spec = {'repo_id': 'audit/model', 'revision': 'a' * 40,
                    'model_filename': 'model.ckpt', 'config_filename': 'config.yaml',
                    'model_sha256': hashlib.sha256(data['model.ckpt']).hexdigest(),
                    'config_sha256': hashlib.sha256(data['config.yaml']).hexdigest(),
                    'friendly_name': 'audit'}
            separator = mock.Mock(model_file_dir=directory, torch_device='cpu',
                                  _ai_rvc_custom_models_installed=False)
            with mock.patch.dict(_CUSTOM_AUDIO_SEPARATOR_MODELS, {'audit.ckpt': spec}), \
                 mock.patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('Unexpected network request')):
                _install_custom_audio_separator_models(separator)
                separator.download_model_files('audit.ckpt')
                (Path(directory) / 'model.ckpt').write_bytes(b'corrupt weight!')
                with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                    separator.download_model_files('audit.ckpt')


if __name__ == '__main__':
    unittest.main()
