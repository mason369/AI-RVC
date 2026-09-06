import json
import hashlib
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from configs.persistence import update_config
from tools import download_models
from ui import app


class AuditContractsTests(unittest.TestCase):
    def test_setting_changes_preserve_latest_unrelated_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'device':'cpu','cover':{'index_rate':0.35}}))
            update_config(path, {'language':'en_US'})
            update_config(path, {'device':'auto'})
            data = json.loads(path.read_text())
            self.assertEqual(data, {'device':'auto','language':'en_US','cover':{'index_rate':0.35}})

    def test_invalid_config_or_failed_write_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            original = b'{"device":"cpu"}'
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                update_config(path, {'device':'invalid'})
            with patch('configs.persistence.os.replace', side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):
                    update_config(path, {'language':'en_US'})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_ui_rejects_nonfinite_numbers_and_fractional_integers(self):
        for value in [float('nan'), float('inf'), float('-inf'), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                app._read_ui_float(value, 'volume', 0, 200)
        for value in [0.5, True, float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                app._read_ui_int(value, 'speaker_id', 0, 4)

    def test_base_model_is_not_ready_from_existence_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            valid = b'valid model content'
            spec = {'path':'model.pth','size_bytes':len(valid),'sha256':hashlib.sha256(valid).hexdigest()}
            with patch.object(download_models, 'MODELS', {'model':spec}), patch.object(download_models, 'get_project_root', return_value=Path(directory)):
                path.write_bytes(b'broken')
                self.assertFalse(download_models.check_model('model'))
                path.write_bytes(valid)
                self.assertTrue(download_models.check_model('model'))
                path.write_bytes(b'x' * len(valid))
                self.assertFalse(download_models.check_model('model'))

    def test_install_launch_propagates_child_failure(self):
        import install
        with patch('install.subprocess.run', return_value=Mock(returncode=7)):
            with self.assertRaises(SystemExit) as raised:
                install.launch_app('python')
        self.assertEqual(raised.exception.code, 7)

    def test_language_catalog_uses_native_gradio_metadata(self):
        from ui.live_i18n import catalog, ui_text
        translations = catalog().translations_dict
        self.assertEqual(set(translations), {'zh-CN','en'})
        self.assertEqual(ui_text('language','settings').to_dict()['key'], 'settings.language')
        self.assertNotEqual(translations['zh-CN']['settings.language'], translations['en']['settings.language'])
