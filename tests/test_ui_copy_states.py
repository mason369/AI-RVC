"""Check observable status semantics and localized validation failures."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ui.app as app


class CopyStateTests(unittest.TestCase):
    def test_importable_component_does_not_imply_weights_present(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'dereverb.ckpt'
            with patch('infer.separator.check_roformer_available', return_value=True), \
                 patch('tools.download_models.get_default_separator_asset_paths', return_value={'RoFormer De-Reverb Stereo': [missing]}), \
                 patch.object(app, 'i18n', app.load_i18n('zh_CN')):
                status = app.check_mature_deecho_status()
            self.assertIn('分离组件可导入', status)
            self.assertIn('dereverb.ckpt', status)
            self.assertIn('文件缺失', status)
            self.assertNotIn('✅', status)

    def test_selected_route_is_not_success_when_component_missing(self):
        with patch('infer.separator.check_roformer_available', return_value=False), \
             patch.object(app, 'i18n', app.load_i18n('zh_CN')):
            status = app.get_cover_vc_route_status('current', True)
        self.assertIn('已选择', status)
        self.assertIn('分离组件无法导入', status)
        self.assertNotIn('✅', status)

    def test_file_presence_and_validation_are_distinct(self):
        with patch('tools.download_models.check_model', return_value=False), \
             patch('tools.download_models.get_missing_upstream_rvc_files', return_value=['missing runtime.py']), \
             patch('tools.download_models.check_default_separator_models', return_value={'Leap': True, 'DeReverb': False}), \
             patch.object(app, 'i18n', app.load_i18n('zh_CN')):
            status = app.check_models_status()
        self.assertIn('Leap: 文件已存在', status)
        self.assertIn('DeReverb: 文件缺失', status)
        self.assertIn('官方 RVC 运行组件: 缺失或校验失败', status)
        self.assertIn('missing runtime.py', status)

    def test_invalid_numbers_show_localized_field_and_range(self):
        for locale, expected in [('zh_CN', '音调偏移'), ('en_US', 'Pitch Shift')]:
            with self.subTest(locale=locale), patch.object(app, 'i18n', app.load_i18n(locale)):
                for invalid in (True, float('nan'), float('inf'), -25, 'invalid'):
                    with self.subTest(value=invalid), self.assertRaises(ValueError) as error:
                        app._read_ui_float(invalid, 'pitch_shift', -24, 24)
                    self.assertIn(expected.lower(), str(error.exception).lower())
                    self.assertIn('-24', str(error.exception))
                with self.assertRaises(ValueError):
                    app._read_ui_int(0.5, 'pitch_shift', -24, 24)
                self.assertEqual(2, app._read_ui_int(2, 'pitch_shift', -24, 24))


if __name__ == '__main__':
    unittest.main()
