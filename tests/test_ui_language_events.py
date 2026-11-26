"""Exercise queued UI values and reload updates across every supported locale."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import app
from tools import character_models


class LanguageEventTests(unittest.TestCase):
    def test_queued_values_survive_switch_and_reload_refreshes_dynamic_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            configuration = {**app.config, 'language': 'zh_CN',
                             'cover': {**app.config.get('cover', {}), 'source_constraint_mode': 'auto', 'vc_pipeline_mode': 'current'}}
            path.write_text(json.dumps(configuration), encoding='utf-8')
            with patch.object(app, 'CONFIG_PATH', path), patch.object(app, 'config', configuration), \
                    patch.object(app, 'i18n', app.load_i18n('zh_CN')), \
                    patch.object(app, 'set_console_language'), \
                    patch.object(character_models, 'get_character_models_dir', return_value=Path(directory) / 'characters'), \
                    patch.dict(character_models._VERSION_NOTE_CACHE, {}, clear=True), \
                    patch.object(character_models, '_VERSION_NOTE_CACHE_LOADED', False):
                blocks = app.create_ui()
                try:
                    change = next(fn for fn in blocks.fns.values() if getattr(fn.fn, '__name__', '') == 'change_language')
                    restore = next(fn for fn in blocks.fns.values() if getattr(fn.fn, '__name__', '') == 'restore_language')
                    initial = [component.value for component in change.inputs[1:]]
                    source, pipeline, preset = change.inputs[7:10]
                    self.assertEqual([source.value, pipeline.value, preset.value], ['auto', 'current', 'mix_preset_universal'])
                    # These inputs were created in Chinese, like a queued event or
                    # the initial component configuration sent on browser reload.
                    for code, label in app.SUPPORTED_LANGUAGES.items():
                        with self.subTest(locale=code):
                            updates = change.fn(label, *initial)
                            self.assertEqual(updates[change.inputs[0]]['value'], label)
                            self.assertEqual(updates[pipeline]['value'], 'current')
                            self.assertEqual(updates[source]['value'], 'auto')
                            self.assertEqual(updates[preset]['value'], 'mix_preset_universal')
                            self.assertEqual(app.update_route_controls(pipeline.value, True, source.value)[2]['value'], 'auto')
                            saved = path.read_bytes()
                            loaded = restore.fn(*initial)
                            self.assertEqual(loaded[change.inputs[0]]['value'], label)
                            self.assertEqual(path.read_bytes(), saved, 'A page load must not rewrite settings')
                            self.assertEqual(loaded[pipeline]['choices'], list(app.get_vc_pipeline_mode_option_maps()[0].items()))
                            for control in (source, pipeline, preset, change.inputs[11]):
                                self.assertNotIn("value", loaded[control])
                            values = list(loaded.values())
                            self.assertTrue(any(update.get('headers') == [app.t(key, 'ui') for key in ('model_name', 'model_path', 'index_path')] for update in values))
                            self.assertTrue(any(isinstance(update.get('value'), str) and app.t('status_scope', 'models') in update['value'] for update in values))
                            self.assertEqual(json.loads(saved)['language'], code)
                finally:
                    blocks.close()

    def test_presets_have_identical_effective_values_in_every_locale(self):
        values = {}
        for code in app.SUPPORTED_LANGUAGES:
            with patch.object(app, 'i18n', app.load_i18n(code)):
                current = {key: app.apply_cover_mix_preset(key) for key in app.MIX_PRESET_KEYS}
                self.assertEqual([key for _, key in app.get_cover_mix_preset_options()], list(app.MIX_PRESET_KEYS))
                if values:
                    self.assertEqual(current, values)
                values = current

    def test_stable_values_still_reject_unknown_modes_and_presets(self):
        for code in app.SUPPORTED_LANGUAGES:
            with patch.object(app, 'i18n', app.load_i18n(code)):
                with self.assertRaises(ValueError):
                    app.update_route_controls('unknown-route', True, 'auto')
                with self.assertRaises(ValueError):
                    app.apply_cover_mix_preset('unknown-preset')


if __name__ == '__main__':
    unittest.main()
