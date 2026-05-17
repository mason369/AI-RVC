import json
import tempfile
import unittest
from pathlib import Path

from ui import app as ui_app


class UiLanguageTests(unittest.TestCase):
    def test_default_language_is_chinese_when_config_has_no_language(self):
        self.assertEqual(ui_app.get_configured_language({}), "zh_CN")

    def test_configured_language_accepts_english(self):
        self.assertEqual(ui_app.get_configured_language({"language": "en_US"}), "en_US")

    def test_configured_language_rejects_unsupported_locale(self):
        with self.assertRaises(ValueError):
            ui_app.get_configured_language({"language": "ja_JP"})

    def test_language_choice_maps_display_labels_to_locale_codes(self):
        self.assertEqual(ui_app.resolve_language_choice("中文"), "zh_CN")
        self.assertEqual(ui_app.resolve_language_choice("English"), "en_US")
        self.assertEqual(ui_app.resolve_language_choice("en_US"), "en_US")

    def test_save_language_setting_persists_locale_code(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps({"device": "cpu"}), encoding="utf-8")

            status = ui_app.save_language_setting("English", config_path=config_path)
            saved = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["language"], "en_US")
        self.assertIn("English", status)
        self.assertIn("restart", status.lower())

    def test_language_packs_include_selector_keys(self):
        for lang in ("zh_CN", "en_US"):
            data = ui_app.load_i18n(lang)
            self.assertIn("settings", data)
            self.assertIn("language", data["settings"])
            self.assertIn("save_language", data["settings"])
            self.assertIn("language_saved_restart", data["settings"])

    def test_language_packs_include_primary_ui_keys(self):
        required_ui_keys = {
            "cover_usage",
            "series_filter",
            "keyword_search",
            "keyword_placeholder",
            "character_choice_info",
            "download_character_info",
            "refresh_models",
            "download_selected_character",
            "download_series_all",
            "download_all_characters",
            "download_status",
            "model_name",
            "model_path",
            "index_path",
            "no_models",
            "positive_pitch_info",
            "normal_volume_info",
            "reverb_info",
        }
        required_settings_keys = {
            "runtime_settings",
            "compute_device",
            "save_settings",
            "settings_saved_restart",
            "status",
            "cpu_slow",
            "about_body",
            "model_sources",
        }

        for lang in ("zh_CN", "en_US"):
            data = ui_app.load_i18n(lang)
            self.assertTrue(required_ui_keys.issubset(data.get("ui", {})))
            self.assertTrue(required_settings_keys.issubset(data.get("settings", {})))

    def test_ui_exposes_language_selector_and_save_handler(self):
        source = Path("ui/app.py").read_text(encoding="utf-8")

        self.assertIn('label=t("language", "settings")', source)
        self.assertIn("choices=list(LANGUAGE_LABEL_TO_CODE.keys())", source)
        self.assertIn("fn=save_language_setting", source)
        self.assertIn('t("cover_usage", "ui")', source)
        self.assertIn('t("runtime_settings", "settings")', source)


if __name__ == "__main__":
    unittest.main()
