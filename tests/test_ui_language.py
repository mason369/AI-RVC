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
            "all_series",
            "unknown",
            "enabled",
            "disabled",
            "language_korean",
            "language_japanese",
            "language_chinese",
            "language_english",
            "character_label_meta_separator",
            "character_label_template",
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
        required_message_keys = {
            "download_network_error",
            "please_select_character_to_download",
            "character_download_complete",
            "bulk_download_complete",
            "please_upload_song",
            "please_select_character",
            "character_model_missing",
            "cover_complete_status",
            "cover_process_failed",
            "vc_pipeline_mode_status",
            "all_files_dir_status",
        }
        required_character_detail_keys = {
            "downloaded_empty",
            "available_empty",
            "version_label",
            "continuity",
            "repo",
            "local_weight",
            "internal_key",
            "detail_code_line",
            "detail_text_line",
        }
        required_route_status_keys = {
            "mature_auto_preferred_suffix",
            "mature_current_preferred",
            "official_route_title",
            "strict_route_ready_title",
            "route_current_model",
            "strict_route_flow",
            "auto_route_missing_title",
        }
        required_device_info_keys = {
            "pytorch_version",
            "available_backends",
            "gpu_line",
            "backend_version",
            "no_gpu_cpu",
        }

        for lang in ("zh_CN", "en_US"):
            data = ui_app.load_i18n(lang)
            self.assertTrue(required_ui_keys.issubset(data.get("ui", {})))
            self.assertTrue(required_settings_keys.issubset(data.get("settings", {})))
            self.assertTrue(required_message_keys.issubset(data.get("messages", {})))
            self.assertTrue(required_character_detail_keys.issubset(data.get("character_details", {})))
            self.assertTrue(required_route_status_keys.issubset(data.get("route_status", {})))
            self.assertTrue(required_device_info_keys.issubset(data.get("device_info", {})))

    def test_character_metadata_values_are_not_localized(self):
        original_i18n = ui_app.i18n
        try:
            ui_app.i18n = ui_app.load_i18n("en_US")
            char_info = {
                "name": "rin",
                "display": "Rin Hoshizora",
                "source": "Love Live!",
                "continuity": "μ's",
                "version_label": "500 epochs·40k",
                "distribution": "HuggingFace",
                "repo": "trioskosmos/rvc_models",
                "source_page_url": "https://huggingface.co/trioskosmos/rvc_models",
                "download_url": "https://huggingface.co/trioskosmos/rvc_models/resolve/main/rin.pth",
                "model_path": "assets/weights/characters/rin/rin.pth",
                "index_path": "assets/weights/characters/rin/rin.index",
            }

            label = ui_app.format_character_label(char_info)
            details = ui_app.format_character_details(char_info, downloaded=True)

        finally:
            ui_app.i18n = original_i18n

        self.assertIn("[Japanese]", label)
        self.assertIn("trioskosmos/rvc_models", label)
        self.assertIn("500 epochs·40k", label)
        self.assertIn("- Source repository: `trioskosmos/rvc_models`", details)
        self.assertIn("- Version tag: `500 epochs·40k`", details)
        self.assertIn("assets/weights/characters/rin/rin.pth", details)
        self.assertNotIn("版本标识", details)
        self.assertNotIn("来源仓库", details)

    def test_ui_exposes_language_selector_and_save_handler(self):
        source = Path("ui/app.py").read_text(encoding="utf-8")

        self.assertIn('label=t("language", "settings")', source)
        self.assertIn("choices=list(LANGUAGE_LABEL_TO_CODE.keys())", source)
        self.assertIn("fn=save_language_setting", source)
        self.assertIn('t("cover_usage", "ui")', source)
        self.assertIn('t("runtime_settings", "settings")', source)
        self.assertIn("allow_custom_value=True", source)


if __name__ == "__main__":
    unittest.main()
