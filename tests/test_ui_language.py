import json
import inspect
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ui import app as ui_app


class UiLanguageTests(unittest.TestCase):
    def test_load_config_applies_only_explicit_valid_device_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps({"device": "cuda", "language": "zh_CN"}),
                encoding="utf-8",
            )
            with mock.patch.object(ui_app, "CONFIG_PATH", config_path), mock.patch.dict(
                os.environ,
                {"AI_RVC_DEVICE": "cpu"},
            ):
                self.assertEqual(ui_app.load_config()["device"], "cpu")

            with mock.patch.object(ui_app, "CONFIG_PATH", config_path), mock.patch.dict(
                os.environ,
                {"AI_RVC_DEVICE": "unsupported"},
            ):
                with self.assertRaisesRegex(ValueError, "Unsupported AI_RVC_DEVICE"):
                    ui_app.load_config()

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
            "custom_model_name",
            "custom_model_name_placeholder",
            "custom_model_category",
            "custom_model_default_category",
            "custom_model_source",
            "custom_model_default_source",
            "custom_model_file",
            "custom_index_file",
            "import_custom_model",
            "custom_model_status",
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
            "custom_model_import_complete",
            "custom_model_import_error",
            "please_upload_song",
            "please_select_character",
            "character_model_missing",
            "cover_complete_status",
            "cover_process_failed",
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
            "quality_default_route_title",
            "quality_default_route_flow",
            "quality_default_route_note",
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

    def test_english_language_pack_matches_chinese_keys_and_placeholders(self):
        def flatten(data, prefix=""):
            result = {}
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    result.update(flatten(value, path))
                else:
                    result[path] = value
            return result

        placeholder_re = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
        zh = flatten(ui_app.load_i18n("zh_CN"))
        en = flatten(ui_app.load_i18n("en_US"))

        self.assertEqual(set(zh), set(en))
        for key in sorted(set(zh) & set(en)):
            if isinstance(zh[key], str) and isinstance(en[key], str):
                self.assertEqual(
                    set(placeholder_re.findall(zh[key])),
                    set(placeholder_re.findall(en[key])),
                    msg=f"Placeholder mismatch for {key}",
                )

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

    def test_default_route_status_shows_quality_upstream_official_route(self):
        status = ui_app.get_cover_vc_route_status("auto", "current", True)

        self.assertIn("默认质量链路", status)
        self.assertIn("内置官方 RVC", status)

    def test_english_route_status_has_no_chinese_runtime_build_label(self):
        original_i18n = ui_app.i18n
        original_config = ui_app.config
        try:
            ui_app.i18n = ui_app.load_i18n("en_US")
            ui_app.config = {**original_config, "language": "en_US"}
            status = ui_app.get_cover_vc_route_status("auto", "current", True)
        finally:
            ui_app.i18n = original_i18n
            ui_app.config = original_config

        self.assertIn("Runtime build:", status)
        self.assertFalse(any("一" <= char <= "鿿" for char in status))

    def test_runtime_build_label_rejects_unsupported_language(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runtime build label language"):
            ui_app.get_runtime_build_label("ja_JP")

    def test_route_status_accepts_dropdown_labels(self):
        _, vc_value_to_label = ui_app.get_vc_preprocess_option_maps()
        _, pipeline_value_to_label = ui_app.get_vc_pipeline_mode_option_maps()

        official_status = ui_app.get_cover_vc_route_status(
            vc_value_to_label["auto"],
            pipeline_value_to_label["official"],
            True,
        )
        self.assertIn("当前使用内置官方 RVC 实现", official_status)

        strict_status = ui_app.get_cover_vc_route_status(
            vc_value_to_label["uvr_deecho"],
            pipeline_value_to_label["current"],
            False,
        )
        self.assertNotIn("自动模式", strict_status)

    def test_automatic_cover_settings_ignore_ui_parameter_overrides(self):
        settings = ui_app.resolve_automatic_cover_settings(
            {
                "cover": {
                    "index_rate": 0.5,
                    "speaker_id": 0,
                    "karaoke_separation": True,
                    "karaoke_merge_backing_into_accompaniment": True,
                    "vc_preprocess_mode": "auto",
                    "source_constraint_mode": "auto",
                    "vc_pipeline_mode": "current",
                    "singing_repair": False,
                    "default_vocals_volume": 100,
                    "default_accompaniment_volume": 100,
                    "default_reverb": 0,
                    "rms_mix_rate": 0.0,
                    "backing_mix": 0.0,
                }
            }
        )

        self.assertEqual(settings["pitch_shift"], 0)
        self.assertEqual(settings["index_ratio"], 0.5)
        self.assertEqual(settings["speaker_id"], 0)
        self.assertEqual(settings["vc_preprocess_mode"], "auto")
        self.assertEqual(settings["source_constraint_mode"], "auto")
        self.assertEqual(settings["vc_pipeline_mode"], "current")
        self.assertNotIn("singing_repair", settings)
        self.assertEqual(settings["vocals_volume"], 1.0)
        self.assertEqual(settings["accompaniment_volume"], 1.0)
        self.assertEqual(settings["reverb_amount"], 0.0)

    def test_automatic_cover_settings_reject_invalid_config_instead_of_fallback(self):
        with self.assertRaisesRegex(ValueError, "Invalid cover config"):
            ui_app.resolve_automatic_cover_settings(
                {
                    "cover": {
                        "index_rate": "not-a-number",
                        "vc_preprocess_mode": "direct",
                        "source_constraint_mode": "auto",
                        "vc_pipeline_mode": "current",
                    }
                }
            )

    def test_cover_usage_is_one_click_without_parameter_step(self):
        zh = ui_app.load_i18n("zh_CN")
        en = ui_app.load_i18n("en_US")

        self.assertNotIn("调整参数", zh["ui"]["cover_usage"])
        self.assertNotIn("Adjust parameters", en["ui"]["cover_usage"])
        self.assertIn("点击", zh["ui"]["cover_usage"])
        self.assertIn("Start Cover", en["ui"]["cover_usage"])

    def test_effective_cover_controls_remain_manual_and_deprecated_options_are_removed(self):
        source = Path("ui/app.py").read_text(encoding="utf-8")

        self.assertRegex(source, r"t\(['\"]automatic_cover_settings['\"], ['\"]cover['\"]\)")
        effective_ui_tokens = [
            "cover_pitch_shift",
            "cover_index_rate",
            "cover_speaker_id",
            "cover_karaoke",
            "cover_karaoke_merge_backing",
            "cover_vc_preprocess_mode",
            "cover_source_constraint_mode",
            "cover_vc_pipeline_mode",
            "cover_mix_preset",
            "cover_vocals_volume",
            "cover_accompaniment_volume",
            "cover_reverb",
            "cover_rms_mix_rate",
            "cover_backing_mix",
            "apply_cover_mix_preset",
            "get_cover_mix_defaults",
            "get_cover_mix_presets",
            "get_vc_preprocess_option_maps",
            "get_source_constraint_option_maps",
            "get_vc_pipeline_mode_option_maps",
        ]
        for token in effective_ui_tokens:
            self.assertIn(token, source)

        removed_ui_tokens = [
            "cover_singing_repair",
            "update_singing_repair_visibility",
            "vc_preprocess_direct",
            "vc_preprocess_legacy",
        ]
        for token in removed_ui_tokens:
            self.assertNotIn(token, source)

        self.assertEqual(
            list(inspect.signature(ui_app.process_cover).parameters),
            [
                "audio_path",
                "character_name",
                "pitch_shift",
                "index_ratio",
                "speaker_id",
                "karaoke_separation",
                "karaoke_merge_backing_into_accompaniment",
                "vc_preprocess_mode",
                "source_constraint_mode",
                "vc_pipeline_mode",
                "vocals_volume",
                "accompaniment_volume",
                "reverb_amount",
                "rms_mix_rate",
                "backing_mix",
                "progress",
            ],
        )
        zh_cover = ui_app.load_i18n("zh_CN")["cover"]
        self.assertIn("手动调整", zh_cover["manual_cover_settings"])
        self.assertIn("不会偷偷降级", zh_cover["automatic_cover_settings_info"])
        self.assertNotIn("singing_repair", zh_cover)
        self.assertNotIn("vc_preprocess_direct", zh_cover)
        self.assertNotIn("vc_preprocess_legacy", zh_cover)
        self.assertNotIn("singing_repair = _read_cover_bool", source)
        self.assertIn("singing_repair=False", source)

    def test_ui_exposes_language_selector_and_save_handler(self):
        source = Path("ui/app.py").read_text(encoding="utf-8")

        self.assertIn('label=t("language", "settings")', source)
        self.assertIn("choices=list(LANGUAGE_LABEL_TO_CODE.keys())", source)
        self.assertIn("fn=save_language_setting", source)
        self.assertIn('t("cover_usage", "ui")', source)
        self.assertIn('t("runtime_settings", "settings")', source)
        self.assertIn("allow_custom_value=True", source)
        self.assertIn("custom_model_file = gr.File", source)
        self.assertIn("fn=import_custom_character_model_ui", source)


if __name__ == "__main__":
    unittest.main()
