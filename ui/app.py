# -*- coding: utf-8 -*-
"""
Gradio 界面 - RVC AI 翻唱
"""
import os
import json
import math
import re
import shutil
import tempfile
import uuid
import zipfile
import gradio as gr
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Set

from lib.console_i18n import console_literals, localize_console_message, set_console_language
from lib.logger import log
from lib.runtime_build import get_runtime_build_label, get_runtime_build_short_label
from infer.separator import ROFORMER_DEFAULT_MODEL, KARAOKE_DEFAULT_MODEL
from ui.live_i18n import ui_text, catalog as ui_catalog, LANGUAGE_CHANGE_JS, bind_static_translations
from configs.persistence import update_config
from ui.multitrack_player import (
    player_html,
    THEME_INIT_SCRIPT,
    PLAYER_BRIDGE_JS,
    PLAYER_UPDATE_JS,
    PLAYER_CLEAR_JS,
)

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config.json"
DEFAULT_LANGUAGE = "zh_CN"
SUPPORTED_LANGUAGES = {
    "zh_CN": "中文",
    "en_US": "English",
}
LANGUAGE_LABEL_TO_CODE = {label: code for code, label in SUPPORTED_LANGUAGES.items()}
MIX_PRESET_KEYS = ("mix_preset_universal", "mix_preset_vocal", "mix_preset_accompaniment", "mix_preset_live")

# 加载语言包
def load_i18n(lang: str = "zh_CN") -> dict:
    """加载语言包"""
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {lang}")
    i18n_path = ROOT_DIR / "i18n" / f"{lang}.json"
    if not i18n_path.exists():
        raise FileNotFoundError(f"Language pack not found: {i18n_path}")
    with open(i18n_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 加载配置
def load_config() -> dict:
    """加载配置，并应用调用方显式提供的运行设备。"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    else:
        loaded = {}

    explicit_device = os.environ.get("AI_RVC_DEVICE", "").strip().lower()
    if explicit_device:
        allowed_devices = {"auto", "cuda", "xpu", "directml", "mps", "cpu"}
        if explicit_device not in allowed_devices:
            raise ValueError(
                f"Unsupported AI_RVC_DEVICE={explicit_device!r}; "
                f"expected one of: {', '.join(sorted(allowed_devices))}"
            )
        loaded["device"] = explicit_device
    from configs.schema import validate_config
    return validate_config(loaded)


def normalize_config(config: dict) -> dict:
    """Normalize legacy path keys to top-level entries."""
    if not config:
        return {}

    paths = config.get("paths", {})
    if "hubert_path" not in config and "hubert" in paths:
        config["hubert_path"] = paths["hubert"]
    if "rmvpe_path" not in config and "rmvpe" in paths:
        config["rmvpe_path"] = paths["rmvpe"]
    if "weights_dir" not in config and "weights" in paths:
        config["weights_dir"] = paths["weights"]
    if "output_dir" not in config and "outputs" in paths:
        config["output_dir"] = paths["outputs"]
    elif config.get("output_dir") == "output" and "outputs" in paths:
        config["output_dir"] = paths["outputs"]
    if "temp_dir" not in config and "temp" in paths:
        config["temp_dir"] = paths["temp"]

    return config

config = normalize_config(load_config())
i18n = load_i18n(str(config.get("language", DEFAULT_LANGUAGE)).strip() or DEFAULT_LANGUAGE)
pipeline = None
_GRADIO_TEMP_DOWNLOAD_PREFIX_RE = re.compile(
    r"^[A-Za-z]__.*?_gradio_[^_]+_",
    flags=re.IGNORECASE,
)


def t(key: str, section: str = None) -> str:
    """获取翻译文本"""
    if section:
        return i18n.get(section, {}).get(key, key)
    return i18n.get(key, key)


def tf(key: str, section: str = None, **kwargs) -> str:
    """Format translated UI text while preserving supplied technical values."""
    return t(key, section).format(**kwargs)


def _all_series_label() -> str:
    return t("all_series", "ui")


def _unknown_label() -> str:
    return t("unknown", "ui")


def _is_all_series(series: Optional[str]) -> bool:
    text = str(series or "").strip()
    return text in {"", "全部", "All", _all_series_label()}


def _normalize_series_choice(series: Optional[str]) -> str:
    return _all_series_label() if _is_all_series(series) else str(series)


def _display_series_label(series: Optional[str]) -> str:
    text = str(series or "").strip()
    return _unknown_label() if text in {"", "未知", "Unknown"} else text


def _series_matches(char_series: Optional[str], selected_series: str) -> bool:
    return _display_series_label(char_series) == _normalize_series_choice(selected_series)


def _bool_status_label(value: bool) -> str:
    return t("enabled", "ui") if value else t("disabled", "ui")


def get_configured_language(config_data: Optional[dict] = None) -> str:
    """Return the configured UI language, failing on unsupported values."""
    selected_config = config if config_data is None else config_data
    language = str(selected_config.get("language", DEFAULT_LANGUAGE)).strip() or DEFAULT_LANGUAGE
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    return language


def resolve_language_choice(language_choice: str) -> str:
    """Normalize a dropdown label or language code to a supported locale code."""
    choice = str(language_choice or "").strip()
    if choice in SUPPORTED_LANGUAGES:
        return choice
    if choice in LANGUAGE_LABEL_TO_CODE:
        return LANGUAGE_LABEL_TO_CODE[choice]
    raise ValueError(f"Unsupported language choice: {language_choice}")


def get_current_language_label() -> str:
    return SUPPORTED_LANGUAGES[get_configured_language()]


def save_language_setting(language_choice: str, config_path: Optional[Path] = None) -> str:
    """Persist language atomically, then update runtime text for subsequent events."""
    global i18n
    language = resolve_language_choice(language_choice)
    message_i18n = load_i18n(language)
    target_path = Path(config_path) if config_path is not None else CONFIG_PATH
    update_config(target_path, {'language': language})
    if target_path == CONFIG_PATH:
        config['language'] = language
        i18n = message_i18n
        set_console_language(language)
    return message_i18n['settings']['language_saved'].format(language=SUPPORTED_LANGUAGES[language])


def _read_cover_choice(
    cover_cfg: Dict[str, Any],
    key: str,
    default: str,
    allowed: Set[str],
) -> str:
    value = str(cover_cfg.get(key, default)).strip().lower()
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid cover config: cover.{key}={value!r}; expected one of: {allowed_values}"
        )
    return value


def _read_cover_float(
    cover_cfg: Dict[str, Any],
    key: str,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    raw_value = cover_cfg.get(key, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid cover config: cover.{key} must be a number, got {raw_value!r}"
        ) from exc
    if isinstance(raw_value, bool) or not math.isfinite(value) or value < min_value or value > max_value:
        raise ValueError(
            f"Invalid cover config: cover.{key}={value}; expected {min_value}-{max_value}"
        )
    return value


def _read_cover_int(
    cover_cfg: Dict[str, Any],
    key: str,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    raw_value = cover_cfg.get(key, default)
    try:
        from infer.contracts import integer
        value = integer(raw_value, key, min_value, max_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid cover config: cover.{key} must be an integer, got {raw_value!r}"
        ) from exc
    if isinstance(raw_value, bool) or not math.isfinite(value) or value < min_value or value > max_value:
        raise ValueError(
            f"Invalid cover config: cover.{key}={value}; expected {min_value}-{max_value}"
        )
    return value


def _read_cover_bool(cover_cfg: Dict[str, Any], key: str, default: bool) -> bool:
    raw_value = cover_cfg.get(key, default)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(
        f"Invalid cover config: cover.{key} must be true or false, got {raw_value!r}"
    )


def resolve_automatic_cover_settings(config_data: Optional[dict] = None) -> Dict[str, object]:
    """Resolve one-click cover settings from explicit config values."""
    selected_config = config if config_data is None else config_data
    if not isinstance(selected_config, dict):
        raise ValueError("Invalid cover config: root config must be an object")
    cover_cfg = selected_config.get("cover", {})
    if not isinstance(cover_cfg, dict):
        raise ValueError("Invalid cover config: cover must be an object")

    return {
        "pitch_shift": 0,
        "index_ratio": _read_cover_float(cover_cfg, "index_rate", 0.5, 0.0, 1.0),
        "speaker_id": _read_cover_int(cover_cfg, "speaker_id", 0, 0, 2**31 - 1),
        "karaoke_separation": _read_cover_bool(cover_cfg, "karaoke_separation", True),
        "karaoke_merge_backing_into_accompaniment": _read_cover_bool(
            cover_cfg,
            "karaoke_merge_backing_into_accompaniment",
            True,
        ),
        "source_constraint_mode": _read_cover_choice(
            cover_cfg,
            "source_constraint_mode",
            "auto",
            {"auto", "off", "on"},
        ),
        "vc_pipeline_mode": _read_cover_choice(
            cover_cfg,
            "vc_pipeline_mode",
            "current",
            {"current", "official"},
        ),
        "vocals_volume": _read_cover_float(cover_cfg, "default_vocals_volume", 100.0, 0.0, 200.0) / 100.0,
        "accompaniment_volume": _read_cover_float(
            cover_cfg,
            "default_accompaniment_volume",
            100.0,
            0.0,
            200.0,
        ) / 100.0,
        "reverb_amount": _read_cover_float(cover_cfg, "default_reverb", 0.0, 0.0, 100.0) / 100.0,
        "rms_mix_rate": _read_cover_float(cover_cfg, "rms_mix_rate", 0.0, 0.0, 1.0),
        "backing_mix": _read_cover_float(cover_cfg, "backing_mix", 0.0, 0.0, 1.0),
    }


def _resolve_labeled_choice(
    label_to_value: Dict[str, str],
    selected: str,
    field_name: str,
) -> str:
    raw_value = str(selected or "").strip()
    normalized = label_to_value.get(raw_value, raw_value.lower())
    allowed = set(label_to_value.values())
    if normalized not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid cover UI value: {field_name}={selected!r}; expected one of: {allowed_values}"
        )
    return normalized


def _ui_field_label(field_name: str) -> str:
    key = {"reverb_amount": "vocals_reverb"}.get(field_name, field_name)
    return t(key, "cover")


def _read_ui_float(value, field_name: str, min_value: float, max_value: float) -> float:
    message = tf("invalid_ui_number", "messages", field=_ui_field_label(field_name),
                 minimum=min_value, maximum=max_value)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if isinstance(value, bool) or not math.isfinite(result) or not min_value <= result <= max_value:
        raise ValueError(message)
    return result


def _read_ui_int(value, field_name: str, min_value: int, max_value: int) -> int:
    from infer.contracts import integer
    try:
        return integer(value, field_name, min_value, max_value)
    except ValueError as exc:
        raise ValueError(tf("invalid_ui_integer", "messages", field=_ui_field_label(field_name),
                            minimum=min_value, maximum=max_value)) from exc


def get_cover_mix_defaults() -> Dict[str, int]:
    """Return cover mix defaults from the validated automatic settings."""
    defaults = resolve_automatic_cover_settings(config)
    return {
        "vocals_volume": int(round(float(defaults["vocals_volume"]) * 100)),
        "accompaniment_volume": int(round(float(defaults["accompaniment_volume"]) * 100)),
        "reverb": int(round(float(defaults["reverb_amount"]) * 100)),
    }


def get_cover_mix_presets() -> Tuple[Dict[str, Dict[str, int]], str]:
    """Return effective cover mix presets; these update real mix parameters."""
    defaults = get_cover_mix_defaults()
    presets = {
        t("mix_preset_universal", "cover"): defaults.copy(),
        t("mix_preset_vocal", "cover"): {
            "vocals_volume": min(200, defaults["vocals_volume"] + 15),
            "accompaniment_volume": max(0, defaults["accompaniment_volume"] - 10),
            "reverb": max(0, defaults["reverb"] - 5),
        },
        t("mix_preset_accompaniment", "cover"): {
            "vocals_volume": max(0, defaults["vocals_volume"] - 10),
            "accompaniment_volume": min(200, defaults["accompaniment_volume"] + 15),
            "reverb": max(0, defaults["reverb"] - 5),
        },
        t("mix_preset_live", "cover"): {
            "vocals_volume": defaults["vocals_volume"],
            "accompaniment_volume": defaults["accompaniment_volume"],
            "reverb": min(100, defaults["reverb"] + 10),
        },
    }
    default_name = t("mix_preset_universal", "cover")
    return presets, default_name


def apply_cover_mix_preset(preset_name: str) -> Tuple[int, int, int]:
    """Return the actual mix slider values for a selected preset."""
    presets, _ = get_cover_mix_presets()
    if preset_name in MIX_PRESET_KEYS:
        preset_name = t(preset_name, "cover")
    if preset_name not in presets:
        raise ValueError(f"Unknown mix preset: {preset_name}")
    preset = presets[preset_name]
    return preset["vocals_volume"], preset["accompaniment_volume"], preset["reverb"]


def get_cover_mix_preset_options() -> List[Tuple[str, str]]:
    """Keep event values stable while translating the displayed preset names."""
    return [(t(key, "cover"), key) for key in MIX_PRESET_KEYS]



def get_source_constraint_option_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build effective source constraint dropdown option maps."""
    label_to_value = {
        t("source_constraint_auto", "cover"): "auto",
        t("source_constraint_off", "cover"): "off",
        t("source_constraint_on", "cover"): "on",
    }
    value_to_label = {value: label for label, value in label_to_value.items()}
    return label_to_value, value_to_label


def get_vc_pipeline_mode_option_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build effective VC pipeline mode dropdown option maps."""
    label_to_value = {
        t("vc_pipeline_mode_current", "cover"): "current",
        t("vc_pipeline_mode_official", "cover"): "official",
    }
    value_to_label = {value: label for label, value in label_to_value.items()}
    return label_to_value, value_to_label


def init_pipeline():
    """初始化推理管道"""
    global pipeline

    if pipeline is not None:
        return pipeline

    from infer.pipeline import VoiceConversionPipeline

    device = config.get("device", "cuda")
    pipeline = VoiceConversionPipeline(device=device)

    # 加载 HuBERT
    hubert_path = ROOT_DIR / config.get("hubert_path", "assets/hubert/hubert_base.pt")
    if hubert_path.exists():
        pipeline.load_hubert(str(hubert_path))

    # 加载 F0 提取器
    rmvpe_path = ROOT_DIR / config.get("rmvpe_path", "assets/rmvpe/rmvpe.pt")
    if rmvpe_path.exists():
        pipeline.load_f0_extractor("rmvpe", str(rmvpe_path))

    return pipeline


def download_base_models() -> str:
    """下载基础模型"""
    from tools.download_models import download_required_models

    try:
        success = download_required_models()
        if success:
            return t("download_complete", "messages")
        else:
            return t("download_network_error", "messages")
    except Exception as e:
        return f"{t('download_failed', 'messages')}: {str(e)}"


# ===== 翻唱功能相关函数 =====

def get_downloaded_character_list() -> list:
    """获取已下载的角色列表"""
    from tools.character_models import list_downloaded_characters
    return list_downloaded_characters()


def get_downloaded_character_series() -> list:
    """获取已下载角色的系列列表"""
    characters = get_downloaded_character_list()
    series = sorted({_display_series_label(c.get("series")) for c in characters})
    return [_all_series_label()] + series


def get_available_character_list() -> list:
    """获取可下载的角色列表"""
    from tools.character_models import list_available_characters
    return list_available_characters()


def get_available_character_series() -> list:
    """获取可用系列列表"""
    from tools.character_models import list_available_series
    return sorted({_display_series_label(series) for series in list_available_series()})


def _localized_language_tag(lang: str) -> str:
    text = str(lang or "").strip()
    if not text:
        return text
    lowered = text.lower()
    if text in {"韩文", "韓文"} or "kr" in lowered or "korean" in lowered:
        return t("language_korean", "ui")
    if text in {"日文", "日本語"} or "jp" in lowered or "japanese" in lowered:
        return t("language_japanese", "ui")
    if text in {"中文", "汉语", "漢語"} or "cn" in lowered or "chinese" in lowered:
        return t("language_chinese", "ui")
    if text in {"英文", "英语", "英語"} or lowered in {"en", "english"}:
        return t("language_english", "ui")
    return text


def _localize_character_meta_value(value: str) -> str:
    mapping = i18n.get("character_details", {}).get("metadata_values", {})
    if isinstance(mapping, dict):
        return str(mapping.get(value, value))
    return value


def format_character_label(char_info: dict) -> str:
    """格式化角色展示名称，明确显示版本、归属和来源。"""
    display = char_info.get("base_display") or char_info.get("display") or char_info.get("description") or char_info.get("name", "")
    source = char_info.get("source") or _unknown_label()
    name = char_info.get("name", "")
    lang_tag = get_character_language_tag(char_info)
    parts: List[str] = []
    continuity = _localize_character_meta_value(str(char_info.get("continuity") or "").strip())
    version_label = str(char_info.get("version_label") or "").strip()
    distribution = _localize_character_meta_value(str(char_info.get("distribution") or "").strip())
    repo = str(char_info.get("repo") or "").strip()

    status = char_info.get("compatibility_status")
    if status:
        parts.append(t(f"status_{status}", "character_details"))
    if (char_info.get('source_check') or {}).get('status') == 'unavailable':
        parts.append(t('source_unavailable', 'character_details'))
    if char_info.get("download_supported") is False:
        parts.append(t("manual_download_required", "character_details"))

    if continuity:
        parts.append(continuity)
    if version_label:
        parts.append(version_label)
    parts.append(source)
    if distribution and repo:
        parts.append(f"{distribution}: {repo}")
    elif distribution:
        parts.append(distribution)
    elif repo:
        parts.append(repo)

    meta = t("character_label_meta_separator", "ui").join(part for part in parts if part)
    return tf(
        "character_label_template",
        "ui",
        language=lang_tag,
        display=display,
        meta=meta,
        name=name,
    )


def get_character_language_tag(char_info: dict) -> str:
    """推断语言类型，用于下拉前缀标签"""
    lang = char_info.get("lang")
    if lang:
        return _localized_language_tag(lang)
    text = " ".join(
        str(char_info.get(k, "")) for k in ("display", "description", "name")
    ).lower()
    if "韩" in text or "kr" in text or "korean" in text:
        return t("language_korean", "ui")
    if "日" in text or "jp" in text or "japanese" in text:
        return t("language_japanese", "ui")
    if "中" in text or "cn" in text or "chinese" in text:
        return t("language_chinese", "ui")
    if "en" in text or "english" in text:
        return t("language_english", "ui")

    source = char_info.get("source", "")
    if source.startswith("Love Live!") or "ホロライブ" in source or "偶像大师" in source or "赛马娘" in source:
        return t("language_japanese", "ui")
    if "原神" in source or "崩坏" in source or "明日方舟" in source or "碧蓝航线" in source:
        return t("language_chinese", "ui")
    if "VOCALOID" in source or "Project SEKAI" in source:
        return t("language_japanese", "ui")
    if "Hololive" in source:
        return t("language_japanese", "ui")
    if "蔚蓝档案" in source or "绝区零" in source:
        return t("language_japanese", "ui")
    return t("language_chinese", "ui")


def _find_character_entry(selection: str, downloaded: bool) -> Optional[dict]:
    if not selection:
        return None
    chars = get_downloaded_character_list() if downloaded else get_available_character_list()
    resolved = resolve_character_name(selection)
    for char in chars:
        if selection == char.get("name") or selection == format_character_label(char):
            return char
        if resolved and char.get("name") == resolved:
            return char
    return None


def _character_detail_code(label_key: str, value: str) -> str:
    return tf(
        "detail_code_line",
        "character_details",
        label=t(label_key, "character_details"),
        value=value,
    )


def _character_detail_text(label_key: str, value: str) -> str:
    return tf(
        "detail_text_line",
        "character_details",
        label=t(label_key, "character_details"),
        value=value,
    )


def format_character_details(char_info: Optional[dict], downloaded: bool = False) -> str:
    if not char_info:
        if downloaded:
            return t("downloaded_empty", "character_details")
        return t("available_empty", "character_details")

    title = char_info.get("base_display") or char_info.get("display") or char_info.get("name", "")
    lines = [f"**{title}**"]

    status = char_info.get("compatibility_status")
    if status:
        status_text = t(f"status_{status}", "character_details")
        if status == "validated":
            status_text = f"{status_text} · {t('validation_scope_note', 'character_details')}"
        lines.append(_character_detail_text("compatibility", status_text))
    if char_info.get("compatibility_error"):
        lines.append(_character_detail_text("validation_error", str(char_info["compatibility_error"])))
    source_check = char_info.get('source_check') or {}
    if source_check.get('status') == 'unavailable':
        lines.append(t('source_unavailable_note', 'character_details').format(
            date=source_check['checked_at'], reason=source_check['reason']))
    if char_info.get("download_supported") is False:
        lines.append(t("manual_download_note", "character_details"))

    version_label = str(char_info.get("version_label") or "").strip()
    continuity = _localize_character_meta_value(str(char_info.get("continuity") or "").strip())
    role = _localize_character_meta_value(str(char_info.get("role") or "").strip())
    source = str(char_info.get("source") or "").strip()
    distribution = _localize_character_meta_value(str(char_info.get("distribution") or "").strip())
    repo = str(char_info.get("repo") or "").strip()
    source_page_url = str(char_info.get("source_page_url") or "").strip()
    download_url = str(char_info.get("download_url") or "").strip()

    if version_label:
        lines.append(_character_detail_code("version_label", version_label))
    if continuity:
        lines.append(_character_detail_code("continuity", continuity))
    if role:
        lines.append(_character_detail_code("role", role))
    if source:
        lines.append(_character_detail_code("source", source))
    if distribution:
        lines.append(_character_detail_code("distribution", distribution))
    if repo:
        lines.append(_character_detail_code("repo", repo))
    if source_page_url:
        lines.append(_character_detail_text("source_page_url", source_page_url))
    if download_url and download_url != source_page_url:
        lines.append(_character_detail_text("download_url", download_url))
    if downloaded and char_info.get("model_path"):
        lines.append(_character_detail_code("local_weight", char_info["model_path"]))
    if downloaded and char_info.get("index_path"):
        lines.append(_character_detail_code("local_index", char_info["index_path"]))
    lines.append(_character_detail_code("internal_key", char_info.get("name", "")))
    return "\n".join(lines)


def get_downloaded_character_details(selection: str) -> str:
    return format_character_details(_find_character_entry(selection, downloaded=True), downloaded=True)


def get_available_character_details(selection: str) -> str:
    return format_character_details(_find_character_entry(selection, downloaded=False), downloaded=False)


def update_model_controls(selection: str):
    """按权重的真实能力更新控件；选择模型不会触发推理。"""
    from tools.character_models import get_character_model_path
    from infer.contracts import inspect_checkpoint
    import torch
    model = get_character_model_path(resolve_character_name(selection)) if selection else None
    if model is None:
        return gr.update(value=0, maximum=1, interactive=False), gr.update(value=0, interactive=False), gr.update(value=0, interactive=False)
    checkpoint = torch.load(model["model_path"], map_location="cpu", weights_only=False)
    contract = inspect_checkpoint(checkpoint, model["model_path"])
    has_index = bool(model.get("index_path"))
    speaker_id = _read_ui_int(config.get("cover", {}).get("speaker_id", 0),
                              "speaker_id", 0, contract.speaker_count - 1)
    return (
        gr.update(value=speaker_id, maximum=max(1, contract.speaker_count - 1), interactive=contract.speaker_count > 1),
        gr.update(value=0, interactive=contract.uses_f0),
        gr.update(value=int(round(config.get("cover", {}).get("index_rate", .5) * 100)) if has_index else 0, interactive=has_index),
    )


def get_downloaded_character_choices(series: str = "全部", keyword: str = "") -> list:
    """获取已下载角色的下拉选项"""
    chars = get_downloaded_character_list()
    if series and not _is_all_series(series):
        chars = [c for c in chars if _series_matches(c.get("series"), series)]
    if keyword:
        kw = keyword.strip().lower()
        if kw:
            chars = [
                c for c in chars
                if kw in c.get("name", "").lower()
                or kw in c.get("display", "").lower()
                or kw in c.get("source", "").lower()
                or kw in str(c.get("repo", "")).lower()
                or kw in str(c.get("continuity", "")).lower()
                or kw in str(c.get("version_label", "")).lower()
                or kw in str(c.get("distribution", "")).lower()
            ]
    return [(format_character_label(c), c["name"]) for c in chars]


def resolve_character_name(selection: str) -> str:
    """将下拉显示文本解析为实际角色名"""
    if not selection:
        return selection
    from tools.character_models import list_downloaded_characters
    for c in list_downloaded_characters():
        if selection == c.get("name") or selection == format_character_label(c):
            return c.get("name")
    if " · " in selection:
        return selection.split(" · ")[-1].strip()
    parts = selection.strip().split()
    return parts[-1] if parts else selection


def get_character_filename_display(char_info: Optional[dict], fallback: str) -> str:
    """Return a compact, readable character label for generated filenames."""
    if not char_info:
        return fallback

    base = (
        str(char_info.get("base_display") or "").strip()
        or str(char_info.get("display") or "").strip()
        or fallback
    )
    parts = [part.strip() for part in re.split(r"\s*/\s*", base) if part.strip()]

    variant = str(char_info.get("variant") or "").strip()
    if not variant:
        version_label = str(char_info.get("version_label") or "").strip()
        variant = re.split(r"\s*[·•]\s*", version_label, maxsplit=1)[0].strip()

    if variant and variant.lower() not in {part.lower() for part in parts}:
        parts.append(variant)

    return "-".join(parts) if parts else fallback


def clean_gradio_temp_download_name(filename: str) -> str:
    """Remove Gradio cache path prefixes that can leak into browser downloads."""
    clean_name = _GRADIO_TEMP_DOWNLOAD_PREFIX_RE.sub("", str(filename or "")).strip()
    if not clean_name:
        raise ValueError(f"Cannot derive a clean download filename from: {filename}")
    return clean_name


def normalize_download_output_path(path: Optional[str]) -> Optional[str]:
    """Return a same-directory path whose basename is safe for browser downloads."""
    if not path:
        return path

    source_path = Path(path)
    clean_name = clean_gradio_temp_download_name(source_path.name)
    if clean_name == source_path.name:
        return str(source_path)

    clean_path = source_path.with_name(clean_name)
    shutil.copy2(source_path, clean_path)
    return str(clean_path)


def get_available_character_choices(series: str = "全部", keyword: str = "") -> list:
    """获取可下载角色的下拉选项"""
    chars = get_available_character_list()
    if series and not _is_all_series(series):
        chars = [c for c in chars if _series_matches(c.get("series"), series)]
    if keyword:
        kw = keyword.strip().lower()
        if kw:
            chars = [
                c for c in chars
                if kw in c.get("name", "").lower()
                or kw in c.get("display", "").lower()
                or kw in c.get("source", "").lower()
                or kw in str(c.get("repo", "")).lower()
                or kw in str(c.get("continuity", "")).lower()
                or kw in str(c.get("version_label", "")).lower()
                or kw in str(c.get("distribution", "")).lower()
            ]
    return [(format_character_label(c), c["name"]) for c in chars]


def _refresh_downloaded_updates(series: str, keyword: str) -> Tuple[Dict, Dict]:
    series_choices = get_downloaded_character_series()
    series = _normalize_series_choice(series)
    if series not in series_choices:
        series = _all_series_label()
    return (
        gr.update(choices=series_choices, value=series),
        gr.update(choices=get_downloaded_character_choices(series, keyword), value=None)
    )


def download_character(name: str, selected_series: str = "全部", keyword: str = "") -> Tuple[str, Dict, Dict]:
    """下载角色模型"""
    from tools.character_models import download_character_model

    if not name:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return t("please_select_character_to_download", "messages"), choices_update, series_update

    try:
        success = download_character_model(name)
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        if success:
            return (
                tf("character_download_complete", "messages", name=name),
                choices_update,
                series_update
            )
        else:
            return (
                tf("character_download_failed", "messages", name=name),
                choices_update,
                series_update
            )
    except Exception as e:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return (
            tf("character_download_error", "messages", error=str(e)),
            choices_update,
            series_update
        )


def download_all_characters(series: str = "全部", selected_series: str = "全部", keyword: str = "") -> Tuple[str, Dict, Dict]:
    """批量下载角色模型"""
    from tools.character_models import download_all_character_models

    try:
        series_arg = None if _is_all_series(series) else series
        result = download_all_character_models(series=series_arg)
        ok = result.get("success", [])
        failed = result.get("failed", [])
        status = tf("bulk_download_complete", "messages", count=len(ok))
        if failed:
            status += tf("bulk_download_failed_items", "messages", count=len(failed), names=", ".join(failed))
        for name, error in result.get('errors', {}).items():
            status += f"\n{name}: {error}"
        if result.get('not_attempted'):
            status += tf("bulk_download_not_attempted", "messages", count=len(result['not_attempted']),
                         names=", ".join(result['not_attempted']))
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return status, choices_update, series_update
    except Exception as e:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return tf("bulk_download_error", "messages", error=str(e)), choices_update, series_update


def import_custom_character_model_ui(
    display_name: str,
    category: str,
    source: str,
    model_file: Any,
    index_file: Any,
    selected_series: str = "全部",
    keyword: str = "",
) -> Tuple[str, Dict, Dict, Dict, Any]:
    """导入用户上传的自定义 RVC 模型并刷新角色选择。"""
    from tools.character_models import import_custom_character_model

    try:
        record = import_custom_character_model(
            model_file=model_file,
            index_file=index_file,
            display_name=display_name,
            source=source or t("custom_model_default_source", "ui"),
            category=category or t("custom_model_default_category", "ui"),
        )
        model_series = _display_series_label(record.get("series"))
        series_choices = get_downloaded_character_series()
        if model_series not in series_choices:
            series_choices.append(model_series)
            series_choices = [_all_series_label()] + sorted(
                choice for choice in series_choices if not _is_all_series(choice)
            )
        choices = get_downloaded_character_choices(model_series, "")
        return (
            tf("custom_model_import_complete", "messages", name=record.get("name", "")),
            gr.update(choices=choices, value=record.get("name")),
            gr.update(choices=series_choices, value=model_series),
            gr.update(value=""),
            format_character_details(record, downloaded=True),
        )
    except Exception as e:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return (
            tf("custom_model_import_error", "messages", error=str(e)),
            choices_update,
            series_update,
            gr.update(),
            gr.update(),
        )


def update_download_choices(series: str, keyword: str) -> Dict:
    """更新下载下拉列表"""
    return gr.update(choices=get_available_character_choices(series, keyword), value=None)


def update_downloaded_choices(series: str, keyword: str) -> Dict:
    """更新已下载角色下拉列表"""
    return gr.update(choices=get_downloaded_character_choices(series, keyword), value=None)


def refresh_downloaded_controls(series: str, keyword: str) -> Tuple[Dict, Dict]:
    """刷新已下载角色的筛选和列表"""
    return _refresh_downloaded_updates(series, keyword)


def cover_status_text(data):
    if not data:
        return ""
    fields = dict(data.get("fields", {}))
    if "error" in fields and not data.get("raw_error"):
        with console_literals(*data.get("literals", ())):
            fields["error"] = localize_console_message(fields["error"], get_configured_language())
    lines = [tf(data["message"], "messages", **fields)]
    if data.get("stages"):
        names = [t("stage_" + stage, "cover") for stage in data["stages"]]
        lines.append(tf("reused", "cover", stages=" / ".join(names)))
    if "route" in data:
        lines.append(get_cover_vc_route_status(**data["route"]).splitlines()[0])
    lines.extend(tf(key, "messages", value=value) for key, value in data.get("details", {}).items() if value)
    return "\n".join(lines)


def process_cover(
    audio_path: str,
    character_name: str,
    pitch_shift: int,
    index_ratio: float,
    speaker_id: float,
    karaoke_separation: bool,
    karaoke_merge_backing_into_accompaniment: bool,
    source_constraint_mode: str,
    vc_pipeline_mode: str,
    vocals_volume: float,
    accompaniment_volume: float,
    reverb_amount: float,
    rms_mix_rate: float,
    backing_mix: float,
    input_mode: str = "song",
    progress=gr.Progress(),
    *, reuse_state=None, run_directory=None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], str]:
    """
    处理翻唱

    Returns:
        Tuple[cover, converted_vocals, original_vocals, lead_vocals, backing_vocals,
        accompaniment, accompaniment_without_harmony, status]
    """
    _none7 = (None, None, None, None, None, None, None)
    def finish(message, paths=_none7, **details):
        data = dict(message=message, **details)
        if reuse_state is not None:
            reuse_state["selected_status"] = data
        return *paths, cover_status_text(data)

    if audio_path is None:
        return finish("please_upload_song")
    if not character_name:
        return finish("please_select_character")

    literals = [Path(audio_path).stem, ROOT_DIR, character_name]
    try:
        from tools.character_models import get_character_model_path, get_character_info
        from infer.cover_pipeline import get_cover_pipeline

        # 获取角色模型路径
        resolved_name = resolve_character_name(character_name)
        char_meta = get_character_info(resolved_name, downloaded_only=True) or {}
        model_info = get_character_model_path(resolved_name)
        if model_info is None:
            return finish("character_model_missing", fields={"name": resolved_name})

        # 进度回调
        def progress_callback(msg: str, step: int, total: int):
            if total > 0:
                progress(
                    step / total,
                    desc=localize_console_message(msg, get_configured_language()),
                )

        # 获取流水线
        device = config.get("device", "cuda")
        pipeline = get_cover_pipeline(device)

        cover_cfg = config.get("cover", {})
        demucs_model = cover_cfg.get("demucs_model", "htdemucs")
        demucs_shifts = int(cover_cfg.get("demucs_shifts", 2))
        demucs_overlap = float(cover_cfg.get("demucs_overlap", 0.25))
        demucs_split = bool(cover_cfg.get("demucs_split", True))
        separator = cover_cfg.get("separator", "roformer")
        roformer_model = cover_cfg.get("roformer_model", ROFORMER_DEFAULT_MODEL)
        uvr5_model = cover_cfg.get("uvr5_model")
        uvr5_agg = int(cover_cfg.get("uvr5_agg", 10))
        uvr5_format = cover_cfg.get("uvr5_format", "wav")
        use_official = bool(cover_cfg.get("use_official", True))
        f0_method = cover_cfg.get("f0_method", config.get("f0_method", "rmvpe"))
        protect = cover_cfg.get("protect", config.get("protect", 0.33))
        silence_gate = cover_cfg.get("silence_gate", True)
        silence_threshold_db = cover_cfg.get("silence_threshold_db", -40.0)
        silence_smoothing_ms = cover_cfg.get("silence_smoothing_ms", 50.0)
        silence_min_duration_ms = cover_cfg.get("silence_min_duration_ms", 200.0)
        karaoke_model = cover_cfg.get(
            "karaoke_model",
            KARAOKE_DEFAULT_MODEL,
        )
        source_label_to_value, _ = get_source_constraint_option_maps()
        pipeline_label_to_value, _ = get_vc_pipeline_mode_option_maps()

        source_constraint_mode = _resolve_labeled_choice(
            source_label_to_value,
            source_constraint_mode,
            "source_constraint_mode",
        )
        vc_pipeline_mode = _resolve_labeled_choice(
            pipeline_label_to_value,
            vc_pipeline_mode,
            "vc_pipeline_mode",
        )

        pitch_shift = _read_ui_int(pitch_shift, "pitch_shift", -12, 12)
        index_ratio = _read_ui_float(index_ratio, "index_rate", 0.0, 100.0) / 100.0
        from infer.contracts import inspect_checkpoint
        import torch
        model_contract = inspect_checkpoint(torch.load(model_info["model_path"], map_location="cpu", weights_only=False), model_info["model_path"])
        speaker_id = _read_ui_int(speaker_id, "speaker_id", 0, model_contract.speaker_count - 1)
        karaoke_separation = bool(karaoke_separation)
        karaoke_merge_backing_into_accompaniment = bool(karaoke_merge_backing_into_accompaniment)
        vocals_volume = _read_ui_float(vocals_volume, "vocals_volume", 0.0, 200.0) / 100.0
        accompaniment_volume = _read_ui_float(
            accompaniment_volume,
            "accompaniment_volume",
            0.0,
            200.0,
        ) / 100.0
        reverb_amount = _read_ui_float(reverb_amount, "reverb_amount", 0.0, 100.0) / 100.0
        rms_mix_rate = _read_ui_float(rms_mix_rate, "rms_mix_rate", 0.0, 100.0) / 100.0
        backing_mix = _read_ui_float(backing_mix, "backing_mix", 0.0, 100.0) / 100.0

        # 输出目录
        output_dir = ROOT_DIR / config.get("paths", {}).get(
            "outputs",
            config.get("output_dir", "outputs")
        )
        if run_directory is not None:
            output_dir = output_dir / run_directory
        elif reuse_state is not None:
            output_dir = output_dir / f"cover_{uuid.uuid4().hex}"

        display_name = get_character_filename_display(char_meta, resolved_name)
        literals.extend((display_name, model_info["model_path"], model_info.get("index_path"), output_dir))
        with console_literals(*literals):
            result = pipeline.process(
                input_audio=audio_path,
                input_mode=input_mode,
                model_path=model_info["model_path"],
                index_path=model_info.get("index_path"),
                pitch_shift=pitch_shift,
                index_ratio=index_ratio,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                speaker_id=speaker_id,
                f0_method=f0_method,
                demucs_model=demucs_model,
                demucs_shifts=demucs_shifts,
                demucs_overlap=demucs_overlap,
                demucs_split=demucs_split,
                roformer_model=roformer_model,
                separator=separator,
                uvr5_model=uvr5_model,
                uvr5_agg=uvr5_agg,
                uvr5_format=uvr5_format,
                use_official=use_official,
                silence_gate=silence_gate,
                silence_threshold_db=silence_threshold_db,
                silence_smoothing_ms=silence_smoothing_ms,
                silence_min_duration_ms=silence_min_duration_ms,
                vocals_volume=vocals_volume,
                accompaniment_volume=accompaniment_volume,
                reverb_amount=reverb_amount,
                backing_mix=backing_mix,
                karaoke_separation=karaoke_separation,
                karaoke_model=karaoke_model,
                karaoke_merge_backing_into_accompaniment=karaoke_merge_backing_into_accompaniment,
                source_constraint_mode=source_constraint_mode,
                vc_pipeline_mode=vc_pipeline_mode,
                output_dir=str(output_dir),
                model_display_name=display_name,
                output_name_suffixes={
                    "cover": t("final_cover", "cover"),
                    "vocals": re.sub(r"\s*[（(][^）)]*[）)]\s*$", "", t("input_vocals" if input_mode == "vocals" else "original_vocals", "cover")),
                    "converted_vocals": t("converted_vocals", "cover"),
                    "accompaniment": t("accompaniment", "cover"),
                    "accompaniment_without_harmony": t(
                        "accompaniment_without_harmony",
                        "cover",
                    ),
                    "lead_vocals": t("lead_vocals", "cover"),
                    "backing_vocals": t("backing_vocals", "cover"),
                },
                progress_callback=progress_callback,
                reuse_state=reuse_state.setdefault("songs", {}) if reuse_state is not None else None,
            )

        for output_key in (
            "cover",
            "converted_vocals",
            "vocals",
            "lead_vocals",
            "backing_vocals",
            "accompaniment",
            "accompaniment_without_harmony",
        ):
            if result.get(output_key):
                result[output_key] = normalize_download_output_path(result[output_key])

        return finish(
            "vocals_complete_status" if input_mode == "vocals" else "cover_complete_status",
            paths=(result.get("cover"), result["converted_vocals"], result.get("vocals"),
                   result.get("lead_vocals"), result.get("backing_vocals"), result.get("accompaniment"),
                   result.get("accompaniment_without_harmony")),
            stages=result["reused_stages"].split(",") if result.get("reused_stages") else [],
            route=dict(vc_pipeline_mode=vc_pipeline_mode, use_official=use_official, input_mode=input_mode),
            details=dict(model_version_status=char_meta.get("version_label"),
                         character_continuity_status=char_meta.get("continuity"),
                         model_source_status=char_meta.get("repo"), all_files_dir_status=result.get("all_files_dir")),
        )

    except Exception as e:
        import traceback
        error_msg = str(e)
        with console_literals(*literals):
            log.error(f"处理失败: {error_msg or traceback.format_exc()}")
        if not error_msg:
            cause = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
            error_msg = type(e).__name__
            if cause is not None and str(cause):
                error_msg += f": {cause}"
        return finish("cover_process_failed", fields={"error": error_msg}, literals=literals)


def _download_button_update(path: Optional[str]) -> Dict[str, Any]:
    return gr.update(value=path, visible=bool(path))


def _cover_download_button_updates(
    cover: Optional[str],
    converted_vocals: Optional[str],
    original_vocals: Optional[str],
    lead_vocals: Optional[str],
    backing_vocals: Optional[str],
    accompaniment: Optional[str],
    accompaniment_without_harmony: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return (
        _download_button_update(cover),
        _download_button_update(converted_vocals),
        _download_button_update(original_vocals),
        _download_button_update(lead_vocals),
        _download_button_update(backing_vocals),
        _download_button_update(accompaniment),
        _download_button_update(accompaniment_without_harmony),
    )


def update_route_controls(mode: str, karaoke: bool, source_mode: Optional[str] = None, merge_backing: Optional[bool] = None):
    """只更新可用性和明确的路由值，不触发处理。"""
    pipeline_labels, _ = get_vc_pipeline_mode_option_maps()
    source_choices, _ = get_source_constraint_option_maps()
    official = _resolve_labeled_choice(pipeline_labels, mode, "vc_pipeline_mode") == "official"
    active_karaoke = bool(karaoke) and not official
    return (
        gr.update(value=False if official else karaoke, interactive=not official),
        gr.update(value=(True if merge_backing is None else merge_backing) if active_karaoke else False, interactive=active_karaoke),
        gr.update(value="off" if official else _resolve_labeled_choice(
            source_choices, source_mode or "auto", "source_constraint_mode"), interactive=not official),
    )


def process_cover_with_downloads(
    audio_path: str,
    character_name: str,
    pitch_shift: int,
    index_ratio: float,
    speaker_id: float,
    karaoke_separation: bool,
    karaoke_merge_backing_into_accompaniment: bool,
    source_constraint_mode: str,
    vc_pipeline_mode: str,
    vocals_volume: float,
    accompaniment_volume: float,
    reverb_amount: float,
    rms_mix_rate: float,
    backing_mix: float,
    input_mode: str = "song",
    progress=gr.Progress(),
    reuse_state=None,
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    str,
]:
    result = process_cover(
        audio_path,
        character_name,
        pitch_shift,
        index_ratio,
        speaker_id,
        karaoke_separation,
        karaoke_merge_backing_into_accompaniment,
        source_constraint_mode,
        vc_pipeline_mode,
        vocals_volume,
        accompaniment_volume,
        reverb_amount,
        rms_mix_rate,
        backing_mix,
        input_mode=input_mode,
        progress=progress,
        reuse_state=reuse_state,
    )
    return _player_result(result)


def _player_result(result):
    *paths, status = result
    return *paths, *_cover_download_button_updates(*paths), status


def select_batch_result(selection, rows, reuse_state):
    if selection is None:
        reuse_state.pop("selected_status", None)
        return _player_result((None,) * 7 + ("",))
    row = rows[int(selection)]
    reuse_state["selected_status"] = row["status"]
    return _player_result((*row["result"][:7], cover_status_text(row["status"])))


def batch_summary(rows, archive_error=None):
    if not rows:
        return ""
    succeeded = sum(bool(row["result"][1]) for row in rows)
    lines = [tf("batch_summary", "cover", succeeded=succeeded, failed=len(rows) - succeeded)]
    for index, row in enumerate(rows, 1):
        result = row["result"]
        status = t("batch_done", "cover") if result[1] else cover_status_text(row["status"])
        lines.append(f'{index}. {row["name"]} — {status}')
    if archive_error:
        lines.append(tf("batch_archive_error", "cover", error=archive_error))
    return "\n".join(lines)


def process_cover_batch(
    audio_paths, character_name, pitch_shift, index_ratio, speaker_id,
    karaoke_separation, karaoke_merge_backing_into_accompaniment,
    source_constraint_mode, vc_pipeline_mode, vocals_volume, accompaniment_volume,
    reverb_amount, rms_mix_rate, backing_mix, input_mode, reuse_state,
    progress=gr.Progress(),
):
    reuse_state.pop("batch_archive_error", None)
    rows = []
    empty_picker = gr.update(choices=[], value=None, visible=False)
    empty_download = _download_button_update(None)
    if not audio_paths or not character_name:
        key = "please_upload_song" if not audio_paths else "please_select_character"
        yield rows, empty_picker, t(key, "messages"), empty_download
        return

    run_directory = Path(f"batch_{uuid.uuid4().hex}")
    total = len(audio_paths)
    for index, path in enumerate(audio_paths):
        name = Path(path).name
        current = tf("batch_current", "cover", number=index + 1, total=total, name=name)
        yield rows, empty_picker, "\n".join(filter(None, (batch_summary(rows), current))), empty_download

        def item_progress(fraction, desc):
            progress((index + fraction) / total, desc=f"{current} · {desc}")

        try:
            result = process_cover(
                path, character_name, pitch_shift, index_ratio, speaker_id,
                karaoke_separation, karaoke_merge_backing_into_accompaniment,
                source_constraint_mode, vc_pipeline_mode, vocals_volume, accompaniment_volume,
                reverb_amount, rms_mix_rate, backing_mix, input_mode,
                progress=item_progress, reuse_state=reuse_state,
                run_directory=run_directory / f"{index + 1:03d}",
            )
            status = reuse_state["selected_status"]
        except Exception as exc:
            # Preserve external diagnostics even if rendering that item's status failed.
            detail = str(exc) or type(exc).__name__
            status = dict(message="cover_process_failed", fields={"error": detail}, raw_error=True)
            with console_literals(detail):
                log.error(f"处理失败: {detail}")
            result = (None,) * 7 + (cover_status_text(status),)
        rows.append({"name": name, "result": result, "status": status})

    choices = [(f'{i + 1}. {row["name"]}', str(i)) for i, row in enumerate(rows) if row["result"][1]]
    download = empty_download
    if choices:
        # Each item's export directory is a child of this batch directory.
        first = rows[int(choices[0][1])]["result"]
        archive = Path(first[1]).parent.parent / "covers.zip"
        partial = archive.with_suffix(".zip.part")
        try:
            with zipfile.ZipFile(partial, "w") as bundle:
                for _, value in choices:
                    result = rows[int(value)]["result"]
                    track = Path(result[0] or result[1])
                    bundle.write(track, f"{int(value) + 1:03d}/{track.name}")
            partial.replace(archive)
            download = _download_button_update(str(archive))
        except (OSError, zipfile.LargeZipFile) as exc:
            reuse_state["batch_archive_error"] = str(exc)
            try:
                partial.unlink(missing_ok=True)
            except OSError as cleanup_error:
                reuse_state["batch_archive_error"] += f" ({cleanup_error})"
    picker = gr.update(choices=choices, value=choices[0][1] if choices else None, visible=bool(choices))
    yield rows, picker, batch_summary(rows, reuse_state.get("batch_archive_error")), download


def check_mature_deecho_status() -> str:
    """Report importability and file presence separately from inference results."""
    from tools.download_models import MATURE_DEECHO_MODELS, check_model, get_default_separator_asset_paths
    from infer.separator import ROFORMER_DEREVERB_DEFAULT_MODEL, check_roformer_available

    ready = check_roformer_available()
    lines = [t("component_available" if ready else "component_unavailable", "models")]
    paths = get_default_separator_asset_paths(ROOT_DIR)["RoFormer De-Reverb Stereo"]
    for path in paths:
        state = t("file_present" if path.is_file() else "file_missing", "models")
        lines.append(f"{state}: {path.name}")
    if not all(path.is_file() for path in paths):
        lines.append(t("mature_roformer_auto_download_note", "route_status"))
    for name in MATURE_DEECHO_MODELS:
        state = t("optional_present" if check_model(name) else "optional_missing", "models")
        lines.append(f"{state}: {name}")
    if not ready:
        lines.append(t("mature_missing_strict", "route_status"))
    lines.append(t("status_scope", "models"))
    return "\n".join(lines)


def download_mature_deecho_models_ui() -> str:
    """Download mature DeEcho models."""
    from tools.download_models import download_mature_deecho_models

    try:
        success = download_mature_deecho_models()
        status = check_mature_deecho_status()
        prefix = t("download_complete_status", "messages") if success else t("download_warning_status", "messages")
        return f"{prefix}\n\n{status}"
    except Exception as e:
        return tf("download_error_status", "messages", error=str(e))


def get_cover_vc_route_status(
    vc_pipeline_mode: Optional[str] = None,
    use_official: Optional[bool] = None,
    karaoke_separation: Optional[bool] = None,
    input_mode: str = "song",
) -> str:
    """Return the active VC route shown in the cover UI."""
    if input_mode == "vocals":
        return t("vocals_route", "route_status")
    from infer.separator import ROFORMER_DEREVERB_DEFAULT_MODEL, check_roformer_available

    cover_cfg = config.get("cover", {})
    pipeline_label_to_value, _ = get_vc_pipeline_mode_option_maps()
    pipeline_mode = _resolve_labeled_choice(
        pipeline_label_to_value,
        vc_pipeline_mode if vc_pipeline_mode is not None else cover_cfg.get("vc_pipeline_mode", "current"),
        "vc_pipeline_mode",
    )
    effective_use_official = bool(cover_cfg.get("use_official", True)) if use_official is None else bool(use_official)
    roformer_ready = check_roformer_available()
    preferred = f"RoFormer {ROFORMER_DEREVERB_DEFAULT_MODEL}" if roformer_ready else None
    newline = chr(10)
    component_status = t("component_available" if roformer_ready else "component_unavailable", "models")

    if pipeline_mode == "official":
        return newline.join([
            t("official_route_title", "route_status"),
            t("official_route_flow", "route_status"),
            t("official_route_note", "route_status"),
            component_status,
        ])

    if pipeline_mode == "current" and effective_use_official:
        karaoke_enabled = cover_cfg.get("karaoke_separation", True) if karaoke_separation is None else karaoke_separation
        actual_flow = t("quality_default_route_flow", "route_status")
        if (not karaoke_enabled or cover_cfg.get("separator", "roformer") != "roformer"
                or cover_cfg.get("roformer_model", ROFORMER_DEFAULT_MODEL) != ROFORMER_DEFAULT_MODEL):
            selected_separator = cover_cfg.get("separator", "roformer")
            separation_model = cover_cfg.get({"roformer": "roformer_model", "uvr5": "uvr5_model", "demucs": "demucs_model"}[selected_separator], selected_separator)
            actual_flow = tf("configured_route_flow", "route_status", separator=separation_model,
                             karaoke="MVSep 9205" if karaoke_enabled else t("disabled", "ui"))
        return newline.join([
            t("quality_default_route_title", "route_status"),
            actual_flow,
            t("quality_default_route_note", "route_status"),
            component_status,
        ])

    karaoke_enabled = cover_cfg.get("karaoke_separation", True) if karaoke_separation is None else karaoke_separation
    selected_separator = cover_cfg.get("separator", "roformer")
    separation_model = cover_cfg.get({"roformer": "roformer_model", "uvr5": "uvr5_model", "demucs": "demucs_model"}[selected_separator], selected_separator)
    return newline.join([
        t("strict_route_ready_title" if preferred else "strict_route_unavailable_title", "route_status"),
        tf("route_current_model", "route_status", model=preferred) if preferred else t("strict_route_unavailable_advice", "route_status"),
        tf("local_configured_route_flow", "route_status", separator=separation_model,
           karaoke="MVSep 9205" if karaoke_enabled else t("disabled", "ui")),
        component_status,
    ])


def check_models_status() -> str:
    """Show the scope and result of each local asset check."""
    from tools.download_models import (
        check_default_separator_models, check_model, get_missing_upstream_rvc_files, REQUIRED_MODELS,
    )
    lines = []
    for name in REQUIRED_MODELS:
        state = t("validation_passed" if check_model(name) else "validation_failed", "models")
        lines.append(f"{name}: {state}")
    problems = get_missing_upstream_rvc_files()
    state = t("validation_failed" if problems else "validation_passed", "models")
    lines.append(f"{t('official_runtime_status', 'models')}: {state}")
    lines.extend(str(problem) for problem in problems)
    lines.extend(["", t("default_separator_models", "models")])
    for name, exists in check_default_separator_models().items():
        state = t("file_present" if exists else "file_missing", "models")
        lines.append(f"{name}: {state}")
    lines.extend(["", t("status_scope", "models")])
    return "\n".join(lines)


def get_device_info() -> str:
    """获取设备信息"""
    import torch
    from lib.device import get_device_info as _get_info, _is_rocm, _has_xpu, _has_directml, _has_mps

    lines = []
    lines.append(tf("pytorch_version", "device_info", version=torch.__version__))

    info = _get_info()
    lines.append(tf("available_backends", "device_info", backends=", ".join(info["backends"])))

    for dev in info["devices"]:
        mem = f"{dev['total_memory_gb']} GB" if dev.get("total_memory_gb") else "N/A"
        lines.append(tf("gpu_line", "device_info", name=dev["name"], backend=dev["backend"], memory=mem))

    if torch.cuda.is_available():
        ver = torch.version.hip if _is_rocm() else torch.version.cuda
        label = "ROCm" if _is_rocm() else "CUDA"
        lines.append(tf("backend_version", "device_info", label=label, version=ver))

    if not info["devices"]:
        lines.append(t("no_gpu_cpu", "device_info"))

    return "\n".join(lines)


# Precision Audio Workbench CSS (Light & Dark, zero nested card frames)
CUSTOM_CSS = """
:root,
:root[data-rvc-theme="light"] {
    color-scheme: light;
    --rvc-bg: #f5f6f8;
    --rvc-surface: #ffffff;
    --rvc-surface-subtle: #fafbfc;
    --rvc-surface-inset: #f2f3f6;
    --rvc-surface-hover: #eaecf0;
    --rvc-border: #e2e5ea;
    --rvc-border-strong: #cdd2dc;
    --rvc-text: #111318;
    --rvc-text-secondary: #4a505c;
    --rvc-text-muted: #707685;
    --rvc-accent: #d4600a;
    --rvc-accent-hover: #b84f06;
    --rvc-accent-active: #984004;
    --rvc-accent-soft: rgba(212, 96, 10, 0.08);
    --rvc-accent-border: rgba(212, 96, 10, 0.36);
    --rvc-on-accent: #ffffff;
    --rvc-focus-ring: rgba(212, 96, 10, 0.22);
    --rvc-table-head: #f3f4f7;
    --rvc-table-odd: #ffffff;
    --rvc-table-even: #fafbfc;
    --rvc-table-hover: #eff1f5;
    --rvc-code-bg: #f0f2f5;
    --rvc-code-text: #272b35;
    --rvc-seg-active: #ffffff;
    --rvc-error-bg: #fef2f2;
    --rvc-error-border: #fca5a5;
    --rvc-error-text: #991b1b;
    --rvc-radius: 4px;
    --rvc-radius-sm: 3px;
    --rvc-transition: 140ms cubic-bezier(0.2, 0, 0, 1);
}

:root[data-rvc-theme="dark"],
.dark {
    color-scheme: dark;
    --rvc-bg: #0b0c0f;
    --rvc-surface: #12141a;
    --rvc-surface-subtle: #0f1116;
    --rvc-surface-inset: #181b23;
    --rvc-surface-hover: #212530;
    --rvc-border: #242834;
    --rvc-border-strong: #343a49;
    --rvc-text: #f3f4f6;
    --rvc-text-secondary: #aeb5c4;
    --rvc-text-muted: #788094;
    --rvc-accent: #e59225;
    --rvc-accent-hover: #f0a43b;
    --rvc-accent-active: #cf7d15;
    --rvc-accent-soft: rgba(229, 146, 37, 0.13);
    --rvc-accent-border: rgba(229, 146, 37, 0.42);
    --rvc-on-accent: #120d06;
    --rvc-focus-ring: rgba(229, 146, 37, 0.28);
    --rvc-table-head: #181b23;
    --rvc-table-odd: #12141a;
    --rvc-table-even: #151820;
    --rvc-table-hover: #1e232e;
    --rvc-code-bg: #1b1f29;
    --rvc-code-text: #d7dce6;
    --rvc-seg-active: #242835;
    --rvc-error-bg: #261215;
    --rvc-error-border: #7f1d1d;
    --rvc-error-text: #fecaca;
    --rvc-radius: 4px;
    --rvc-radius-sm: 3px;
    --rvc-transition: 140ms cubic-bezier(0.2, 0, 0, 1);
}

/* Base Canvas */
html,
body {
    background-color: var(--rvc-bg) !important;
    color: var(--rvc-text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    background: var(--rvc-bg) !important;
    color: var(--rvc-text) !important;
    min-height: 100vh;
    --body-background-fill: var(--rvc-bg);
    --background-fill-primary: var(--rvc-surface);
    --background-fill-secondary: var(--rvc-surface-subtle);
    --block-background-fill: var(--rvc-surface);
    --block-border-color: var(--rvc-border);
    --block-border-width: 0px;
    --block-shadow: none;
    --block-radius: var(--rvc-radius);
    --container-radius: var(--rvc-radius);
    --input-radius: var(--rvc-radius);
    --button-large-radius: var(--rvc-radius);
    --button-small-radius: var(--rvc-radius-sm);
    --block-label-background-fill: var(--rvc-surface);
    --block-label-border-color: var(--rvc-border);
    --block-label-text-color: var(--rvc-text-secondary);
    --block-title-text-color: var(--rvc-text);
    --border-color-primary: var(--rvc-border);
    --border-color-accent: var(--rvc-accent);
    --loader-color: var(--rvc-accent);
    --error-background-fill: var(--rvc-error-bg);
    --error-border-color: var(--rvc-error-border);
    --error-text-color: var(--rvc-error-text);
    --input-background-fill: var(--rvc-surface-inset);
    --input-background-fill-focus: var(--rvc-surface);
    --input-border-color: var(--rvc-border-strong);
    --input-border-color-focus: var(--rvc-accent);
    --input-placeholder-color: var(--rvc-text-muted);
    --body-text-color: var(--rvc-text);
    --body-text-color-subdued: var(--rvc-text-muted);
    --color-accent: var(--rvc-accent);
    --color-accent-soft: var(--rvc-accent-soft);
    --link-text-color: var(--rvc-accent);
    --link-text-color-hover: var(--rvc-accent-hover);
    --slider-color: var(--rvc-accent);
    --checkbox-background-color: var(--rvc-surface-inset);
    --checkbox-background-color-selected: var(--rvc-accent);
    --checkbox-border-color: var(--rvc-border-strong);
    --checkbox-border-color-selected: var(--rvc-accent);
    --checkbox-label-background-fill: var(--rvc-surface-inset);
    --checkbox-label-background-fill-selected: var(--rvc-accent-soft);
    --checkbox-label-text-color: var(--rvc-text);
    --checkbox-label-text-color-selected: var(--rvc-accent);
    --table-even-background-fill: var(--rvc-table-even);
    --table-odd-background-fill: var(--rvc-table-odd);
    --table-row-focus: var(--rvc-table-hover);
    --table-border-color: var(--rvc-border);
    --button-secondary-background-fill: var(--rvc-surface);
    --button-secondary-background-fill-hover: var(--rvc-surface-hover);
    --button-secondary-border-color: var(--rvc-border-strong);
    --button-secondary-text-color: var(--rvc-text);
    --shadow-drop: none;
    --shadow-drop-lg: none;
}

.gradio-container main {
    width: 100% !important;
    max-width: 1420px !important;
    margin-inline: auto !important;
    padding: 0 24px 36px !important;
    box-sizing: border-box;
}

.gradio-container footer,
.gradio-container .built-with {
    display: none !important;
}

/* Strip Gradio default block boxes to eliminate double borders everywhere */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
.gradio-container .form {
    gap: 12px !important;
    overflow: visible !important;
}

/* Typography & Prose */
.gradio-container .prose {
    color: var(--rvc-text) !important;
    line-height: 1.6;
    max-width: none !important;
}
.gradio-container .prose :is(p, li, a, code) {
    overflow-wrap: anywhere;
}
.gradio-container .prose :is(h1, h2, h3, h4) {
    color: var(--rvc-text) !important;
    line-height: 1.35;
    letter-spacing: -0.015em;
}
.gradio-container .prose a {
    color: var(--rvc-accent) !important;
    text-decoration: none;
    font-weight: 500;
}
.gradio-container .prose a:hover {
    color: var(--rvc-accent-hover) !important;
    text-decoration: underline;
}
.gradio-container .prose code {
    background: var(--rvc-code-bg) !important;
    color: var(--rvc-code-text) !important;
    border: 1px solid var(--rvc-border) !important;
    padding: 1px 5px !important;
    border-radius: var(--rvc-radius-sm) !important;
    font-size: 0.85em !important;
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace !important;
}
.gradio-container .prose blockquote {
    border-left: 2px solid var(--rvc-border-strong) !important;
    background: transparent !important;
    padding: 4px 0 4px 12px !important;
    color: var(--rvc-text-muted) !important;
    margin: 10px 0 !important;
}

.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container select:focus-visible {
    outline: 2px solid var(--rvc-accent) !important;
    outline-offset: 1px !important;
}

.gradio-container [role="option"] {
    white-space: normal !important;
    overflow-wrap: anywhere;
    padding: 8px 12px !important;
    line-height: 1.45;
}
.gradio-container .row > .column {
    min-width: 0;
}

/* Flush Single-Row App Topbar */
.gradio-container .top-header {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 14px 0 12px !important;
    background: transparent !important;
    border-bottom: 1px solid var(--rvc-border) !important;
}
.gradio-container .top-header > .top-brand {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: auto !important;
    gap: 0 !important;
}
.gradio-container .top-header > .top-actions {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    width: auto !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
    gap: 0 !important;
}
.top-brand-wrap {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.top-brand-row {
    display: flex;
    align-items: baseline;
    gap: 9px;
}
.top-brand-title {
    color: var(--rvc-text) !important;
    font-size: 19px !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em;
}
.top-brand-ver {
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
    color: var(--rvc-text-muted);
    background: var(--rvc-surface);
    border: 1px solid var(--rvc-border-strong);
    padding: 1px 6px;
    border-radius: var(--rvc-radius-sm);
    line-height: 1.35;
}
.top-brand-subtitle {
    color: var(--rvc-text-muted) !important;
    font-size: 12px !important;
    line-height: 1.4;
}

/* Strictly Aligned 34px Single-Row Top Toolbar */
.gradio-container .top-controls-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 10px !important;
    margin: 0 !important;
    width: auto !important;
}
.gradio-container .theme-toggle-slot {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
.gradio-container .theme-toggle-slot > .html-container,
.gradio-container .theme-toggle-slot > .prose {
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
}
.gradio-container button.rvc-theme-btn {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 12px !important;
    margin: 0 !important;
    border-radius: var(--rvc-radius) !important;
    border: 1px solid var(--rvc-border-strong) !important;
    background: var(--rvc-surface) !important;
    color: var(--rvc-text) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
    transition: background var(--rvc-transition), border-color var(--rvc-transition), color var(--rvc-transition) !important;
}
.gradio-container button.rvc-theme-btn:hover {
    background: var(--rvc-surface-hover) !important;
    border-color: var(--rvc-accent) !important;
    color: var(--rvc-text) !important;
}
.gradio-container .rvc-theme-icon {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 14px !important;
    height: 14px !important;
    flex: 0 0 auto !important;
    color: var(--rvc-accent) !important;
}
.gradio-container .rvc-theme-text {
    display: inline-block !important;
    line-height: 1 !important;
}

.gradio-container .language-switch-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 8px !important;
    margin: 0 !important;
    padding-left: 10px !important;
    border-left: 1px solid var(--rvc-border) !important;
    flex: 0 0 auto !important;
    width: auto !important;
}
.gradio-container .language-switch-row > .block,
.gradio-container .language-switch-row > .form {
    width: auto !important;
    min-width: 0 !important;
    flex: 0 0 auto !important;
}
.gradio-container .language-switch-row label {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 0 !important;
    margin: 0 !important;
    height: 34px !important;
}
.gradio-container .language-status-note {
    min-height: 0 !important;
    margin: 0 !important;
    text-align: right;
}
.gradio-container .language-status-note:empty,
.gradio-container .language-status-note.block:has(.prose:empty) {
    display: none !important;
}
.gradio-container .language-status-note p {
    margin: 4px 0 0 !important;
    color: var(--rvc-accent) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    line-height: 1.4;
}

/* Primary Top-Level Underline Navigation */
.gradio-container .rvc-main-tabs > .tab-wrapper > .tab-container,
.gradio-container .rvc-main-tabs > .tab-nav,
.gradio-container .rvc-main-tabs > div[role="tablist"] {
    background: transparent !important;
    border: 0 !important;
    border-bottom: 1px solid var(--rvc-border) !important;
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 4px !important;
    min-height: 42px !important;
}
.gradio-container .rvc-main-tabs > .tab-wrapper > .tab-container > button,
.gradio-container .rvc-main-tabs > .tab-nav > button,
.gradio-container .rvc-main-tabs > div[role="tablist"] > button {
    color: var(--rvc-text-muted) !important;
    background: transparent !important;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    min-height: 42px !important;
    padding: 9px 16px 7px !important;
    margin-bottom: -1px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: color var(--rvc-transition), border-color var(--rvc-transition) !important;
}
.gradio-container .rvc-main-tabs > .tab-wrapper > .tab-container > button:hover,
.gradio-container .rvc-main-tabs > .tab-nav > button:hover,
.gradio-container .rvc-main-tabs > div[role="tablist"] > button:hover {
    color: var(--rvc-text) !important;
    border-bottom-color: var(--rvc-border-strong) !important;
}
.gradio-container .rvc-main-tabs > .tab-wrapper > .tab-container > button.selected,
.gradio-container .rvc-main-tabs > .tab-wrapper > .tab-container > button[aria-selected="true"],
.gradio-container .rvc-main-tabs > .tab-nav > button.selected,
.gradio-container .rvc-main-tabs > .tab-nav > button[aria-selected="true"],
.gradio-container .rvc-main-tabs > div[role="tablist"] > button.selected,
.gradio-container .rvc-main-tabs > div[role="tablist"] > button[aria-selected="true"] {
    color: var(--rvc-text) !important;
    font-weight: 700 !important;
    border-bottom-color: var(--rvc-accent) !important;
}
.gradio-container .rvc-main-tabs > [role="tabpanel"] {
    min-width: 0;
    padding-top: 14px !important;
}

/* Integrated Compact Sub-Tabs (Single Track / Batch Cover) */
.gradio-container .rvc-subtabs {
    margin-top: 2px !important;
}
.gradio-container .rvc-subtabs > .tab-wrapper {
    height: auto !important;
    padding: 0 !important;
    margin-bottom: 10px !important;
    width: 100% !important;
}
.gradio-container .rvc-subtabs > .tab-wrapper > .tab-container,
.gradio-container .rvc-subtabs > .tab-nav,
.gradio-container .rvc-subtabs > div[role="tablist"] {
    display: flex !important;
    width: 100% !important;
    height: 34px !important;
    min-height: 34px !important;
    background: transparent !important;
    border: 0 !important;
    border-bottom: 1px solid var(--rvc-border) !important;
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 6px !important;
}
.gradio-container .rvc-subtabs > .tab-wrapper > .tab-container::after {
    display: none !important;
}
.gradio-container .rvc-subtabs > .tab-wrapper > .tab-container > button,
.gradio-container .rvc-subtabs > .tab-nav > button,
.gradio-container .rvc-subtabs > div[role="tablist"] > button {
    flex: 0 0 auto !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 12px !important;
    margin: 0 0 -1px 0 !important;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--rvc-text-muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    box-shadow: none !important;
    transition: color var(--rvc-transition), border-color var(--rvc-transition) !important;
}
.gradio-container .rvc-subtabs > .tab-wrapper > .tab-container > button::after {
    display: none !important;
}
.gradio-container .rvc-subtabs > .tab-wrapper > .tab-container > button:hover,
.gradio-container .rvc-subtabs > .tab-nav > button:hover,
.gradio-container .rvc-subtabs > div[role="tablist"] > button:hover {
    color: var(--rvc-text) !important;
    background: transparent !important;
    border-bottom-color: var(--rvc-border-strong) !important;
}
.gradio-container .rvc-subtabs > .tab-wrapper > .tab-container > button.selected,
.gradio-container .rvc-subtabs > .tab-nav > button.selected,
.gradio-container .rvc-subtabs > div[role="tablist"] > button.selected {
    background: transparent !important;
    border-bottom-color: var(--rvc-accent) !important;
    color: var(--rvc-text) !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}
.gradio-container .rvc-subtabs > [role="tabpanel"] {
    padding-top: 2px !important;
    gap: 10px !important;
}

/* Inline Minimal Guide Line */
.rvc-guide-line {
    margin: 0 0 12px !important;
}
.rvc-guide-line .prose p {
    margin: 0 !important;
    font-size: 12.5px !important;
    color: var(--rvc-text-muted) !important;
    line-height: 1.5 !important;
}
.rvc-guide-line .prose strong {
    color: var(--rvc-text-secondary) !important;
    font-weight: 600 !important;
}

/* Precision Architectural Grids */
.rvc-workbench {
    display: grid !important;
    grid-template-columns: minmax(0, 1.24fr) minmax(320px, 1fr);
    gap: 16px !important;
    align-items: start !important;
}
.rvc-sticky-col {
    position: sticky !important;
    top: 14px !important;
}
.rvc-two-col-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px !important;
    align-items: start !important;
}

.rvc-panel {
    background: var(--rvc-surface) !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: var(--rvc-radius) !important;
    padding: 18px 20px !important;
    gap: 12px !important;
    min-width: 0 !important;
}
.rvc-panel-stack {
    margin-top: 16px !important;
}

/* Clean Section Headers */
.rvc-head-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px !important;
    padding-bottom: 9px !important;
    border-bottom: 1px solid var(--rvc-border) !important;
    margin: 0 !important;
}
.rvc-head-row-spaced {
    margin-top: 6px !important;
}
.rvc-head-title .prose :is(h3, h4) {
    margin: 0 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: var(--rvc-text) !important;
    letter-spacing: 0.01em;
}
.gradio-container button.rvc-head-btn,
.gradio-container .rvc-head-btn button {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    height: 28px !important;
    min-height: 28px !important;
    padding: 0 11px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    line-height: 1 !important;
    border-radius: var(--rvc-radius-sm) !important;
}

.rvc-subhead .prose :is(h4, h5) {
    margin: 4px 0 2px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    color: var(--rvc-text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.rvc-muted-copy .prose p {
    margin: 0 !important;
    font-size: 12.5px !important;
    color: var(--rvc-text-muted) !important;
    line-height: 1.5 !important;
}

.rvc-inline-note {
    padding: 8px 10px !important;
    background: var(--rvc-surface-subtle) !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: var(--rvc-radius) !important;
}
.rvc-inline-note .prose p {
    margin: 0 !important;
    font-size: 12px !important;
    color: var(--rvc-text-secondary) !important;
    line-height: 1.5 !important;
}

/* Compact Character Metadata Inspector */
.rvc-detail-box {
    padding: 10px 13px !important;
    background: var(--rvc-surface-subtle) !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: var(--rvc-radius) !important;
}
.rvc-detail-box .prose {
    font-size: 12px !important;
    line-height: 1.55 !important;
}
.rvc-detail-box .prose > p:first-child {
    margin: 0 0 6px !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid var(--rvc-border) !important;
    font-size: 13px !important;
    color: var(--rvc-text) !important;
}
.rvc-detail-box .prose ul {
    margin: 0 !important;
    padding: 0 !important;
    list-style: none !important;
}
.rvc-detail-box .prose li {
    margin: 2px 0 !important;
    color: var(--rvc-text-secondary) !important;
}
.rvc-detail-box .prose code {
    word-break: break-all !important;
}

.rvc-action-divider {
    padding-top: 12px !important;
    border-top: 1px solid var(--rvc-border) !important;
    gap: 10px !important;
}

/* Labels & Field Descriptions (Unified flex column & height for 0px row alignment) */
.gradio-container label.container {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 0 !important;
}
.gradio-container label > span:first-child,
.gradio-container label > span[data-testid="block-info"],
.gradio-container label.container > span[data-testid="block-info"],
.gradio-container fieldset > legend,
.gradio-container fieldset > span[data-testid="block-info"],
.gradio-container span[data-testid="block-info"] {
    display: block !important;
    color: var(--rvc-text-secondary) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    line-height: 18px !important;
    min-height: 18px !important;
    padding: 0 !important;
    margin: 0 0 6px 0 !important;
}
.gradio-container .gr-info,
.gradio-container span.info {
    display: block !important;
    color: var(--rvc-text-muted) !important;
    font-size: 11.5px !important;
    line-height: 1.45 !important;
    margin: -2px 0 6px 0 !important;
}

/* Single-Layer Crisp Text Inputs & Textareas */
.gradio-container label.container input[data-testid="textbox"],
.gradio-container input[data-testid="textbox"],
.gradio-container input[type="text"]:not([role="listbox"]) {
    height: 38px !important;
    min-height: 38px !important;
    max-height: 38px !important;
    padding: 0 11px !important;
    margin: 0 !important;
    line-height: normal !important;
    box-sizing: border-box !important;
}
.gradio-container input[type="number"] {
    height: 28px !important;
    min-height: 28px !important;
    max-height: 28px !important;
    padding: 0 8px !important;
    margin: 0 !important;
    font-size: 12.5px !important;
    line-height: 1 !important;
    text-align: center !important;
    box-sizing: border-box !important;
}
.gradio-container textarea,
.gradio-container input[type="text"]:not([role="listbox"]),
.gradio-container input[type="number"] {
    background: var(--rvc-surface-inset) !important;
    border: 1px solid var(--rvc-border-strong) !important;
    border-radius: var(--rvc-radius) !important;
    color: var(--rvc-text) !important;
    font-size: 13px !important;
    transition: border-color var(--rvc-transition), background var(--rvc-transition), box-shadow var(--rvc-transition);
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:not([role="listbox"]):focus,
.gradio-container input[type="number"]:focus {
    border-color: var(--rvc-accent) !important;
    background: var(--rvc-surface) !important;
    box-shadow: 0 0 0 2px var(--rvc-focus-ring) !important;
    outline: none !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: var(--rvc-text-muted) !important;
    opacity: 0.9;
}

/* Single-Layer Monospace Status Consoles */
.status-box textarea,
.status-box input {
    font-family: ui-monospace, "Cascadia Mono", Consolas, "SFMono-Regular", monospace !important;
    line-height: 1.58 !important;
    font-size: 12px !important;
    padding: 9px 11px !important;
    background: var(--rvc-surface-subtle) !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: var(--rvc-radius) !important;
    color: var(--rvc-text-secondary) !important;
    overflow-wrap: anywhere;
    white-space: pre-wrap !important;
    overflow: auto !important;
    box-sizing: border-box !important;
}
.cover-progress-status textarea {
    min-height: 82px !important;
}
@supports (field-sizing: content) {
    .status-box textarea {
        field-sizing: content;
        height: auto !important;
        min-height: 74px;
        max-height: 240px !important;
        overflow-y: auto !important;
        resize: none;
    }
}

/* Single-Layer Crisp Dropdowns (zero double borders, zero arrow overlap) */
.gradio-container .wrap:has(> .wrap-inner) {
    background: var(--rvc-surface-inset) !important;
    border: 1px solid var(--rvc-border-strong) !important;
    border-radius: var(--rvc-radius) !important;
    height: 38px !important;
    min-height: 38px !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
    overflow: visible !important;
    transition: border-color var(--rvc-transition), background var(--rvc-transition), box-shadow var(--rvc-transition);
}
.gradio-container .wrap:has(> .wrap-inner):focus-within {
    border-color: var(--rvc-accent) !important;
    background: var(--rvc-surface) !important;
    box-shadow: 0 0 0 2px var(--rvc-focus-ring) !important;
}
.gradio-container .wrap > .wrap-inner {
    display: flex !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 10px !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
}
.gradio-container .wrap > .wrap-inner > .secondary-wrap {
    display: flex !important;
    align-items: center !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 6px !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}
.gradio-container .wrap > .wrap-inner > .secondary-wrap input {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    font-size: 13px !important;
    color: var(--rvc-text) !important;
}
.gradio-container .wrap > .wrap-inner > .secondary-wrap :is(.icon-wrap, svg, button) {
    flex: 0 0 auto !important;
    position: static !important;
    top: auto !important;
    right: auto !important;
    transform: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    align-self: center !important;
    margin: 0 !important;
    color: var(--rvc-text-muted) !important;
}
.gradio-container .wrap > .wrap-inner > .secondary-wrap .icon-wrap {
    width: 18px !important;
    height: 18px !important;
}
.gradio-container .wrap > .wrap-inner > .secondary-wrap .icon-wrap svg {
    width: 15px !important;
    height: 15px !important;
    display: block !important;
}

/* Language Dropdown Topbar Exact 34px Override */
.gradio-container .language-switch-row .wrap:has(> .wrap-inner) {
    width: 110px !important;
    min-width: 110px !important;
    height: 34px !important;
    min-height: 34px !important;
    background: var(--rvc-surface) !important;
}
.gradio-container .language-switch-row .wrap > .wrap-inner {
    padding: 0 9px !important;
}
.gradio-container .language-switch-row .wrap > .wrap-inner > .secondary-wrap input {
    font-size: 12.5px !important;
}

.gradio-container ul[role="listbox"],
.gradio-container .options {
    background: var(--rvc-surface) !important;
    border: 1px solid var(--rvc-border-strong) !important;
    border-radius: var(--rvc-radius) !important;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.16) !important;
    color: var(--rvc-text) !important;
    z-index: 60 !important;
}
.gradio-container li[role="option"] {
    background: transparent !important;
    color: var(--rvc-text) !important;
    font-size: 13px !important;
}
.gradio-container li[role="option"]:hover {
    background: var(--rvc-surface-hover) !important;
    color: var(--rvc-accent) !important;
}
.gradio-container li[role="option"][aria-selected="true"],
.gradio-container li[role="option"].selected {
    background: var(--rvc-accent-soft) !important;
    color: var(--rvc-accent) !important;
    font-weight: 600 !important;
}

/* Buttons (Crisp Solid Accent & Hairline Secondary) */
.gradio-container button.primary,
.gradio-container .gr-button-primary {
    background: var(--rvc-accent) !important;
    border: 1px solid var(--rvc-accent) !important;
    border-radius: var(--rvc-radius) !important;
    color: var(--rvc-on-accent) !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    min-height: 38px !important;
    padding: 0 16px !important;
    box-shadow: none !important;
    transition: background var(--rvc-transition), border-color var(--rvc-transition) !important;
}
.gradio-container button.primary:hover:not(:disabled),
.gradio-container .gr-button-primary:hover:not(:disabled) {
    background: var(--rvc-accent-hover) !important;
    border-color: var(--rvc-accent-hover) !important;
}
.gradio-container button.primary:active:not(:disabled),
.gradio-container .gr-button-primary:active:not(:disabled) {
    background: var(--rvc-accent-active) !important;
}
.gradio-container .rvc-primary-cta button,
.gradio-container button.rvc-primary-cta {
    min-height: 44px !important;
    font-size: 14.5px !important;
    letter-spacing: 0.02em;
}
.gradio-container .rvc-compact-btn,
.gradio-container button.rvc-compact-btn {
    width: auto !important;
    min-width: 144px !important;
    align-self: flex-start !important;
}

.gradio-container button.secondary,
.gradio-container .gr-button-secondary {
    background: var(--rvc-surface) !important;
    border: 1px solid var(--rvc-border-strong) !important;
    border-radius: var(--rvc-radius) !important;
    color: var(--rvc-text) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    min-height: 36px !important;
    padding: 0 14px !important;
    box-shadow: none !important;
    transition: background var(--rvc-transition), border-color var(--rvc-transition), color var(--rvc-transition) !important;
}
.gradio-container button.secondary:hover:not(:disabled),
.gradio-container .gr-button-secondary:hover:not(:disabled) {
    background: var(--rvc-surface-hover) !important;
    border-color: var(--rvc-accent) !important;
    color: var(--rvc-accent) !important;
}

.gradio-container button.primary:disabled,
.gradio-container button.secondary:disabled {
    opacity: 0.52 !important;
    cursor: not-allowed !important;
}

/* Language Apply Button Exact 34px Override (placed after generic buttons) */
.gradio-container .language-switch-row button,
.gradio-container .language-switch-row button.secondary {
    height: 34px !important;
    min-height: 34px !important;
    width: auto !important;
    min-width: 0 !important;
    padding: 0 13px !important;
    margin: 0 !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    line-height: 1 !important;
    border-radius: var(--rvc-radius) !important;
    white-space: nowrap !important;
    box-sizing: border-box !important;
}

/* Segmented Radio Controls (Light & Dark High-Contrast) */
.gradio-container .wrap:has(> label > input[type="radio"]) {
    display: inline-flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 3px !important;
    padding: 3px !important;
    width: fit-content !important;
    background: var(--rvc-surface-inset) !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: 5px !important;
    box-shadow: none !important;
}
.gradio-container .wrap:has(> label > input[type="radio"]) > label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 7px !important;
    margin: 0 !important;
    padding: 5px 12px !important;
    min-height: 30px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--rvc-radius-sm) !important;
    color: var(--rvc-text-secondary) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: background var(--rvc-transition), border-color var(--rvc-transition), color var(--rvc-transition);
}
.gradio-container .wrap:has(> label > input[type="radio"]) > label:hover {
    color: var(--rvc-text) !important;
    background: var(--rvc-surface-hover) !important;
}
.gradio-container .wrap:has(> label > input[type="radio"]) > label.selected,
.gradio-container .wrap:has(> label > input[type="radio"]) > label:has(input:checked) {
    background: var(--rvc-seg-active) !important;
    border-color: var(--rvc-accent-border) !important;
    color: var(--rvc-text) !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
}
.gradio-container .wrap:has(> label > input[type="radio"]) > label span {
    color: inherit !important;
    margin: 0 !important;
    font-weight: inherit !important;
}
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"],
.gradio-container input[type="range"] {
    accent-color: var(--rvc-accent) !important;
}

/* Sliders */
.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    background: var(--rvc-border-strong);
    height: 4px;
    border-radius: 2px;
}
.gradio-container input[type="range"]::-moz-range-track {
    background: var(--rvc-border-strong);
    height: 4px;
    border-radius: 2px;
}

/* Clean Hairline Accordions (overrides Gradio inline overflow:auto on .block) */
.gradio-container .block:has(> button.label-wrap),
.gradio-container .block:has(> .accordion),
.gradio-container .accordion,
.gradio-container .gr-accordion {
    background: var(--rvc-surface-subtle) !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: var(--rvc-radius) !important;
    overflow: hidden !important;
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    padding: 0 !important;
}
.gradio-container .block:has(> button.label-wrap) > .wrap,
.gradio-container .accordion > .wrap {
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    background: transparent !important;
    overflow: hidden !important;
}
.gradio-container .block:has(> button.label-wrap) > button.label-wrap,
.gradio-container .accordion > button.label-wrap,
.gradio-container .gr-accordion > .label-wrap {
    box-sizing: border-box !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 10px 14px !important;
    background: transparent !important;
    border: 0 !important;
    color: var(--rvc-text) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    overflow: hidden !important;
}
.gradio-container .block:has(> button.label-wrap) > button.label-wrap.open,
.gradio-container .accordion > button.label-wrap.open {
    margin-bottom: 0 !important;
}
.gradio-container .block:has(> button.label-wrap) > button.label-wrap:hover,
.gradio-container .accordion > button.label-wrap:hover,
.gradio-container .gr-accordion > .label-wrap:hover {
    background: var(--rvc-surface-hover) !important;
}
.gradio-container .block:has(> button.label-wrap) > button.label-wrap .icon,
.gradio-container .accordion > button.label-wrap .icon {
    flex: 0 0 auto !important;
    font-size: 10px !important;
    line-height: 1 !important;
    color: var(--rvc-text-muted) !important;
    margin-right: 2px !important;
}
.gradio-container .block:has(> button.label-wrap) > button.label-wrap.open + div,
.gradio-container .accordion > button.label-wrap.open + div {
    padding: 14px 16px !important;
    border-top: 1px solid var(--rvc-border) !important;
    background: var(--rvc-surface) !important;
    overflow: visible !important;
}

/* Compact Crisp Audio & File Dropzones */
.gradio-container .audio-container,
.gradio-container .upload-container,
.gradio-container .file-preview {
    background: var(--rvc-surface-subtle) !important;
    border: 1px dashed var(--rvc-border-strong) !important;
    border-radius: var(--rvc-radius) !important;
    color: var(--rvc-text) !important;
    transition: border-color var(--rvc-transition), background var(--rvc-transition);
}
.gradio-container .audio-container:hover,
.gradio-container .upload-container:hover {
    border-color: var(--rvc-accent) !important;
}
.gradio-container label[data-testid="block-label"] {
    background: var(--rvc-surface) !important;
    color: var(--rvc-text-secondary) !important;
    border-right: 1px solid var(--rvc-border) !important;
    border-bottom: 1px solid var(--rvc-border) !important;
    border-radius: 0 0 var(--rvc-radius-sm) 0 !important;
    padding: 4px 10px !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    line-height: 1.3 !important;
    gap: 5px !important;
}
.gradio-container .icon-button-wrapper {
    background: var(--rvc-surface) !important;
    border: 1px solid var(--rvc-border-strong) !important;
    border-radius: var(--rvc-radius-sm) !important;
    color: var(--rvc-text-secondary) !important;
}
.gradio-container .rvc-audio-drop .upload-container,
.gradio-container .rvc-audio-drop .center,
.gradio-container .rvc-audio-drop .wrap.center {
    min-height: 148px !important;
}
.gradio-container .waveform-container,
.gradio-container .audio-container {
    --waveform-color: var(--rvc-accent) !important;
    --progress-color: var(--rvc-accent) !important;
}

/* Multitrack Player Frame */
.rvc-multitrack-frame {
    border-radius: var(--rvc-radius);
    background: transparent;
}

/* Clean Precision Data Table */
.gradio-container div[data-testid="dataframe"],
.gradio-container .table-wrap {
    background: var(--rvc-surface) !important;
    color: var(--rvc-text) !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: var(--rvc-radius) !important;
    --table-odd-background-fill: var(--rvc-table-odd) !important;
    --table-even-background-fill: var(--rvc-table-even) !important;
    --table-editing: var(--rvc-table-hover) !important;
}
.gradio-container .model-catalog ::-webkit-scrollbar:horizontal {
    height: 0 !important;
}
.gradio-container .model-catalog table {
    width: 100% !important;
    table-layout: fixed !important;
}
.gradio-container .table-wrap thead th,
.gradio-container div[data-testid="dataframe"] thead th {
    background: var(--rvc-table-head) !important;
    color: var(--rvc-text-secondary) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    border-color: var(--rvc-border) !important;
}
.gradio-container .table-wrap thead th .cell-wrap,
.gradio-container .table-wrap thead th span {
    background: transparent !important;
    color: var(--rvc-text-secondary) !important;
}
.gradio-container .table-wrap tbody tr:nth-child(odd),
.gradio-container .table-wrap tbody tr:nth-child(odd) td,
.gradio-container .table-wrap tbody tr:nth-child(odd) .cell-wrap {
    background-color: var(--rvc-table-odd) !important;
}
.gradio-container .table-wrap tbody tr:nth-child(even),
.gradio-container .table-wrap tbody tr:nth-child(even) td,
.gradio-container .table-wrap tbody tr:nth-child(even) .cell-wrap {
    background-color: var(--rvc-table-even) !important;
}
.gradio-container .table-wrap tbody tr:hover td,
.gradio-container div[data-testid="dataframe"] tbody tr:hover td {
    background-color: var(--rvc-table-hover) !important;
}
.gradio-container .table-wrap tbody td,
.gradio-container div[data-testid="dataframe"] tbody td {
    color: var(--rvc-text) !important;
    border-color: var(--rvc-border) !important;
}
.gradio-container .table-wrap tbody td > *,
.gradio-container .table-wrap tbody td .cell-wrap span {
    background-color: transparent !important;
}
.model-catalog .cell-wrap :is(span, button),
.model-catalog td {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: normal !important;
    line-height: 1.5 !important;
}
.model-catalog td .cell-wrap {
    padding: 8px 10px !important;
}
.model-catalog td [role="button"] {
    color: var(--rvc-text-secondary) !important;
    font-size: 12px !important;
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace !important;
}
.model-catalog td:first-child [role="button"] {
    color: var(--rvc-text) !important;
    font-family: inherit !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* Opaque StatusTracker Progress Overlays, Toasts & Scrollbars */
.gradio-container .wrap.default:not(.hide),
.gradio-container .wrap.full:not(.hide),
.gradio-container .wrap:not(.hide):has(> .progress-level),
.gradio-container .wrap:not(.hide):has(> .meta-text),
.gradio-container .wrap:not(.hide):has(> .meta-text-center),
.gradio-container .wrap:not(.hide):has(> .eta-bar),
.gradio-container .wrap:not(.hide):has(> .loading) {
    background: var(--rvc-surface) !important;
    border: 1px solid var(--rvc-border-strong) !important;
    border-radius: var(--rvc-radius) !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}
.gradio-container .status-box > .wrap.default:not(.hide),
.gradio-container .status-box > .wrap:not(.hide):has(> .progress-level),
.gradio-container .status-box > .wrap:not(.hide):has(> .meta-text) {
    top: 24px !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    padding: 12px 16px 22px !important;
    background: var(--rvc-surface-subtle) !important;
    border: 1px solid var(--rvc-accent-border) !important;
    border-radius: var(--rvc-radius) !important;
}
.gradio-container .wrap.translucent {
    background: color-mix(in srgb, var(--rvc-surface) 92%, transparent) !important;
}
.gradio-container .wrap.generating {
    background: transparent !important;
    border: 1.5px solid var(--rvc-accent) !important;
    border-radius: var(--rvc-radius) !important;
}
.gradio-container .eta-bar {
    display: none !important;
}
.gradio-container .progress-level {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 6px !important;
    box-sizing: border-box !important;
}
.gradio-container .progress-level-inner {
    margin: 0 !important;
    max-width: 100% !important;
    color: var(--rvc-text) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    font-family: inherit !important;
    line-height: 1.4 !important;
    text-align: center !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}
.gradio-container .progress-bar-wrap {
    width: 100% !important;
    max-width: 420px !important;
    height: 6px !important;
    background: var(--rvc-border) !important;
    border: 0 !important;
    border-radius: 999px !important;
    overflow: hidden !important;
}
.gradio-container .progress-bar {
    background: var(--rvc-accent) !important;
    height: 100% !important;
    border-radius: 999px !important;
    transition: width 160ms ease-out !important;
}
.gradio-container .meta-text,
.gradio-container .meta-text-center {
    color: var(--rvc-text-muted) !important;
    font-size: 11px !important;
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace !important;
    line-height: 1.2 !important;
}
.gradio-container .status-box > .wrap .meta-text {
    bottom: 5px !important;
    right: 10px !important;
    padding: 0 !important;
}
.gradio-container .minimal .progress-text {
    background: var(--rvc-surface) !important;
    color: var(--rvc-text) !important;
    padding: 2px 6px !important;
    border-radius: var(--rvc-radius-sm) !important;
}
.gradio-container .toast-body {
    border-radius: var(--rvc-radius) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16) !important;
}
.gradio-container hr {
    border-color: var(--rvc-border) !important;
}
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--rvc-border-strong);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--rvc-text-muted);
}

/* Responsive Layout (Two-column stays active down to 860px for HiDPI laptops) */
@media (max-width: 860px) {
    .rvc-workbench,
    .rvc-two-col-grid {
        grid-template-columns: minmax(0, 1fr) !important;
    }
    .rvc-sticky-col {
        position: static !important;
        top: auto !important;
    }
}

@media (max-width: 680px) {
    .gradio-container main {
        padding: 0 14px 28px !important;
    }
    .gradio-container .top-header {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 10px !important;
        padding: 12px 0 !important;
    }
    .gradio-container .top-header > .top-actions {
        align-items: stretch !important;
        width: 100% !important;
    }
    .top-brand-title {
        font-size: 18px !important;
    }
    .gradio-container .top-controls-row {
        display: grid !important;
        grid-template-columns: auto minmax(0, 1fr);
        width: 100% !important;
        gap: 8px !important;
    }
    .gradio-container button.rvc-theme-btn {
        height: 36px !important;
        min-height: 36px !important;
        padding: 0 11px !important;
    }
    .gradio-container .language-switch-row {
        padding-left: 0 !important;
        border-left: 0 !important;
        width: 100% !important;
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) auto !important;
        gap: 8px !important;
    }
    .gradio-container .language-switch-row .wrap:has(> .wrap-inner) {
        width: 100% !important;
        min-width: 0 !important;
        height: 36px !important;
        min-height: 36px !important;
    }
    .gradio-container .language-switch-row button,
    .gradio-container .language-switch-row button.secondary {
        height: 36px !important;
        min-height: 36px !important;
    }
    .gradio-container .language-status-note {
        text-align: left;
    }
    .rvc-panel {
        padding: 14px !important;
    }
    .gradio-container .rvc-main-tabs > .tab-wrapper {
        height: 44px;
        padding-bottom: 0;
    }
    .gradio-container .rvc-main-tabs > .tab-wrapper > .tab-container {
        width: 100%;
        height: 44px;
        min-height: 44px;
        padding-right: 1px;
        box-sizing: border-box;
        align-items: stretch;
        gap: 0;
    }
    .gradio-container .rvc-main-tabs > .tab-wrapper > .tab-container > button {
        flex: 1 1 0;
        min-width: 0;
        padding: 8px 4px !important;
        font-size: 13.5px !important;
        white-space: normal;
        line-height: 1.25;
        justify-content: center;
    }
    .gradio-container input:not([type="range"]),
    .gradio-container textarea:not([disabled]) {
        font-size: 16px !important;
    }
    .gradio-container button.lg {
        min-height: 44px;
    }
    .gradio-container .row:not(.language-switch-row):not(.top-controls-row):not(.rvc-head-row) > button {
        align-self: stretch !important;
    }
    .model-catalog td .cell-wrap {
        padding: 6px 4px !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    .gradio-container button,
    .gradio-container button.rvc-theme-btn {
        transition: none !important;
    }
}
"""


def _theme_toggle_html() -> str:
    return (
        '<button id="rvc-theme-toggle" type="button" class="rvc-theme-btn" aria-pressed="true">'
        '<span class="rvc-theme-icon" aria-hidden="true">'
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg></span>'
        '<span class="rvc-theme-text">' + t("theme_dark", "settings") + '</span>'
        '</button>'
    )


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    with gr.Blocks(
        title=t("app_title"),
        theme=gr.themes.Base(
            primary_hue="amber",
            secondary_hue="slate",
            neutral_hue="slate",
        ).set(
            # Geometry: crisp minimal radii and zero default nested block frames
            block_border_width="0px",
            block_shadow="none",
            block_radius="4px",
            container_radius="4px",
            input_radius="4px",
            button_large_radius="4px",
            button_small_radius="3px",
            # Light mode tokens
            body_background_fill="#f5f6f8",
            block_background_fill="#ffffff",
            block_border_color="#e2e5ea",
            block_label_background_fill="#ffffff",
            block_label_text_color="#4a505c",
            block_title_text_color="#111318",
            input_background_fill="#f2f3f6",
            input_border_color="#cdd2dc",
            button_primary_background_fill="#d4600a",
            button_primary_background_fill_hover="#b84f06",
            button_primary_text_color="#ffffff",
            button_secondary_background_fill="#ffffff",
            button_secondary_background_fill_hover="#eaecf0",
            button_secondary_text_color="#111318",
            body_text_color="#111318",
            body_text_color_subdued="#707685",
            link_text_color="#d4600a",
            link_text_color_hover="#b84f06",
            slider_color="#d4600a",
            checkbox_background_color="#f2f3f6",
            checkbox_border_color="#cdd2dc",
            checkbox_label_text_color="#111318",
            # Dark mode tokens
            body_background_fill_dark="#0b0c0f",
            block_background_fill_dark="#12141a",
            block_border_color_dark="#242834",
            block_label_background_fill_dark="#12141a",
            block_label_text_color_dark="#aeb5c4",
            block_title_text_color_dark="#f3f4f6",
            input_background_fill_dark="#181b23",
            input_border_color_dark="#343a49",
            button_primary_background_fill_dark="#e59225",
            button_primary_background_fill_hover_dark="#f0a43b",
            button_primary_text_color_dark="#120d06",
            button_secondary_background_fill_dark="#13151a",
            button_secondary_background_fill_hover_dark="#222631",
            button_secondary_text_color_dark="#f3f4f6",
            body_text_color_dark="#f3f4f6",
            body_text_color_subdued_dark="#777f92",
            link_text_color_dark="#e59225",
            link_text_color_hover_dark="#f0a43b",
            slider_color_dark="#e59225",
            checkbox_background_color_dark="#1a1d25",
            checkbox_border_color_dark="#353b4a",
            checkbox_label_text_color_dark="#f3f4f6",
        ),
        head=f"{THEME_INIT_SCRIPT}<style>{CUSTOM_CSS}</style>",
        js=PLAYER_BRIDGE_JS.replace('__RVC_LANGUAGE__', get_configured_language()),
    ) as app:

        with gr.Row(elem_classes=["top-header"]):
            with gr.Column(scale=5, elem_classes=["top-brand"]):
                gr.Markdown(ui_text('brand'))

            with gr.Column(scale=4, elem_classes=["top-actions"]):
                with gr.Row(elem_classes=["top-controls-row"]):
                    gr.HTML(_theme_toggle_html(), elem_classes=["theme-toggle-slot"])
                    with gr.Row(elem_classes=["language-switch-row"]):
                        language_dropdown = gr.Dropdown(
                            label=ui_text("language", "settings"),
                            choices=list(LANGUAGE_LABEL_TO_CODE.keys()),
                            value=get_current_language_label(),
                            interactive=True,
                            container=False,
                        )
                        save_language_btn = gr.Button(
                            ui_text("save_language", "settings"),
                            variant="secondary",
                        )

                language_status = gr.Markdown(
                    "",
                    elem_classes=["language-status-note"],
                )

        with gr.Tabs(elem_classes=["rvc-main-tabs"]):
            # ===== 歌曲翻唱标签页（默认首屏） =====
            with gr.Tab(ui_text("cover", "tabs")):
                reuse_state = gr.State({})
                batch_rows = gr.State([])
                gr.Markdown(ui_text("cover_usage", "ui"), elem_classes=["rvc-guide-line"])

                with gr.Row(elem_classes=["rvc-workbench"]):
                    # 左侧主工作区：音频输入（单首 + 批量统一入口）与声线选择
                    with gr.Column(scale=7, elem_classes=["rvc-panel"]):
                        with gr.Row(elem_classes=["rvc-head-row"]):
                            gr.Markdown(f"#### 01 · {ui_text('upload_song', 'cover')}", elem_classes=["rvc-head-title"])
                        cover_input_mode = gr.Radio(
                            label=ui_text("input_mode", "cover"),
                            choices=[(t("input_mode_song", "cover"), "song"), (t("input_mode_vocals", "cover"), "vocals")],
                            value=config.get("cover", {}).get("input_mode", "song"),
                            info=ui_text("input_mode_info", "cover"),
                        )
                        with gr.Tabs(elem_classes=["rvc-subtabs"]):
                            with gr.Tab(ui_text("single_mode", "cover")):
                                cover_input_audio = gr.Audio(
                                    label=ui_text("input_audio", "cover"),
                                    type="filepath",
                                    elem_classes=["rvc-audio-drop"],
                                )
                            with gr.Tab(ui_text("batch", "cover")):
                                gr.Markdown(ui_text("batch_help", "cover"), elem_classes=["rvc-muted-copy"])
                                batch_files = gr.File(
                                    label=ui_text("batch_files", "cover"),
                                    file_count="multiple",
                                    file_types=["audio"],
                                    type="filepath",
                                )
                                with gr.Row():
                                    batch_btn = gr.Button(ui_text("batch_start", "cover"), variant="primary")
                                    batch_download = gr.DownloadButton(
                                        label=ui_text("batch_download", "cover"),
                                        visible=False,
                                        variant="secondary",
                                    )
                                batch_picker = gr.Dropdown(
                                    label=ui_text("batch_preview", "cover"),
                                    choices=[],
                                    interactive=True,
                                    visible=False,
                                )
                                batch_status = gr.Textbox(
                                    label=ui_text("progress", "cover"),
                                    placeholder=ui_text("batch_status_placeholder", "cover"),
                                    interactive=False,
                                    lines=3,
                                    max_lines=8,
                                    elem_classes=["status-box"],
                                )

                        with gr.Row(elem_classes=["rvc-head-row", "rvc-head-row-spaced"]):
                            gr.Markdown(f"#### 02 · {ui_text('select_character', 'cover')}", elem_classes=["rvc-head-title"])
                            refresh_char_btn = gr.Button(
                                ui_text('refresh', 'conversion'),
                                size="sm",
                                variant="secondary",
                                elem_classes=["rvc-head-btn"],
                            )

                        with gr.Row():
                            downloaded_series = gr.Dropdown(
                                label=ui_text("series_filter", "ui"),
                                choices=get_downloaded_character_series(),
                                value=_all_series_label(),
                                interactive=True,
                                scale=1,
                            )
                            downloaded_keyword = gr.Textbox(
                                label=ui_text("keyword_search", "ui"),
                                placeholder=ui_text("keyword_placeholder", "ui"),
                                interactive=True,
                                scale=1,
                            )

                        initial_character_choices = get_downloaded_character_choices(_all_series_label(), "")
                        initial_character = initial_character_choices[0][1] if initial_character_choices else None
                        character_dropdown = gr.Dropdown(
                            label=ui_text("character", "cover"),
                            choices=initial_character_choices,
                            value=initial_character,
                            allow_custom_value=False,
                            interactive=True,
                            info=ui_text("character_choice_info", "ui"),
                        )

                        character_details = gr.Markdown(
                            value=get_downloaded_character_details(initial_character),
                            elem_classes=["rvc-detail-box"],
                        )

                        with gr.Accordion(ui_text("download_character", "cover"), open=False):
                            series_choices = [_all_series_label()] + get_available_character_series()
                            with gr.Row():
                                download_series = gr.Dropdown(
                                    label=ui_text("series_filter", "ui"),
                                    choices=series_choices,
                                    value=_all_series_label(),
                                    interactive=True,
                                )
                                download_keyword = gr.Textbox(
                                    label=ui_text("keyword_search", "ui"),
                                    placeholder=ui_text("keyword_placeholder", "ui"),
                                    interactive=True,
                                )

                            available_choices = get_available_character_choices(_all_series_label(), "")
                            initial_available = available_choices[0][1] if available_choices else None
                            download_char_dropdown = gr.Dropdown(
                                label=ui_text("select_to_download", "cover"),
                                choices=available_choices,
                                value=initial_available,
                                interactive=True,
                                info=ui_text("download_character_info", "ui")
                            )

                            download_char_details = gr.Markdown(
                                value=get_available_character_details(initial_available),
                                elem_classes=["rvc-detail-box"],
                            )

                            with gr.Row():
                                download_char_btn = gr.Button(
                                    ui_text('download_selected_character', 'ui'),
                                    variant="primary",
                                )
                                download_all_series_btn = gr.Button(
                                    ui_text('download_series_all', 'ui'),
                                    variant="secondary",
                                )
                                download_all_btn = gr.Button(
                                    ui_text('download_all_characters', 'ui'),
                                    variant="secondary",
                                )

                            download_char_status = gr.Textbox(
                                label=ui_text("download_status", "ui"),
                                interactive=False,
                                elem_classes=["status-box"],
                            )

                        with gr.Accordion(ui_text("upload_custom_character", "cover"), open=False):
                            with gr.Row():
                                custom_model_name = gr.Textbox(
                                    label=ui_text("custom_model_name", "ui"),
                                    placeholder=ui_text("custom_model_name_placeholder", "ui"),
                                    interactive=True,
                                )
                                custom_model_category = gr.Textbox(
                                    label=ui_text("custom_model_category", "ui"),
                                    value=t("custom_model_default_category", "ui"),
                                    interactive=True,
                                )
                            custom_model_source = gr.Textbox(
                                label=ui_text("custom_model_source", "ui"),
                                value=t("custom_model_default_source", "ui"),
                                interactive=True,
                            )
                            with gr.Row():
                                custom_model_file = gr.File(
                                    label=ui_text("custom_model_file", "ui"),
                                    file_types=[".pth", ".zip"],
                                    type="filepath",
                                )
                                custom_index_file = gr.File(
                                    label=ui_text("custom_index_file", "ui"),
                                    file_types=[".index"],
                                    type="filepath",
                                )
                            custom_model_upload_btn = gr.Button(
                                ui_text('import_custom_model', 'ui'),
                                variant="primary",
                            )
                            custom_model_status = gr.Textbox(
                                label=ui_text("custom_model_status", "ui"),
                                interactive=False,
                                elem_classes=["status-box"],
                            )

                    # 右侧控制台（5 栏）：处理流程、参数调整、启动与进度
                    with gr.Column(scale=5, elem_classes=["rvc-panel", "rvc-sticky-col"]):
                        with gr.Row(elem_classes=["rvc-head-row"]):
                            gr.Markdown(f"#### 03 · {ui_text('automatic_cover_settings', 'cover')}", elem_classes=["rvc-head-title"])
                        gr.Markdown(ui_text("automatic_cover_settings_info", "cover"), elem_classes=["rvc-muted-copy"])
                        cover_cfg = config.get("cover", {})
                        auto_settings = resolve_automatic_cover_settings(config)
                        source_label_to_value, _ = get_source_constraint_option_maps()
                        pipeline_label_to_value, _ = get_vc_pipeline_mode_option_maps()

                        cover_vc_route_status = gr.Textbox(
                            label=ui_text("vc_preprocess_status", "cover"),
                            value=get_cover_vc_route_status(
                                cover_cfg.get("vc_pipeline_mode", "current"),
                                cover_cfg.get("use_official", True),
                                cover_cfg.get("karaoke_separation", True),
                                cover_cfg.get("input_mode", "song"),
                            ),
                            info=ui_text("vc_preprocess_status_info", "cover"),
                            interactive=False,
                            lines=4,
                            elem_classes=["status-box"]
                        )

                        with gr.Accordion(ui_text("manual_cover_settings", "cover"), open=False):
                            gr.Markdown(f"##### {ui_text('conversion_settings', 'cover')}", elem_classes=["rvc-subhead"])

                            with gr.Row():
                                cover_pitch_shift = gr.Slider(
                                    label=ui_text("pitch_shift", "cover"),
                                    minimum=-12,
                                    maximum=12,
                                    value=int(auto_settings["pitch_shift"]),
                                    step=1,
                                    info=ui_text("positive_pitch_info", "ui")
                                )
                                cover_index_rate = gr.Slider(
                                    label=ui_text("index_rate", "cover"),
                                    minimum=0,
                                    maximum=100,
                                    value=int(round(float(auto_settings["index_ratio"]) * 100)),
                                    step=5,
                                    info=ui_text("index_rate_info", "cover"),
                                )

                            with gr.Row():
                                cover_speaker_id = gr.Slider(
                                    label=ui_text("speaker_id", "cover"),
                                    minimum=0,
                                    maximum=max(1, int(auto_settings["speaker_id"])),
                                    value=int(auto_settings["speaker_id"]),
                                    interactive=False,
                                    step=1,
                                    info=ui_text("speaker_id_info", "cover"),
                                )
                                cover_vc_pipeline_mode = gr.Dropdown(
                                    label=ui_text("vc_pipeline_mode", "cover"),
                                    choices=list(pipeline_label_to_value.items()),
                                    value=auto_settings["vc_pipeline_mode"],
                                    info=ui_text("vc_pipeline_mode_info", "cover"),
                                )

                            cover_mix_heading = gr.Markdown(f"##### {ui_text('mix_settings', 'cover')}", elem_classes=["rvc-subhead"])
                            with gr.Row():
                                cover_karaoke = gr.Checkbox(
                                    label=ui_text("karaoke_separation", "cover"),
                                    value=bool(auto_settings["karaoke_separation"]),
                                    info=ui_text("karaoke_separation_info", "cover")
                                )
                                cover_karaoke_merge_backing = gr.Checkbox(
                                    label=ui_text("karaoke_merge_backing", "cover"),
                                    value=bool(auto_settings["karaoke_merge_backing_into_accompaniment"]),
                                    info=ui_text("karaoke_merge_backing_info", "cover")
                                )

                            cover_dereverb_info = gr.Markdown(
                                ui_text("fixed_dereverb_info", "cover"),
                                elem_classes=["rvc-inline-note"],
                            )

                            mix_presets, default_mix_preset = get_cover_mix_presets()
                            default_mix = mix_presets[default_mix_preset]

                            with gr.Row():
                                cover_source_constraint_mode = gr.Dropdown(
                                    label=ui_text("source_constraint_mode", "cover"),
                                    choices=list(source_label_to_value.items()),
                                    value=auto_settings["source_constraint_mode"],
                                    info=ui_text("source_constraint_mode_info", "cover"),
                                )
                                cover_mix_preset = gr.Dropdown(
                                    label=ui_text("mix_preset", "cover"),
                                    choices=get_cover_mix_preset_options(),
                                    value="mix_preset_universal",
                                    info=ui_text("mix_preset_info", "cover"),
                                    interactive=True
                                )

                            with gr.Row():
                                cover_vocals_volume = gr.Slider(
                                    label=ui_text("vocals_volume", "cover"),
                                    minimum=0,
                                    maximum=200,
                                    value=default_mix["vocals_volume"],
                                    step=5,
                                    info=ui_text("normal_volume_info", "ui")
                                )
                                cover_accompaniment_volume = gr.Slider(
                                    label=ui_text("accompaniment_volume", "cover"),
                                    minimum=0,
                                    maximum=200,
                                    value=default_mix["accompaniment_volume"],
                                    step=5,
                                    info=ui_text("normal_volume_info", "ui")
                                )

                            with gr.Row():
                                cover_reverb = gr.Slider(
                                    label=ui_text("vocals_reverb", "cover"),
                                    minimum=0,
                                    maximum=100,
                                    value=default_mix["reverb"],
                                    step=5,
                                    info=ui_text("reverb_info", "ui")
                                )
                                cover_rms_mix_rate = gr.Slider(
                                    label=ui_text("rms_mix_rate", "cover"),
                                    minimum=0,
                                    maximum=100,
                                    value=int(round(float(auto_settings["rms_mix_rate"]) * 100)),
                                    step=5,
                                    info=ui_text("rms_mix_rate_info", "cover"),
                                )

                            cover_backing_mix = gr.Slider(
                                label=ui_text("backing_mix", "cover"),
                                minimum=0,
                                maximum=100,
                                value=int(round(float(auto_settings["backing_mix"]) * 100)),
                                step=5,
                                info=ui_text("backing_mix_info", "cover"),
                            )

                        with gr.Column(elem_classes=["rvc-action-divider"]):
                            cover_btn = gr.Button(
                                ui_text('start_cover', 'cover'),
                                variant="primary",
                                size="lg",
                                elem_classes=["rvc-primary-cta"],
                            )
                            gr.Markdown(ui_text("reuse_help", "cover"), elem_classes=["rvc-muted-copy"])
                            cover_status = gr.Textbox(
                                label=ui_text("progress", "cover"),
                                placeholder=ui_text("progress_placeholder", "cover"),
                                interactive=False,
                                lines=5,
                                max_lines=10,
                                elem_classes=["status-box", "cover-progress-status"]
                            )

                # 中央全宽核心舞台：多轨试听与高清分轨输出
                with gr.Column(elem_classes=["rvc-panel", "rvc-panel-stack"]):
                    with gr.Row(elem_classes=["rvc-head-row"]):
                        gr.Markdown(f"#### 04 · {ui_text('results', 'cover')}", elem_classes=["rvc-head-title"])
                    gr.HTML(player_html(get_configured_language()), elem_id="rvc-player")

                # Keep the existing API output contract and Gradio file authorization.
                with gr.Group(visible=False):
                    def _cover_download_label(output_label: str) -> str:
                        return f"{ui_text('download', 'cover')} · {output_label}"

                    with gr.Row():
                        with gr.Column():
                            cover_output = gr.Audio(
                                label=ui_text("final_cover", "cover"),
                                type="filepath",
                                interactive=False
                            )
                            cover_download_btn = gr.DownloadButton(
                                label=_cover_download_label(t("final_cover", "cover")),
                                value=None,
                                visible=False,
                                size="sm",
                                variant="secondary",
                                elem_classes=["cover-download-button"],
                            )

                    with gr.Row():
                        with gr.Column():
                            cover_converted_vocals_output = gr.Audio(
                                label=ui_text("converted_vocals", "cover"),
                                type="filepath",
                                interactive=False
                            )
                            cover_converted_vocals_download_btn = gr.DownloadButton(
                                label=_cover_download_label(t("converted_vocals", "cover")),
                                value=None,
                                visible=False,
                                size="sm",
                                variant="secondary",
                                elem_classes=["cover-download-button"],
                            )
                        with gr.Column():
                            cover_original_vocals_output = gr.Audio(
                                label=ui_text("original_vocals", "cover"),
                                type="filepath",
                                interactive=False
                            )
                            cover_original_vocals_download_btn = gr.DownloadButton(
                                label=_cover_download_label(t("original_vocals", "cover")),
                                value=None,
                                visible=False,
                                size="sm",
                                variant="secondary",
                                elem_classes=["cover-download-button"],
                            )

                    with gr.Row():
                        with gr.Column():
                            cover_lead_vocals_output = gr.Audio(
                                label=ui_text("lead_vocals", "cover"),
                                type="filepath",
                                interactive=False
                            )
                            cover_lead_vocals_download_btn = gr.DownloadButton(
                                label=_cover_download_label(t("lead_vocals", "cover")),
                                value=None,
                                visible=False,
                                size="sm",
                                variant="secondary",
                                elem_classes=["cover-download-button"],
                            )
                        with gr.Column():
                            cover_backing_vocals_output = gr.Audio(
                                label=ui_text("backing_vocals", "cover"),
                                type="filepath",
                                interactive=False
                            )
                            cover_backing_vocals_download_btn = gr.DownloadButton(
                                label=_cover_download_label(t("backing_vocals", "cover")),
                                value=None,
                                visible=False,
                                size="sm",
                                variant="secondary",
                                elem_classes=["cover-download-button"],
                            )

                    with gr.Row():
                        with gr.Column():
                            cover_accompaniment_output = gr.Audio(
                                label=ui_text("accompaniment", "cover"),
                                type="filepath",
                                interactive=False
                            )
                            cover_accompaniment_download_btn = gr.DownloadButton(
                                label=_cover_download_label(t("accompaniment", "cover")),
                                value=None,
                                visible=False,
                                size="sm",
                                variant="secondary",
                                elem_classes=["cover-download-button"],
                            )
                        with gr.Column():
                            cover_accompaniment_without_harmony_output = gr.Audio(
                                label=ui_text("accompaniment_without_harmony", "cover"),
                                type="filepath",
                                interactive=False
                            )
                            cover_accompaniment_without_harmony_download_btn = gr.DownloadButton(
                                label=_cover_download_label(
                                    t("accompaniment_without_harmony", "cover")
                                ),
                                value=None,
                                visible=False,
                                size="sm",
                                variant="secondary",
                                elem_classes=["cover-download-button"],
                            )

            # ===== 模型管理标签页 =====
            with gr.Tab(ui_text("models", "tabs")):
                with gr.Row(elem_classes=["rvc-two-col-grid"]):
                    with gr.Column(elem_classes=["rvc-panel"]):
                        with gr.Row(elem_classes=["rvc-head-row"]):
                            gr.Markdown(f"#### {ui_text('base_models', 'models')}", elem_classes=["rvc-head-title"])
                        gr.Markdown(ui_text("base_models_desc", "models"), elem_classes=["rvc-muted-copy"])

                        with gr.Row():
                            check_btn = gr.Button(
                                ui_text('check_status', 'models'),
                                variant="secondary"
                            )
                            download_btn = gr.Button(
                                ui_text('download_required', 'models'),
                                variant="primary"
                            )

                        model_status = gr.Textbox(
                            label=ui_text("model_status", "models"),
                            placeholder=ui_text("model_status_placeholder", "models"),
                            interactive=False,
                            lines=6,
                            elem_classes=["status-box"]
                        )

                    with gr.Column(elem_classes=["rvc-panel"]):
                        with gr.Row(elem_classes=["rvc-head-row"]):
                            gr.Markdown(f"#### {ui_text('mature_deecho_models', 'models')}", elem_classes=["rvc-head-title"])
                        gr.Markdown(ui_text("mature_deecho_models_desc", "models"), elem_classes=["rvc-muted-copy"])

                        with gr.Row():
                            mature_deecho_check_btn = gr.Button(
                                ui_text('mature_deecho_check', 'models'),
                                variant="secondary"
                            )
                            mature_deecho_download_btn = gr.Button(
                                ui_text('download_mature_deecho', 'models'),
                                variant="primary"
                            )

                        mature_deecho_status = gr.Textbox(
                            label=ui_text("mature_deecho_status", "models"),
                            interactive=False,
                            lines=6,
                            value=check_mature_deecho_status(),
                            elem_classes=["status-box"]
                        )

                with gr.Column(elem_classes=["rvc-panel", "rvc-panel-stack"]):
                    with gr.Row(elem_classes=["rvc-head-row"]):
                        gr.Markdown(f"#### {ui_text('voice_models', 'models')}", elem_classes=["rvc-head-title"])
                        refresh_table_btn = gr.Button(
                            ui_text('refresh_models', 'ui'),
                            size="sm",
                            variant="secondary",
                            elem_classes=["rvc-head-btn"],
                        )
                    gr.Markdown(ui_text("voice_models_desc", "models"), elem_classes=["rvc-muted-copy"])

                    def get_model_table():
                        from infer.pipeline import list_voice_models
                        weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")
                        models = list_voice_models(str(weights_dir))
                        if not models:
                            return []
                        return [[m["name"], m["model_path"], m.get("index_error") or m.get("index_path") or ""] for m in models]

                    model_table = gr.Dataframe(
                        headers=[
                            t("model_name", "ui"),
                            t("model_path", "ui"),
                            t("index_path", "ui"),
                        ],
                        value=get_model_table(),
                        interactive=False,
                        wrap=True,
                        column_widths=["27%", "36%", "36%"],
                        elem_classes=["model-catalog"],
                    )

            # ===== 设置标签页 =====
            with gr.Tab(ui_text("settings", "tabs")):
                with gr.Row(elem_classes=["rvc-two-col-grid"]):
                    with gr.Column(elem_classes=["rvc-panel"]):
                        with gr.Row(elem_classes=["rvc-head-row"]):
                            gr.Markdown(f"#### {ui_text('device_info', 'settings')}", elem_classes=["rvc-head-title"])
                            refresh_device_btn = gr.Button(
                                ui_text('refresh_device', 'settings'),
                                size="sm",
                                variant="secondary",
                                elem_classes=["rvc-head-btn"],
                            )

                        device_info = gr.Textbox(
                            label=ui_text("current_device", "settings"),
                            value=get_device_info(),
                            interactive=False,
                            lines=5,
                            elem_classes=["status-box"]
                        )

                    with gr.Column(elem_classes=["rvc-panel"]):
                        with gr.Row(elem_classes=["rvc-head-row"]):
                            gr.Markdown(f'#### {ui_text("runtime_settings", "settings")}', elem_classes=["rvc-head-title"])

                        def _build_device_choices():
                            from lib.device import _has_xpu, _has_directml, _has_mps, _is_rocm
                            import torch
                            choices = [(t('device_auto', 'settings'), 'auto')]
                            if torch.cuda.is_available():
                                label = "ROCm (AMD GPU)" if _is_rocm() else "CUDA (NVIDIA GPU)"
                                choices.append((label, "cuda"))
                            if _has_xpu():
                                choices.append(("XPU (Intel GPU)", "xpu"))
                            if _has_directml():
                                choices.append(("DirectML (AMD/Intel GPU)", "directml"))
                            if _has_mps():
                                choices.append(("MPS (Apple GPU)", "mps"))
                            choices.append((t("cpu_slow", "settings"), "cpu"))
                            selected = config.get('device', 'auto')
                            if selected not in {value for _, value in choices}:
                                choices.append((selected, selected))
                            return choices

                        managed_device = bool(os.environ.get("AI_RVC_DEVICE", "").strip())
                        device_radio = gr.Radio(
                            label=ui_text("compute_device", "settings"),
                            choices=_build_device_choices(),
                            value=config.get("device", "cuda"),
                            interactive=not managed_device,
                        )

                        save_settings_btn = gr.Button(
                            ui_text('save_settings', 'settings'),
                            variant="primary",
                            interactive=not managed_device,
                            elem_classes=["rvc-compact-btn"],
                        )

                        gr.Markdown(
                            ui_text("device_managed_by_startup", "settings"),
                            visible=managed_device,
                            elem_classes=["rvc-inline-note"],
                        )

                        settings_status = gr.Textbox(
                            label=ui_text("status", "settings"),
                            interactive=False,
                            elem_classes=["status-box"],
                        )

                with gr.Row(elem_classes=["rvc-two-col-grid", "rvc-panel-stack"]):
                    with gr.Column(elem_classes=["rvc-panel"]):
                        with gr.Row(elem_classes=["rvc-head-row"]):
                            gr.Markdown(f"#### {ui_text('about', 'settings')}", elem_classes=["rvc-head-title"])
                        gr.Markdown(ui_text("about_body", "settings"))
                        with gr.Accordion(ui_text("processing_details", "cover"), open=False):
                            gr.Markdown(ui_text("diagnostic_build", "settings"), elem_classes=["rvc-muted-copy"])
                            gr.Textbox(
                                label=ui_text("diagnostic_build", "settings"),
                                value=get_runtime_build_short_label(),
                                show_label=False,
                                interactive=False,
                                elem_classes=["status-box"],
                            )
                            gr.Markdown(ui_text("workflow_models", "models"), elem_classes=["rvc-inline-note"])

                    with gr.Column(elem_classes=["rvc-panel"]):
                        gr.Markdown(ui_text("model_sources", "settings"))

        # ===== 事件绑定 =====
        check_btn.click(
            fn=check_models_status,
            outputs=[model_status]
        )
        download_btn.click(
            fn=download_base_models,
            outputs=[model_status]
        )
        refresh_table_btn.click(
            fn=get_model_table,
            outputs=[model_table]
        )
        refresh_device_btn.click(
            fn=get_device_info,
            outputs=[device_info]
        )

        def save_settings(device):
            if os.environ.get("AI_RVC_DEVICE", "").strip():
                raise gr.Error(t("device_managed_by_startup", "settings"))
            from lib.device import get_device
            get_device(device)  # Reject unavailable devices before writing.
            update_config(CONFIG_PATH, {'device': device})
            return t("settings_saved_restart", "settings")

        save_settings_btn.click(
            fn=save_settings,
            inputs=[device_radio],
            outputs=[settings_status]
        )

        refresh_char_btn.click(
            fn=refresh_downloaded_controls,
            inputs=[downloaded_series, downloaded_keyword],
            outputs=[downloaded_series, character_dropdown]
        )
        refresh_char_btn.click(
            fn=lambda: get_downloaded_character_details(None),
            outputs=[character_details]
        )

        downloaded_series.input(
            fn=update_downloaded_choices,
            inputs=[downloaded_series, downloaded_keyword],
            outputs=[character_dropdown]
        )
        downloaded_series.input(
            fn=lambda: get_downloaded_character_details(None),
            outputs=[character_details]
        )

        downloaded_keyword.input(
            fn=update_downloaded_choices,
            inputs=[downloaded_series, downloaded_keyword],
            outputs=[character_dropdown]
        )
        downloaded_keyword.input(
            fn=lambda: get_downloaded_character_details(None),
            outputs=[character_details]
        )

        character_dropdown.change(
            fn=get_downloaded_character_details,
            inputs=[character_dropdown],
            outputs=[character_details]
        )
        character_dropdown.change(
            fn=update_model_controls,
            inputs=[character_dropdown],
            outputs=[cover_speaker_id, cover_pitch_shift, cover_index_rate],
        )
        app.load(
            fn=update_model_controls, inputs=[character_dropdown],
            outputs=[cover_speaker_id, cover_pitch_shift, cover_index_rate],
        )

        custom_model_upload_btn.click(
            fn=import_custom_character_model_ui,
            inputs=[
                custom_model_name,
                custom_model_category,
                custom_model_source,
                custom_model_file,
                custom_index_file,
                downloaded_series,
                downloaded_keyword,
            ],
            outputs=[
                custom_model_status,
                character_dropdown,
                downloaded_series,
                downloaded_keyword,
                character_details,
            ],
        )

        download_series.input(
            fn=update_download_choices,
            inputs=[download_series, download_keyword],
            outputs=[download_char_dropdown]
        )
        download_series.input(
            fn=lambda: get_available_character_details(None),
            outputs=[download_char_details]
        )

        download_keyword.input(
            fn=update_download_choices,
            inputs=[download_series, download_keyword],
            outputs=[download_char_dropdown]
        )
        download_keyword.input(
            fn=lambda: get_available_character_details(None),
            outputs=[download_char_details]
        )

        download_char_dropdown.change(
            fn=get_available_character_details,
            inputs=[download_char_dropdown],
            outputs=[download_char_details]
        )

        download_char_btn.click(
            fn=download_character,
            inputs=[download_char_dropdown, downloaded_series, downloaded_keyword],
            outputs=[download_char_status, character_dropdown, downloaded_series]
        )
        download_char_btn.click(
            fn=get_available_character_details,
            inputs=[download_char_dropdown],
            outputs=[download_char_details]
        )

        download_all_series_btn.click(
            fn=download_all_characters,
            inputs=[download_series, downloaded_series, downloaded_keyword],
            outputs=[download_char_status, character_dropdown, downloaded_series]
        )
        download_all_series_btn.click(
            fn=lambda: get_downloaded_character_details(None),
            outputs=[character_details]
        )

        download_all_btn.click(
            fn=lambda series, keyword: download_all_characters(_all_series_label(), series, keyword),
            inputs=[downloaded_series, downloaded_keyword],
            outputs=[download_char_status, character_dropdown, downloaded_series]
        )
        download_all_btn.click(
            fn=lambda: get_downloaded_character_details(None),
            outputs=[character_details]
        )

        cover_mix_preset.input(
            fn=apply_cover_mix_preset,
            inputs=[cover_mix_preset],
            outputs=[
                cover_vocals_volume,
                cover_accompaniment_volume,
                cover_reverb
            ]
        )

        mature_deecho_check_btn.click(
            fn=check_mature_deecho_status,
            outputs=[mature_deecho_status]
        )
        mature_deecho_check_btn.click(
            fn=lambda mode, karaoke, input_mode: get_cover_vc_route_status(mode, karaoke_separation=karaoke, input_mode=input_mode),
            inputs=[cover_vc_pipeline_mode, cover_karaoke, cover_input_mode],
            outputs=[cover_vc_route_status]
        )

        mature_deecho_download_btn.click(
            fn=download_mature_deecho_models_ui,
            outputs=[mature_deecho_status]
        )
        mature_deecho_download_btn.click(
            fn=lambda mode, karaoke, input_mode: get_cover_vc_route_status(mode, karaoke_separation=karaoke, input_mode=input_mode),
            inputs=[cover_vc_pipeline_mode, cover_karaoke, cover_input_mode],
            outputs=[cover_vc_route_status]
        )

        cover_vc_pipeline_mode.input(
            fn=lambda mode, karaoke, input_mode: get_cover_vc_route_status(mode, karaoke_separation=karaoke, input_mode=input_mode),
            inputs=[cover_vc_pipeline_mode, cover_karaoke, cover_input_mode],
            outputs=[cover_vc_route_status]
        )
        cover_karaoke.input(
            fn=lambda mode, karaoke, input_mode: get_cover_vc_route_status(mode, karaoke_separation=karaoke, input_mode=input_mode),
            inputs=[cover_vc_pipeline_mode, cover_karaoke, cover_input_mode], outputs=[cover_vc_route_status],
        )
        cover_vc_pipeline_mode.input(
            fn=update_route_controls,
            inputs=[cover_vc_pipeline_mode, cover_karaoke],
            outputs=[cover_karaoke, cover_karaoke_merge_backing, cover_source_constraint_mode],
        )
        cover_karaoke.input(
            fn=lambda enabled: gr.update(value=bool(enabled), interactive=bool(enabled)),
            inputs=[cover_karaoke], outputs=[cover_karaoke_merge_backing],
        )
        app.load(
            fn=update_route_controls,
            inputs=[cover_vc_pipeline_mode, cover_karaoke, cover_source_constraint_mode, cover_karaoke_merge_backing],
            outputs=[cover_karaoke, cover_karaoke_merge_backing, cover_source_constraint_mode],
        )

        song_controls = [cover_mix_heading, cover_karaoke, cover_karaoke_merge_backing,
                         cover_dereverb_info, cover_source_constraint_mode, cover_mix_preset,
                         cover_vocals_volume, cover_accompaniment_volume, cover_reverb, cover_backing_mix]
        def update_input_mode(input_mode, mode, karaoke):
            return (*[gr.update(visible=input_mode == "song") for _ in song_controls],
                    get_cover_vc_route_status(mode, karaoke_separation=karaoke, input_mode=input_mode))
        cover_input_mode.input(fn=update_input_mode,
            inputs=[cover_input_mode, cover_vc_pipeline_mode, cover_karaoke],
            outputs=[*song_controls, cover_vc_route_status])
        app.load(fn=update_input_mode,
            inputs=[cover_input_mode, cover_vc_pipeline_mode, cover_karaoke],
            outputs=[*song_controls, cover_vc_route_status])

        cover_settings = [
            character_dropdown, cover_pitch_shift, cover_index_rate, cover_speaker_id,
            cover_karaoke, cover_karaoke_merge_backing, cover_source_constraint_mode,
            cover_vc_pipeline_mode, cover_vocals_volume, cover_accompaniment_volume,
            cover_reverb, cover_rms_mix_rate, cover_backing_mix, cover_input_mode,
        ]
        cover_outputs = [
            cover_output, cover_converted_vocals_output, cover_original_vocals_output,
            cover_lead_vocals_output, cover_backing_vocals_output, cover_accompaniment_output,
            cover_accompaniment_without_harmony_output, cover_download_btn,
            cover_converted_vocals_download_btn, cover_original_vocals_download_btn,
            cover_lead_vocals_download_btn, cover_backing_vocals_download_btn,
            cover_accompaniment_download_btn, cover_accompaniment_without_harmony_download_btn, cover_status,
        ]
        cover_event = cover_btn.click(
            fn=process_cover_with_downloads, js=PLAYER_CLEAR_JS,
            inputs=[cover_input_audio, *cover_settings, reuse_state], outputs=cover_outputs,
            concurrency_id="cover-inference", concurrency_limit=1,
        )
        # The player now shows a single result; the previous batch stays selectable.
        cover_event.success(fn=None, inputs=None, outputs=batch_picker,
                            js="() => [null]", queue=False, api_name=False)
        batch_event = batch_btn.click(
            fn=process_cover_batch, js=PLAYER_CLEAR_JS,
            inputs=[batch_files, *cover_settings, reuse_state],
            outputs=[batch_rows, batch_picker, batch_status, batch_download],
            concurrency_id="cover-inference", concurrency_limit=1,
        )
        first_result = batch_event.success(
            fn=select_batch_result, inputs=[batch_picker, batch_rows, reuse_state], outputs=cover_outputs,
        )
        picked_result = batch_picker.input(
            fn=select_batch_result, inputs=[batch_picker, batch_rows, reuse_state], outputs=cover_outputs,
        )
        for result_event in (cover_event, first_result, picked_result):
            result_event.then(fn=None, inputs=cover_outputs[:7], outputs=None, js=PLAYER_UPDATE_JS, queue=False)

        def change_language(choice, series, keyword, character, available_series, available_keyword, available_character,
                            source_mode, pipeline_mode, mix_preset, karaoke, input_mode, rows, session, *, persist=True):
            source_value = _resolve_labeled_choice(get_source_constraint_option_maps()[0], source_mode, 'source_constraint_mode')
            pipeline_value = _resolve_labeled_choice(get_vc_pipeline_mode_option_maps()[0], pipeline_mode, 'vc_pipeline_mode')
            preset_value = _resolve_labeled_choice(dict(get_cover_mix_preset_options()), mix_preset, 'mix_preset')
            status = save_language_setting(choice) if persist else ""
            series = _normalize_series_choice(series)
            available_series = _normalize_series_choice(available_series)
            source_choices, _ = get_source_constraint_option_maps()
            pipeline_choices, _ = get_vc_pipeline_mode_option_maps()
            dynamic_updates = (gr.update(choices=[(t("input_mode_song", "cover"), "song"), (t("input_mode_vocals", "cover"), "vocals")], value=input_mode), status,
                gr.update(choices=get_downloaded_character_series(), value=series),
                gr.update(choices=get_downloaded_character_choices(series, keyword), value=character),
                get_downloaded_character_details(character),
                gr.update(choices=[_all_series_label()] + get_available_character_series(), value=available_series),
                gr.update(choices=get_available_character_choices(available_series, available_keyword), value=available_character),
                get_available_character_details(available_character),
                gr.update(choices=list(source_choices.items()), value=source_value),
                gr.update(choices=list(pipeline_choices.items()), value=pipeline_value),
                gr.update(choices=get_cover_mix_preset_options(), value=preset_value),
                get_cover_vc_route_status(pipeline_value, karaoke_separation=karaoke, input_mode=input_mode),
                check_mature_deecho_status(), get_device_info(),
                gr.update(choices=_build_device_choices()),
                gr.update(value=get_model_table(), headers=[t('model_name','ui'),t('model_path','ui'),t('index_path','ui')]))
            updates = dict(zip(translated_components, refresh_translations(choice)))
            if rows:
                updates[batch_status]["value"] = batch_summary(rows, session.get("batch_archive_error"))
            if session.get("selected_status"):
                updates[cover_status]["value"] = cover_status_text(session["selected_status"])
            updates[language_dropdown] = {**updates.get(language_dropdown, {}),
                                          **gr.update(value=SUPPORTED_LANGUAGES[resolve_language_choice(choice)])}
            for component, update in zip(dynamic_components, dynamic_updates):
                update = update if isinstance(update, dict) else gr.update(value=update)
                updates[component] = {**updates.get(component, {}), **update}
            return updates

        translated_components, refresh_translations = bind_static_translations(app)
        dynamic_components = [cover_input_mode, language_status, downloaded_series, character_dropdown, character_details,
                     download_series, download_char_dropdown, download_char_details,
                     cover_source_constraint_mode, cover_vc_pipeline_mode, cover_mix_preset,
                     cover_vc_route_status, mature_deecho_status, device_info, device_radio, model_table]
        language_outputs = list(dict.fromkeys([language_dropdown, *dynamic_components, *translated_components]))
        language_inputs = [downloaded_series, downloaded_keyword, character_dropdown,
                           download_series, download_keyword, download_char_dropdown,
                           cover_source_constraint_mode, cover_vc_pipeline_mode, cover_mix_preset, cover_karaoke, cover_input_mode,
                           batch_rows, reuse_state]

        def restore_language(*values):
            choice = get_current_language_label()
            updates = change_language(choice, *values, persist=False)
            # A user can change these controls while the initial model checks run.
            # Their internal choices stay valid across languages.
            for control in (cover_input_mode, cover_source_constraint_mode, cover_vc_pipeline_mode, cover_mix_preset):
                updates[control].pop("value", None)
            return updates

        save_language_btn.click(
            fn=change_language,
            inputs=[language_dropdown, *language_inputs],
            outputs=language_outputs,
            concurrency_id="interface-language",
        ).success(fn=None, inputs=[language_dropdown], outputs=None, js=LANGUAGE_CHANGE_JS, queue=False)
        app.load(fn=restore_language, inputs=language_inputs,
                 outputs=language_outputs, concurrency_id="interface-language").then(
            fn=None, inputs=[language_dropdown], outputs=None, js=LANGUAGE_CHANGE_JS, queue=False)

    return app


def _patch_gradio_file_download(blocks):
    """Keep Gradio file authorization/range handling and supply a clean filename."""
    from starlette.datastructures import MutableHeaders
    from urllib.parse import quote, unquote
    from gradio.route_utils import API_PREFIX
    from ui.gradio_routes import iter_gradio_routes

    fastapi_app = getattr(blocks, "server_app", None)
    if fastapi_app is None:
        raise RuntimeError("Gradio server must be started before patching downloads")
    routes = [route for route in iter_gradio_routes(fastapi_app.routes) if getattr(route, 'path', None)
              in (API_PREFIX + '/file={path_or_url:path}', '/file={path_or_url:path}')]
    if not routes:
        raise RuntimeError("Gradio file route was not found")
    if getattr(fastapi_app, '_ai_rvc_download_patch_applied', False):
        return
    original_app = fastapi_app.middleware_stack or fastapi_app.build_middleware_stack()

    async def patched_file_app(scope, receive, send):
        async def send_with_download_name(message):
            if (message.get('type') == 'http.response.start'
                    and 200 <= message.get('status', 200) < 300
                    and any(scope.get('route') is route for route in routes)):
                raw = scope.get('path_params', {}).get('path_or_url')
                if raw is None:
                    raw = unquote(scope.get('path', '').split('/file=', 1)[-1])
                name = re.split(r"[\\/]", str(raw))[-1]
                encoded = quote(clean_gradio_temp_download_name(name), safe='')
                MutableHeaders(scope=message)['content-disposition'] = "inline; filename*=utf-8''" + encoded
            await send(message)

        await original_app(scope, receive, send_with_download_name)

    # Lazy FastAPI routes build a request handler dynamically and bypass route.app.
    # Wrap ASGI responses instead, after Gradio has created its application stack.
    fastapi_app.middleware_stack = patched_file_app
    fastapi_app._ai_rvc_download_patch_applied = True


def launch(
    host: str = "127.0.0.1", port: int = 7860, share: bool = False,
    *, open_browser: bool = True, auth=None, root_path: Optional[str] = None,
):
    """启动 Gradio 界面"""
    from ui.server_runtime import configure_browser_server
    configure_browser_server()
    app = create_ui()
    app.queue()  # 启用队列以支持进度跟踪
    output_directory = (ROOT_DIR / config.get("paths", {}).get(
        "outputs", config.get("output_dir", "outputs")
    )).resolve()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=False,
        prevent_thread_lock=True,
        i18n=ui_catalog(),
        auth=auth,
        root_path=root_path,
        allowed_paths=[str(output_directory)],
    )
    _patch_gradio_file_download(app)
    from ui.gradio_assets import patch_upload_route
    patch_upload_route(app)
    if open_browser:
        import webbrowser
        webbrowser.open(app.local_url)
    app.block_thread()


if __name__ == "__main__":
    launch()
