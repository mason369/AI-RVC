# -*- coding: utf-8 -*-
"""
Gradio 界面 - RVC AI 翻唱
"""
import os
import json
import re
import tempfile
import gradio as gr
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from lib.logger import log
from lib.runtime_build import get_runtime_build_label
from infer.separator import ROFORMER_DEFAULT_MODEL, KARAOKE_DEFAULT_MODEL

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config.json"
DEFAULT_LANGUAGE = "zh_CN"
SUPPORTED_LANGUAGES = {
    "zh_CN": "中文",
    "en_US": "English",
}
LANGUAGE_LABEL_TO_CODE = {label: code for code, label in SUPPORTED_LANGUAGES.items()}

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
    """加载配置"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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
    """Persist UI language selection. Static Gradio labels update after restart."""
    global config, i18n

    language = resolve_language_choice(language_choice)
    target_path = Path(config_path) if config_path is not None else CONFIG_PATH

    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            next_config = normalize_config(json.load(f))
    else:
        next_config = {}

    next_config["language"] = language
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(next_config, f, indent=4, ensure_ascii=False)

    if target_path == CONFIG_PATH:
        config = next_config
        i18n = load_i18n(language)

    message_i18n = load_i18n(language)
    return message_i18n["settings"]["language_saved_restart"].format(
        language=SUPPORTED_LANGUAGES[language]
    )


def _to_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _to_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def get_cover_mix_defaults() -> Dict[str, int]:
    """获取翻唱混音默认值"""
    cover_cfg = config.get("cover", {})
    return {
        "vocals_volume": _to_int(cover_cfg.get("default_vocals_volume", 100), 100),
        "accompaniment_volume": _to_int(cover_cfg.get("default_accompaniment_volume", 100), 100),
        "reverb": _to_int(cover_cfg.get("default_reverb", 10), 10),
    }


def get_cover_mix_presets() -> Tuple[Dict[str, Dict[str, int]], str]:
    """获取混音预设与默认预设名称"""
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
    """根据预设名称返回混音参数"""
    presets, default_name = get_cover_mix_presets()
    preset = presets.get(preset_name) or presets[default_name]
    return preset["vocals_volume"], preset["accompaniment_volume"], preset["reverb"]



def get_vc_preprocess_option_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build VC preprocess dropdown option maps."""
    label_to_value = {
        t("vc_preprocess_auto", "cover"): "auto",
        t("vc_preprocess_uvr_deecho", "cover"): "uvr_deecho",
    }
    value_to_label = {value: label for label, value in label_to_value.items()}
    return label_to_value, value_to_label


def get_source_constraint_option_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build source constraint dropdown option maps."""
    label_to_value = {
        t("source_constraint_auto", "cover"): "auto",
        t("source_constraint_off", "cover"): "off",
        t("source_constraint_on", "cover"): "on",
    }
    value_to_label = {value: label for label, value in label_to_value.items()}
    return label_to_value, value_to_label


def get_vc_pipeline_mode_option_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build VC pipeline mode dropdown option maps."""
    label_to_value = {
        t("vc_pipeline_mode_current", "cover"): "current",
        t("vc_pipeline_mode_official", "cover"): "official",
    }
    value_to_label = {value: label for label, value in label_to_value.items()}
    return label_to_value, value_to_label


def update_singing_repair_visibility(vc_pipeline_mode: str):
    """Only show singing repair option for official mode."""
    pipeline_label_to_value, _ = get_vc_pipeline_mode_option_maps()
    normalized = pipeline_label_to_value.get(
        str(vc_pipeline_mode),
        str(vc_pipeline_mode or "").strip().lower(),
    )
    return gr.update(visible=(normalized == "official"))


def init_pipeline():
    """初始化推理管道"""
    global pipeline

    if pipeline is not None:
        return pipeline

    from infer.pipeline import VoiceConversionPipeline

    device = config.get("device", "cuda")
    pipeline = VoiceConversionPipeline(device=device)
    pipeline.hubert_layer = config.get("hubert_layer", 12)

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


def format_character_label(char_info: dict) -> str:
    """格式化角色展示名称，明确显示版本、归属和来源。"""
    display = char_info.get("base_display") or char_info.get("display") or char_info.get("description") or char_info.get("name", "")
    source = char_info.get("source") or _unknown_label()
    name = char_info.get("name", "")
    lang_tag = get_character_language_tag(char_info)
    parts: List[str] = []
    continuity = str(char_info.get("continuity") or "").strip()
    version_label = str(char_info.get("version_label") or "").strip()
    distribution = str(char_info.get("distribution") or "").strip()
    repo = str(char_info.get("repo") or "").strip()

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

    version_label = str(char_info.get("version_label") or "").strip()
    continuity = str(char_info.get("continuity") or "").strip()
    role = str(char_info.get("role") or "").strip()
    source = str(char_info.get("source") or "").strip()
    distribution = str(char_info.get("distribution") or "").strip()
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
        gr.update(choices=get_downloaded_character_choices(series, keyword))
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
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return status, choices_update, series_update
    except Exception as e:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return tf("bulk_download_error", "messages", error=str(e)), choices_update, series_update


def update_download_choices(series: str, keyword: str) -> Dict:
    """更新下载下拉列表"""
    return gr.update(choices=get_available_character_choices(series, keyword))


def update_downloaded_choices(series: str, keyword: str) -> Dict:
    """更新已下载角色下拉列表"""
    return gr.update(choices=get_downloaded_character_choices(series, keyword))


def refresh_downloaded_controls(series: str, keyword: str) -> Tuple[Dict, Dict]:
    """刷新已下载角色的筛选和列表"""
    return _refresh_downloaded_updates(series, keyword)


def process_cover(
    audio_path: str,
    character_name: str,
    pitch_shift: int,
    index_ratio: float,
    speaker_id: float,
    karaoke_separation: bool,
    karaoke_merge_backing_into_accompaniment: bool,
    vc_preprocess_mode: str,
    source_constraint_mode: str,
    vc_pipeline_mode: str,
    singing_repair: bool,
    vocals_volume: float,
    accompaniment_volume: float,
    reverb_amount: float,
    rms_mix_rate: float,
    backing_mix: float,
    progress=gr.Progress()
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], str]:
    """
    处理翻唱

    Returns:
        Tuple[cover, converted_vocals, original_vocals, lead_vocals, backing_vocals, accompaniment, status]
    """
    _none6 = (None, None, None, None, None, None)
    if audio_path is None:
        return *_none6, t("please_upload_song", "messages")

    if not character_name:
        return *_none6, t("please_select_character", "messages")

    try:
        from tools.character_models import get_character_model_path, get_character_info
        from infer.cover_pipeline import get_cover_pipeline

        # 获取角色模型路径
        resolved_name = resolve_character_name(character_name)
        char_meta = get_character_info(resolved_name, downloaded_only=True) or {}
        model_info = get_character_model_path(resolved_name)
        if model_info is None:
            return *_none6, tf("character_model_missing", "messages", name=resolved_name)

        # 进度回调
        def progress_callback(msg: str, step: int, total: int):
            if total > 0:
                progress(step / total, desc=msg)

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
        filter_radius = cover_cfg.get("filter_radius", config.get("filter_radius", 3))
        protect = cover_cfg.get("protect", config.get("protect", 0.33))
        silence_gate = cover_cfg.get("silence_gate", True)
        silence_threshold_db = cover_cfg.get("silence_threshold_db", -40.0)
        silence_smoothing_ms = cover_cfg.get("silence_smoothing_ms", 50.0)
        silence_min_duration_ms = cover_cfg.get("silence_min_duration_ms", 200.0)
        hubert_layer = cover_cfg.get("hubert_layer", config.get("hubert_layer", 12))
        karaoke_model = cover_cfg.get(
            "karaoke_model",
            KARAOKE_DEFAULT_MODEL,
        )
        default_vc_preprocess_mode = str(cover_cfg.get("vc_preprocess_mode", "auto"))
        default_source_constraint_mode = str(cover_cfg.get("source_constraint_mode", "auto"))
        default_vc_pipeline_mode = str(cover_cfg.get("vc_pipeline_mode", "current"))
        default_singing_repair = bool(cover_cfg.get("singing_repair", False))
        vc_label_to_value, vc_value_to_label = get_vc_preprocess_option_maps()
        source_label_to_value, source_value_to_label = get_source_constraint_option_maps()
        pipeline_label_to_value, pipeline_value_to_label = get_vc_pipeline_mode_option_maps()

        vc_preprocess_mode = vc_label_to_value.get(str(vc_preprocess_mode), str(vc_preprocess_mode or default_vc_preprocess_mode).strip().lower())
        if vc_preprocess_mode not in {"auto", "uvr_deecho"}:
            vc_preprocess_mode = "auto"
        source_constraint_mode = source_label_to_value.get(str(source_constraint_mode), str(source_constraint_mode or default_source_constraint_mode).strip().lower())
        if source_constraint_mode not in {"auto", "off", "on"}:
            source_constraint_mode = default_source_constraint_mode
        vc_pipeline_mode = pipeline_label_to_value.get(str(vc_pipeline_mode), str(vc_pipeline_mode or default_vc_pipeline_mode).strip().lower())
        if vc_pipeline_mode not in {"current", "official"}:
            vc_pipeline_mode = default_vc_pipeline_mode
        singing_repair = bool(singing_repair if singing_repair is not None else default_singing_repair)

        index_ratio = max(0.0, min(1.0, float(index_ratio) / 100.0))
        speaker_id = int(max(0, round(float(speaker_id))))
        rms_mix_rate = max(0.0, min(1.0, float(rms_mix_rate) / 100.0))
        backing_mix = max(0.0, min(1.0, float(backing_mix) / 100.0))

        # 输出目录
        output_dir = ROOT_DIR / config.get("paths", {}).get(
            "outputs",
            config.get("output_dir", "outputs")
        )

        # 执行翻唱
        result = pipeline.process(
            input_audio=audio_path,
            model_path=model_info["model_path"],
            index_path=model_info.get("index_path"),
            pitch_shift=pitch_shift,
            index_ratio=index_ratio,
            filter_radius=filter_radius,
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
            hubert_layer=hubert_layer,
            silence_gate=silence_gate,
            silence_threshold_db=silence_threshold_db,
            silence_smoothing_ms=silence_smoothing_ms,
            silence_min_duration_ms=silence_min_duration_ms,
            vocals_volume=vocals_volume / 100,  # 转换为 0-2 范围
            accompaniment_volume=accompaniment_volume / 100,
            reverb_amount=reverb_amount / 100,
            backing_mix=backing_mix,
            karaoke_separation=bool(karaoke_separation),
            karaoke_model=karaoke_model,
            karaoke_merge_backing_into_accompaniment=bool(karaoke_merge_backing_into_accompaniment),
            vc_preprocess_mode=vc_preprocess_mode,
            source_constraint_mode=source_constraint_mode,
            vc_pipeline_mode=vc_pipeline_mode,
            singing_repair=singing_repair,
            output_dir=str(output_dir),
            model_display_name=resolved_name,
            progress_callback=progress_callback
        )

        status_msg = t("cover_complete_status", "messages")
        status_msg += f"\n{get_cover_vc_route_status(vc_preprocess_mode, vc_pipeline_mode).splitlines()[0]}"
        status_msg += "\n" + tf(
            "vc_pipeline_mode_status",
            "messages",
            value=pipeline_value_to_label.get(vc_pipeline_mode, vc_pipeline_mode),
        )
        status_msg += "\n" + tf(
            "singing_repair_status",
            "messages",
            value=_bool_status_label(singing_repair),
        )
        status_msg += "\n" + tf(
            "source_constraint_status",
            "messages",
            value=source_value_to_label.get(source_constraint_mode, source_constraint_mode),
        )
        if char_meta.get("version_label"):
            status_msg += "\n" + tf("model_version_status", "messages", value=char_meta["version_label"])
        if char_meta.get("continuity"):
            status_msg += "\n" + tf("character_continuity_status", "messages", value=char_meta["continuity"])
        if char_meta.get("repo"):
            status_msg += "\n" + tf("model_source_status", "messages", value=char_meta["repo"])
        status_msg += f"\n{get_runtime_build_label()}"
        if result.get("all_files_dir"):
            status_msg += "\n" + tf("all_files_dir_status", "messages", value=result["all_files_dir"])

        return (
            result["cover"],
            result["converted_vocals"],
            result.get("vocals"),
            result.get("lead_vocals"),
            result.get("backing_vocals"),
            result["accompaniment"],
            status_msg
        )

    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else traceback.format_exc()
        log.error(f"处理失败: {error_msg}")
        return None, None, None, None, None, None, tf("cover_process_failed", "messages", error=error_msg)


def check_mature_deecho_status() -> str:
    """Check mature DeEcho model availability."""
    from tools.download_models import MATURE_DEECHO_MODELS, check_model
    from infer.separator import ROFORMER_DEREVERB_DEFAULT_MODEL, check_roformer_available

    status_lines = []
    roformer_ready = check_roformer_available()
    icon = "✅" if roformer_ready else "❌"
    status_lines.append(
        f"{icon} {ROFORMER_DEREVERB_DEFAULT_MODEL}  {t('mature_auto_preferred_suffix', 'route_status')}"
    )
    if roformer_ready:
        status_lines.append(t("mature_roformer_auto_download_note", "route_status"))

    for name in MATURE_DEECHO_MODELS:
        exists = check_model(name)
        icon = "✅" if exists else "❌"
        suffix = t("mature_legacy_status_suffix", "route_status")
        status_lines.append(f"{icon} {name}{suffix}")

    if roformer_ready:
        status_lines.append("")
        status_lines.append(tf("mature_current_preferred", "route_status", model=f"RoFormer {ROFORMER_DEREVERB_DEFAULT_MODEL}"))
    else:
        status_lines.append("")
        status_lines.append(t("mature_missing_strict", "route_status"))

    return "\n".join(status_lines)


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
    vc_preprocess_mode: Optional[str] = None,
    vc_pipeline_mode: Optional[str] = None,
) -> str:
    """Return the active VC route shown in the cover UI."""
    from infer.separator import ROFORMER_DEREVERB_DEFAULT_MODEL, check_roformer_available

    mode = str(vc_preprocess_mode or config.get("cover", {}).get("vc_preprocess_mode", "auto")).strip().lower()
    pipeline_mode = str(vc_pipeline_mode or config.get("cover", {}).get("vc_pipeline_mode", "current")).strip().lower()
    vc_label_to_value, _ = get_vc_preprocess_option_maps()
    pipeline_label_to_value, _ = get_vc_pipeline_mode_option_maps()
    mode = vc_label_to_value.get(mode, mode)
    pipeline_mode = pipeline_label_to_value.get(pipeline_mode, pipeline_mode)
    roformer_ready = check_roformer_available()
    preferred = f"RoFormer {ROFORMER_DEREVERB_DEFAULT_MODEL}" if roformer_ready else None
    newline = chr(10)
    build_label = get_runtime_build_label()

    if pipeline_mode == "official":
        return newline.join([
            t("official_route_title", "route_status"),
            t("official_route_flow", "route_status"),
            t("official_route_note", "route_status"),
            build_label,
        ])

    if mode == "uvr_deecho":
        if preferred:
            return newline.join([
                t("strict_route_ready_title", "route_status"),
                tf("route_current_model", "route_status", model=preferred),
                t("strict_route_flow", "route_status"),
                build_label,
            ])
        return newline.join([
            t("strict_route_unavailable_title", "route_status"),
            t("strict_route_unavailable_flow", "route_status"),
            t("strict_route_unavailable_advice", "route_status"),
            build_label,
        ])

    if preferred:
        return newline.join([
            t("auto_route_ready_title", "route_status"),
            tf("route_current_model", "route_status", model=preferred),
            t("strict_route_flow", "route_status"),
            build_label,
        ])
    return newline.join([
        t("auto_route_missing_title", "route_status"),
        t("auto_route_missing_reason", "route_status"),
        t("strict_route_unavailable_flow", "route_status"),
        build_label,
    ])


def check_models_status() -> str:
    """检查模型状态"""
    from tools.download_models import check_model, REQUIRED_MODELS

    status_lines = []
    for name in REQUIRED_MODELS:
        exists = check_model(name)
        icon = "✅" if exists else "❌"
        status_lines.append(f"{icon} {name}")

    return "\n".join(status_lines)


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


# 自定义 CSS - 深灰 + 橙色强调配色
CUSTOM_CSS = """
/* 深色主题基础 - 纯色背景 */
.gradio-container {
    background: #121212 !important;
    min-height: 100vh;
}

.main-title {
    text-align: center;
    margin-bottom: 1rem;
    color: #e0e0e0 !important;
}

.runtime-stamp {
    text-align: center;
    margin: -0.3rem 0 1rem 0;
    color: #bdbdbd !important;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.95rem;
}

/* 状态框样式 */
.status-box {
    font-family: 'Consolas', 'Monaco', monospace;
    white-space: pre-wrap;
    background: #1e1e1e !important;
    border: 1px solid #404040 !important;
    color: #9e9e9e !important;
}

/* 提示框 */
.model-hint {
    padding: 1rem;
    background: #1e1e1e !important;
    border: 1px solid #404040 !important;
    border-radius: 8px;
    margin: 1rem 0;
    color: #e0e0e0 !important;
}

/* 成功/错误消息 */
.success-msg {
    color: #4caf50 !important;
    font-weight: bold;
}
.error-msg {
    color: #f44336 !important;
    font-weight: bold;
}

/* 标签页样式 */
.tabs > .tab-nav {
    background: #1e1e1e !important;
    border-bottom: 1px solid #404040 !important;
}
.tabs > .tab-nav > button {
    color: #9e9e9e !important;
    background: transparent !important;
    border: none !important;
    padding: 12px 24px !important;
    transition: color 0.2s ease !important;
}
.tabs > .tab-nav > button:hover {
    color: #e0e0e0 !important;
}
.tabs > .tab-nav > button.selected {
    color: #ff9800 !important;
    border-bottom: 2px solid #ff9800 !important;
    background: transparent !important;
}

/* 输入框和下拉框 */
.gr-input, .gr-dropdown, textarea, input[type="text"] {
    background: #2d2d2d !important;
    border: 1px solid #404040 !important;
    color: #e0e0e0 !important;
}
.gr-input:focus, .gr-dropdown:focus, textarea:focus, input[type="text"]:focus {
    border-color: #ff9800 !important;
    outline: none !important;
}

/* 滑块 */
.gr-slider input[type="range"] {
    background: #404040 !important;
}
.gr-slider input[type="range"]::-webkit-slider-thumb {
    background: #ff9800 !important;
}
.gr-slider input[type="range"]::-moz-range-thumb {
    background: #ff9800 !important;
}
input[type="range"]::-webkit-slider-runnable-track {
    background: #404040 !important;
}
input[type="range"]::-moz-range-track {
    background: #404040 !important;
}

/* 按钮样式 - 主按钮橙色 */
.gr-button-primary, button.primary {
    background: #ff9800 !important;
    border: none !important;
    color: #121212 !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: #ffa726 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(255, 152, 0, 0.3) !important;
}
.gr-button-primary:active, button.primary:active {
    background: #f57c00 !important;
    transform: translateY(0) !important;
}

/* 次要按钮 */
.gr-button-secondary, button.secondary {
    background: #404040 !important;
    border: none !important;
    color: #e0e0e0 !important;
    transition: all 0.2s ease !important;
}
.gr-button-secondary:hover, button.secondary:hover {
    background: #4a4a4a !important;
}

/* 音频播放器 */
.gr-audio {
    background: #1e1e1e !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
}
.gr-audio audio {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
    accent-color: #ff9800 !important;
}
.gr-audio audio::-webkit-media-controls-panel {
    background: #1e1e1e !important;
}
.gr-audio audio::-webkit-media-controls-enclosure {
    background: #1e1e1e !important;
}
.gr-audio audio::-webkit-media-controls-timeline {
    background: #404040 !important;
}
.gr-audio audio::-webkit-media-controls-current-time-display,
.gr-audio audio::-webkit-media-controls-time-remaining-display {
    color: #e0e0e0 !important;
}
.gr-audio audio::-webkit-media-controls-play-button,
.gr-audio audio::-webkit-media-controls-mute-button,
.gr-audio audio::-webkit-media-controls-volume-slider {
    filter: invert(1) sepia(1) saturate(5) hue-rotate(10deg) !important;
}

/* 折叠面板 */
.gr-accordion {
    background: #1e1e1e !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
}
.gr-accordion > .label-wrap {
    background: #1e1e1e !important;
}

/* 表格 */
.gr-dataframe {
    background: #1e1e1e !important;
}
.gr-dataframe table {
    color: #e0e0e0 !important;
}
.gr-dataframe th {
    background: #2d2d2d !important;
    color: #9e9e9e !important;
}
.gr-dataframe td {
    background: #1e1e1e !important;
    border-color: #404040 !important;
}
.gr-dataframe tr:hover td {
    background: #333333 !important;
}
/* Gradio v4 Dataframe */
div[data-testid="dataframe"] {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
    border: 1px solid #404040 !important;
}
div[data-testid="dataframe"] table {
    color: #e0e0e0 !important;
}
div[data-testid="dataframe"] thead th {
    background: #2d2d2d !important;
    color: #9e9e9e !important;
    border-color: #404040 !important;
}
div[data-testid="dataframe"] tbody td {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
    border-color: #404040 !important;
}
div[data-testid="dataframe"] tbody tr:hover td {
    background: #333333 !important;
}
div[data-testid="dataframe"] input,
div[data-testid="dataframe"] textarea {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
    border: 1px solid #404040 !important;
}

/* Markdown 文本 */
.prose {
    color: #e0e0e0 !important;
}
.prose h1, .prose h2, .prose h3, .prose h4 {
    color: #e0e0e0 !important;
}
.prose a {
    color: #ff9800 !important;
}
.prose a:hover {
    color: #ffa726 !important;
}
.prose code {
    background: #2d2d2d !important;
    color: #ff9800 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
.prose blockquote {
    border-left: 3px solid #ff9800 !important;
    background: #1e1e1e !important;
    padding: 8px 16px !important;
    color: #9e9e9e !important;
}

/* 单选按钮和复选框 */
.gr-radio label, .gr-checkbox label {
    color: #e0e0e0 !important;
}
input[type="radio"]:checked + label, input[type="checkbox"]:checked + label {
    color: #ff9800 !important;
}

/* 进度条 */
.progress-bar {
    background: #404040 !important;
}
.progress-bar > div {
    background: #ff9800 !important;
}

/* 分隔线 */
hr {
    border-color: #404040 !important;
}

/* 标签 */
label {
    color: #9e9e9e !important;
}

/* 信息文本 */
.gr-info {
    color: #9e9e9e !important;
}

/* 块/面板背景 */
.gr-block, .gr-box, .gr-panel {
    background: #1e1e1e !important;
    border-color: #404040 !important;
}

/* 下拉菜单选项 */
.gr-dropdown option, select option {
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
}

/* Gradio 下拉选择器完整样式 */
.gr-dropdown, .gr-dropdown select,
div[data-testid="dropdown"],
.dropdown-container,
.svelte-select,
.wrap-inner,
.secondary-wrap {
    background: #2d2d2d !important;
    border: 1px solid #404040 !important;
    color: #e0e0e0 !important;
}

/* 下拉选择器输入框 */
.gr-dropdown input,
div[data-testid="dropdown"] input,
.svelte-select input {
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: none !important;
}

/* 下拉菜单列表 */
.gr-dropdown ul,
.gr-dropdown .options,
div[data-testid="dropdown"] ul,
.svelte-select .listContainer,
.dropdown-menu,
ul[role="listbox"] {
    background: #2d2d2d !important;
    border: 1px solid #404040 !important;
    color: #e0e0e0 !important;
}

/* 下拉菜单选项 */
.gr-dropdown li,
.gr-dropdown .option,
div[data-testid="dropdown"] li,
.svelte-select .listItem,
li[role="option"] {
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
}

/* 下拉菜单选项悬停 */
.gr-dropdown li:hover,
.gr-dropdown .option:hover,
div[data-testid="dropdown"] li:hover,
.svelte-select .listItem:hover,
.svelte-select .listItem.hover,
li[role="option"]:hover {
    background: #404040 !important;
    color: #ff9800 !important;
}

/* 下拉菜单选中项 */
.gr-dropdown li.selected,
.gr-dropdown .option.selected,
.svelte-select .listItem.active,
li[role="option"][aria-selected="true"] {
    background: #333333 !important;
    color: #ff9800 !important;
}

/* 下拉箭头图标 */
.gr-dropdown svg,
div[data-testid="dropdown"] svg,
.svelte-select .indicator svg {
    fill: #9e9e9e !important;
    color: #9e9e9e !important;
}

/* Gradio 3/4 selector compatibility */
.wrap.svelte-1m1zvyj,
.wrap-inner.svelte-1m1zvyj,
.secondary-wrap.svelte-1m1zvyj {
    background: #2d2d2d !important;
    border-color: #404040 !important;
}

.dropdown.svelte-1m1zvyj,
.options.svelte-1m1zvyj {
    background: #2d2d2d !important;
    border: 1px solid #404040 !important;
}

.item.svelte-1m1zvyj {
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
}

.item.svelte-1m1zvyj:hover,
.item.svelte-1m1zvyj.active {
    background: #404040 !important;
    color: #ff9800 !important;
}

/* 单选按钮组样式 */
.gr-radio,
.gr-radio-group,
div[data-testid="radio"] {
    background: transparent !important;
}

.gr-radio label span,
div[data-testid="radio"] label span {
    color: #e0e0e0 !important;
}

.gr-radio input[type="radio"],
div[data-testid="radio"] input[type="radio"] {
    accent-color: #ff9800 !important;
}

/* Radio 按钮容器 */
.radio-group,
.gr-radio-row {
    background: #1e1e1e !important;
}

.radio-group label,
.gr-radio-row label {
    background: #2d2d2d !important;
    border: 1px solid #404040 !important;
    color: #e0e0e0 !important;
}

.radio-group label:hover,
.gr-radio-row label:hover {
    background: #333333 !important;
}

.radio-group label.selected,
.gr-radio-row label.selected,
.radio-group input:checked + label,
.gr-radio-row input:checked + label {
    background: #333333 !important;
    border-color: #ff9800 !important;
    color: #ff9800 !important;
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #1e1e1e;
}
::-webkit-scrollbar-thumb {
    background: #404040;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4a4a4a;
}

/* Dataframe 表头修复 - 强制深色主题 */
table thead th,
table thead td,
.table-wrap thead th,
.table-wrap thead td,
[data-testid="table"] thead th,
[data-testid="table"] thead td {
    background: #2d2d2d !important;
    color: #ff9800 !important;
    border-color: #404040 !important;
}

/* Gradio 4/5 Dataframe 表头 */
.svelte-1kcgrqr thead th,
.svelte-1kcgrqr thead td,
.cell-wrap span,
th .cell-wrap,
th span.svelte-1kcgrqr {
    background: #2d2d2d !important;
    color: #ff9800 !important;
}

/* 音频播放器进度条修复 */
audio::-webkit-media-controls-timeline {
    background: linear-gradient(to right, #ff9800 var(--buffered-width, 0%), #404040 var(--buffered-width, 0%)) !important;
    border-radius: 4px !important;
    height: 4px !important;
}

/* 音频播放器 - Gradio 组件内部 */
.audio-container input[type="range"],
.waveform-container input[type="range"],
div[data-testid="audio"] input[type="range"],
div[data-testid="waveform"] input[type="range"] {
    accent-color: #ff9800 !important;
}

/* WaveSurfer 波形进度条 */
.wavesurfer-region,
.wavesurfer-handle,
wave > wave {
    background: #ff9800 !important;
}

/* Gradio Audio 组件进度条 */
.audio-player input[type="range"]::-webkit-slider-runnable-track {
    background: linear-gradient(to right, #ff9800 0%, #ff9800 var(--value, 0%), #404040 var(--value, 0%), #404040 100%) !important;
}

.audio-player input[type="range"]::-moz-range-track {
    background: linear-gradient(to right, #ff9800 0%, #ff9800 var(--value, 0%), #404040 var(--value, 0%), #404040 100%) !important;
}

.audio-player input[type="range"]::-webkit-slider-thumb {
    background: #ff9800 !important;
}

.audio-player input[type="range"]::-moz-range-thumb {
    background: #ff9800 !important;
}

/* 通用 range input 进度样式 */
input[type="range"] {
    accent-color: #ff9800 !important;
}

/* Gradio 4/5 音频波形 */
.waveform-container,
.audio-container {
    --waveform-color: #ff9800 !important;
    --progress-color: #ff9800 !important;
}
"""


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    with gr.Blocks(
        title=t("app_title"),
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="gray",
            neutral_hue="gray",
        ).set(
            # 背景色
            body_background_fill="#121212",
            body_background_fill_dark="#121212",
            # 面板/卡片背景
            block_background_fill="#1e1e1e",
            block_background_fill_dark="#1e1e1e",
            # 边框
            block_border_color="#404040",
            block_border_color_dark="#404040",
            # 标签背景
            block_label_background_fill="#2d2d2d",
            block_label_background_fill_dark="#2d2d2d",
            # 标签文字
            block_label_text_color="#9e9e9e",
            block_label_text_color_dark="#9e9e9e",
            # 标题文字
            block_title_text_color="#e0e0e0",
            block_title_text_color_dark="#e0e0e0",
            # 输入框
            input_background_fill="#2d2d2d",
            input_background_fill_dark="#2d2d2d",
            input_border_color="#404040",
            input_border_color_dark="#404040",
            # 主按钮 - 橙色
            button_primary_background_fill="#ff9800",
            button_primary_background_fill_dark="#ff9800",
            button_primary_background_fill_hover="#ffa726",
            button_primary_background_fill_hover_dark="#ffa726",
            button_primary_text_color="#121212",
            button_primary_text_color_dark="#121212",
            # 次要按钮 - 深灰
            button_secondary_background_fill="#404040",
            button_secondary_background_fill_dark="#404040",
            button_secondary_background_fill_hover="#4a4a4a",
            button_secondary_background_fill_hover_dark="#4a4a4a",
            button_secondary_text_color="#e0e0e0",
            button_secondary_text_color_dark="#e0e0e0",
            # 文字颜色
            body_text_color="#e0e0e0",
            body_text_color_dark="#e0e0e0",
            body_text_color_subdued="#9e9e9e",
            body_text_color_subdued_dark="#9e9e9e",
            # 链接颜色 - 橙色
            link_text_color="#ff9800",
            link_text_color_dark="#ff9800",
            link_text_color_hover="#ffa726",
            link_text_color_hover_dark="#ffa726",
            # 滑块颜色
            slider_color="#ff9800",
            slider_color_dark="#ff9800",
            # 复选框/单选框
            checkbox_background_color="#2d2d2d",
            checkbox_background_color_dark="#2d2d2d",
            checkbox_border_color="#404040",
            checkbox_border_color_dark="#404040",
            checkbox_label_text_color="#e0e0e0",
            checkbox_label_text_color_dark="#e0e0e0",
        ),
        css=CUSTOM_CSS
    ) as app:

        # 标题
        gr.Markdown(
            f"# 🎤 {t('app_title')}",
            elem_classes=["main-title"]
        )
        gr.Markdown(
            f"<center>{t('app_description')}</center>"
        )
        gr.Markdown(
            f"<div class='runtime-stamp'>{get_runtime_build_label()}</div>"
        )

        with gr.Row(elem_classes=["language-switch-row"]):
            language_dropdown = gr.Dropdown(
                label=t("language", "settings"),
                choices=list(LANGUAGE_LABEL_TO_CODE.keys()),
                value=get_current_language_label(),
                interactive=True,
            )
            save_language_btn = gr.Button(
                f"💾 {t('save_language', 'settings')}",
                variant="secondary",
            )

        language_status = gr.Markdown(t("language_restart_info", "settings"))
        save_language_btn.click(
            fn=save_language_setting,
            inputs=[language_dropdown],
            outputs=[language_status],
        )

        with gr.Tabs():
            # ===== 模型管理标签页 =====
            with gr.Tab(t("models", "tabs")):
                gr.Markdown(f"### 📦 {t('base_models', 'models')}")
                gr.Markdown(t("base_models_desc", "models"))

                with gr.Row():
                    check_btn = gr.Button(
                        f"🔍 {t('check_status', 'models')}",
                        variant="secondary"
                    )
                    download_btn = gr.Button(
                        f"⬇️ {t('download_required', 'models')}",
                        variant="primary"
                    )

                model_status = gr.Textbox(
                    label=t("model_status", "models"),
                    interactive=False,
                    lines=6,
                    elem_classes=["status-box"]
                )

                check_btn.click(
                    fn=check_models_status,
                    outputs=[model_status]
                )

                download_btn.click(
                    fn=download_base_models,
                    outputs=[model_status]
                )

                gr.Markdown("---")
                gr.Markdown(f"### 🎛️ {t('mature_deecho_models', 'models')}")
                gr.Markdown(t("mature_deecho_models_desc", "models"))

                with gr.Row():
                    mature_deecho_check_btn = gr.Button(
                        f"🔍 {t('mature_deecho_check', 'models')}",
                        variant="secondary"
                    )
                    mature_deecho_download_btn = gr.Button(
                        f"⬇️ {t('download_mature_deecho', 'models')}",
                        variant="primary"
                    )

                mature_deecho_status = gr.Textbox(
                    label=t("mature_deecho_status", "models"),
                    interactive=False,
                    lines=7,
                    value=check_mature_deecho_status(),
                    elem_classes=["status-box"]
                )

                gr.Markdown("---")

                gr.Markdown(f"### 🎤 {t('voice_models', 'models')}")
                gr.Markdown(t("voice_models_desc", "models"))

                def get_model_table():
                    from infer.pipeline import list_voice_models
                    weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")
                    models = list_voice_models(str(weights_dir))
                    if not models:
                        return [[t("no_models", "ui"), "", ""]]
                    return [[m["name"], m["model_path"], m.get("index_path", "")] for m in models]

                model_table = gr.Dataframe(
                    headers=[
                        t("model_name", "ui"),
                        t("model_path", "ui"),
                        t("index_path", "ui"),
                    ],
                    value=get_model_table(),
                    interactive=False
                )

                refresh_table_btn = gr.Button(
                    f"🔄 {t('refresh_models', 'ui')}",
                    variant="secondary"
                )

                refresh_table_btn.click(
                    fn=get_model_table,
                    outputs=[model_table]
                )

            # ===== 歌曲翻唱标签页 =====
            with gr.Tab(t("cover", "tabs")):
                gr.Markdown(f"### 🎵 {t('song_cover', 'cover')}")
                gr.Markdown(t("cover_usage", "ui"))

                with gr.Row():
                    # 左侧：输入和角色选择
                    with gr.Column(scale=1):
                        gr.Markdown(f"#### 📁 {t('upload_song', 'cover')}")
                        cover_input_audio = gr.Audio(
                            label=t("input_song", "cover"),
                            type="filepath"
                        )

                        gr.Markdown(f"#### 🎭 {t('select_character', 'cover')}")

                        downloaded_series = gr.Dropdown(
                            label=t("series_filter", "ui"),
                            choices=get_downloaded_character_series(),
                            value=_all_series_label(),
                            interactive=True
                        )

                        downloaded_keyword = gr.Textbox(
                            label=t("keyword_search", "ui"),
                            placeholder=t("keyword_placeholder", "ui"),
                            interactive=True
                        )

                        character_dropdown = gr.Dropdown(
                            label=t("character", "cover"),
                            choices=get_downloaded_character_choices(_all_series_label(), ""),
                            interactive=True,
                            info=t("character_choice_info", "ui")
                        )

                        character_details = gr.Markdown(
                            value=get_downloaded_character_details(None)
                        )

                        with gr.Row():
                            refresh_char_btn = gr.Button(
                                f"🔄 {t('refresh', 'conversion')}",
                                size="sm",
                                variant="secondary"
                            )

                        # 角色下载区域
                        with gr.Accordion(t("download_character", "cover"), open=False):
                            series_choices = [_all_series_label()] + get_available_character_series()
                            download_series = gr.Dropdown(
                                label=t("series_filter", "ui"),
                                choices=series_choices,
                                value=_all_series_label(),
                                interactive=True
                            )

                            download_keyword = gr.Textbox(
                                label=t("keyword_search", "ui"),
                                placeholder=t("keyword_placeholder", "ui"),
                                interactive=True
                            )

                            download_char_dropdown = gr.Dropdown(
                                label=t("select_to_download", "cover"),
                                choices=get_available_character_choices(_all_series_label(), ""),
                                interactive=True,
                                info=t("download_character_info", "ui")
                            )

                            download_char_details = gr.Markdown(
                                value=get_available_character_details(None)
                            )

                            download_char_btn = gr.Button(
                                f"⬇️ {t('download_selected_character', 'ui')}",
                                variant="primary"
                            )

                            download_all_series_btn = gr.Button(
                                f"⬇️ {t('download_series_all', 'ui')}",
                                variant="secondary"
                            )

                            download_all_btn = gr.Button(
                                f"⬇️ {t('download_all_characters', 'ui')}",
                                variant="secondary"
                            )

                            download_char_status = gr.Textbox(
                                label=t("download_status", "ui"),
                                interactive=False
                            )

                    # 右侧：参数设置
                    with gr.Column(scale=1):
                        gr.Markdown(f"#### ⚙️ {t('conversion_settings', 'cover')}")
                        cover_cfg = config.get("cover", {})

                        cover_pitch_shift = gr.Slider(
                            label=t("pitch_shift", "cover"),
                            minimum=-12,
                            maximum=12,
                            value=0,
                            step=1,
                            info=t("positive_pitch_info", "ui")
                        )

                        cover_index_rate = gr.Slider(
                            label=t("index_rate", "cover"),
                            minimum=0,
                            maximum=100,
                            value=_to_int(
                                round(
                                    _to_float(
                                        cover_cfg.get("index_rate", config.get("index_rate", 0.35)),
                                        0.35,
                                    ) * 100
                                ),
                                35,
                            ),
                            step=5,
                            info=t("index_rate_info", "cover"),
                        )

                        cover_speaker_id = gr.Slider(
                            label=t("speaker_id", "cover"),
                            minimum=0,
                            maximum=255,
                            value=_to_int(cover_cfg.get("speaker_id", 0), 0),
                            step=1,
                            info=t("speaker_id_info", "cover"),
                        )

                        gr.Markdown(f"#### 🎚️ {t('mix_settings', 'cover')}")
                        cover_karaoke = gr.Checkbox(
                            label=t("karaoke_separation", "cover"),
                            value=bool(cover_cfg.get("karaoke_separation", True)),
                            info=t("karaoke_separation_info", "cover")
                        )
                        cover_karaoke_merge_backing = gr.Checkbox(
                            label=t("karaoke_merge_backing", "cover"),
                            value=bool(
                                cover_cfg.get(
                                    "karaoke_merge_backing_into_accompaniment",
                                    True
                                )
                            ),
                            info=t("karaoke_merge_backing_info", "cover")
                        )

                        vc_label_to_value, vc_value_to_label = get_vc_preprocess_option_maps()
                        source_label_to_value, source_value_to_label = get_source_constraint_option_maps()
                        pipeline_label_to_value, pipeline_value_to_label = get_vc_pipeline_mode_option_maps()

                        cover_vc_preprocess_mode = gr.Dropdown(
                            label=t("vc_preprocess_mode", "cover"),
                            choices=list(vc_label_to_value.keys()),
                            value=vc_value_to_label.get(str(cover_cfg.get("vc_preprocess_mode", "auto")), list(vc_label_to_value.keys())[0]),
                            info=t("vc_preprocess_mode_info", "cover"),
                        )

                        cover_source_constraint_mode = gr.Dropdown(
                            label=t("source_constraint_mode", "cover"),
                            choices=list(source_label_to_value.keys()),
                            value=source_value_to_label.get(str(cover_cfg.get("source_constraint_mode", "auto")), list(source_label_to_value.keys())[0]),
                            info=t("source_constraint_mode_info", "cover"),
                        )
                        cover_vc_pipeline_mode = gr.Dropdown(
                            label=t("vc_pipeline_mode", "cover"),
                            choices=list(pipeline_label_to_value.keys()),
                            value=pipeline_value_to_label.get(str(cover_cfg.get("vc_pipeline_mode", "current")), list(pipeline_label_to_value.keys())[0]),
                            info=t("vc_pipeline_mode_info", "cover"),
                        )
                        cover_singing_repair = gr.Checkbox(
                            label=t("singing_repair", "cover"),
                            value=bool(cover_cfg.get("singing_repair", False)),
                            info=t("singing_repair_info", "cover"),
                            visible=str(cover_cfg.get("vc_pipeline_mode", "current")).strip().lower() == "official",
                        )

                        cover_vc_route_status = gr.Textbox(
                            label=t("vc_preprocess_status", "cover"),
                            value=get_cover_vc_route_status(
                                cover_cfg.get("vc_preprocess_mode", "auto"),
                                cover_cfg.get("vc_pipeline_mode", "current"),
                            ),
                            info=t("vc_preprocess_status_info", "cover"),
                            interactive=False,
                            lines=3,
                            elem_classes=["status-box"]
                        )

                        mix_presets, default_mix_preset = get_cover_mix_presets()
                        default_mix = mix_presets[default_mix_preset]

                        cover_mix_preset = gr.Dropdown(
                            label=t("mix_preset", "cover"),
                            choices=list(mix_presets.keys()),
                            value=default_mix_preset,
                            info=t("mix_preset_info", "cover"),
                            interactive=True
                        )

                        cover_vocals_volume = gr.Slider(
                            label=t("vocals_volume", "cover"),
                            minimum=0,
                            maximum=200,
                            value=default_mix["vocals_volume"],
                            step=5,
                            info=t("normal_volume_info", "ui")
                        )

                        cover_accompaniment_volume = gr.Slider(
                            label=t("accompaniment_volume", "cover"),
                            minimum=0,
                            maximum=200,
                            value=default_mix["accompaniment_volume"],
                            step=5,
                            info=t("normal_volume_info", "ui")
                        )

                        cover_reverb = gr.Slider(
                            label=t("vocals_reverb", "cover"),
                            minimum=0,
                            maximum=100,
                            value=default_mix["reverb"],
                            step=5,
                            info=t("reverb_info", "ui")
                        )

                        cover_rms_mix_rate = gr.Slider(
                            label=t("rms_mix_rate", "cover"),
                            minimum=0,
                            maximum=100,
                            value=_to_int(
                                round(
                                    _to_float(
                                        cover_cfg.get(
                                            "rms_mix_rate",
                                            config.get("rms_mix_rate", 0.15),
                                        ),
                                        0.15,
                                    ) * 100
                                ),
                                15,
                            ),
                            step=5,
                            info=t("rms_mix_rate_info", "cover"),
                        )

                        cover_backing_mix = gr.Slider(
                            label=t("backing_mix", "cover"),
                            minimum=0,
                            maximum=100,
                            value=_to_int(
                                round(_to_float(cover_cfg.get("backing_mix", 0.0), 0.0) * 100),
                                0,
                            ),
                            step=5,
                            info=t("backing_mix_info", "cover"),
                        )

                # 开始按钮
                cover_btn = gr.Button(
                    f"🚀 {t('start_cover', 'cover')}",
                    variant="primary",
                    size="lg"
                )

                # 状态显示
                cover_status = gr.Textbox(
                    label=t("progress", "cover"),
                    interactive=False,
                    elem_classes=["status-box"]
                )

                # 输出区域
                gr.Markdown(f"#### 🎵 {t('results', 'cover')}")

                with gr.Row():
                    cover_output = gr.Audio(
                        label=t("final_cover", "cover"),
                        type="filepath",
                        interactive=False
                    )

                with gr.Row():
                    cover_converted_vocals_output = gr.Audio(
                        label=t("converted_vocals", "cover"),
                        type="filepath",
                        interactive=False
                    )
                    cover_original_vocals_output = gr.Audio(
                        label=t("original_vocals", "cover"),
                        type="filepath",
                        interactive=False
                    )

                with gr.Row():
                    cover_lead_vocals_output = gr.Audio(
                        label=t("lead_vocals", "cover"),
                        type="filepath",
                        interactive=False
                    )
                    cover_backing_vocals_output = gr.Audio(
                        label=t("backing_vocals", "cover"),
                        type="filepath",
                        interactive=False
                    )

                with gr.Row():
                    cover_accompaniment_output = gr.Audio(
                        label=t("accompaniment", "cover"),
                        type="filepath",
                        interactive=False
                    )

                # 事件绑定
                refresh_char_btn.click(
                    fn=refresh_downloaded_controls,
                    inputs=[downloaded_series, downloaded_keyword],
                    outputs=[downloaded_series, character_dropdown]
                )
                refresh_char_btn.click(
                    fn=lambda: get_downloaded_character_details(None),
                    outputs=[character_details]
                )

                downloaded_series.change(
                    fn=update_downloaded_choices,
                    inputs=[downloaded_series, downloaded_keyword],
                    outputs=[character_dropdown]
                )
                downloaded_series.change(
                    fn=lambda: get_downloaded_character_details(None),
                    outputs=[character_details]
                )

                downloaded_keyword.change(
                    fn=update_downloaded_choices,
                    inputs=[downloaded_series, downloaded_keyword],
                    outputs=[character_dropdown]
                )
                downloaded_keyword.change(
                    fn=lambda: get_downloaded_character_details(None),
                    outputs=[character_details]
                )

                character_dropdown.change(
                    fn=get_downloaded_character_details,
                    inputs=[character_dropdown],
                    outputs=[character_details]
                )

                download_series.change(
                    fn=update_download_choices,
                    inputs=[download_series, download_keyword],
                    outputs=[download_char_dropdown]
                )
                download_series.change(
                    fn=lambda: get_available_character_details(None),
                    outputs=[download_char_details]
                )

                download_keyword.change(
                    fn=update_download_choices,
                    inputs=[download_series, download_keyword],
                    outputs=[download_char_dropdown]
                )
                download_keyword.change(
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

                cover_mix_preset.change(
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
                    fn=get_cover_vc_route_status,
                    inputs=[cover_vc_preprocess_mode, cover_vc_pipeline_mode],
                    outputs=[cover_vc_route_status]
                )

                mature_deecho_download_btn.click(
                    fn=download_mature_deecho_models_ui,
                    outputs=[mature_deecho_status]
                )
                mature_deecho_download_btn.click(
                    fn=get_cover_vc_route_status,
                    inputs=[cover_vc_preprocess_mode, cover_vc_pipeline_mode],
                    outputs=[cover_vc_route_status]
                )

                cover_vc_preprocess_mode.change(
                    fn=get_cover_vc_route_status,
                    inputs=[cover_vc_preprocess_mode, cover_vc_pipeline_mode],
                    outputs=[cover_vc_route_status]
                )

                cover_vc_pipeline_mode.change(
                    fn=get_cover_vc_route_status,
                    inputs=[cover_vc_preprocess_mode, cover_vc_pipeline_mode],
                    outputs=[cover_vc_route_status]
                )
                cover_vc_pipeline_mode.change(
                    fn=update_singing_repair_visibility,
                    inputs=[cover_vc_pipeline_mode],
                    outputs=[cover_singing_repair]
                )

                cover_btn.click(
                    fn=process_cover,
                    inputs=[
                        cover_input_audio,
                        character_dropdown,
                        cover_pitch_shift,
                        cover_index_rate,
                        cover_speaker_id,
                        cover_karaoke,
                        cover_karaoke_merge_backing,
                        cover_vc_preprocess_mode,
                        cover_source_constraint_mode,
                        cover_vc_pipeline_mode,
                        cover_singing_repair,
                        cover_vocals_volume,
                        cover_accompaniment_volume,
                        cover_reverb,
                        cover_rms_mix_rate,
                        cover_backing_mix,
                    ],
                    outputs=[
                        cover_output,
                        cover_converted_vocals_output,
                        cover_original_vocals_output,
                        cover_lead_vocals_output,
                        cover_backing_vocals_output,
                        cover_accompaniment_output,
                        cover_status
                    ]
                )

            # ===== 设置标签页 =====
            with gr.Tab(t("settings", "tabs")):
                gr.Markdown(f"### 💻 {t('device_info', 'settings')}")

                device_info = gr.Textbox(
                    label=t("current_device", "settings"),
                    value=get_device_info(),
                    interactive=False,
                    lines=5,
                    elem_classes=["status-box"]
                )

                refresh_device_btn = gr.Button(
                    f"🔄 {t('refresh_device', 'settings')}",
                    variant="secondary"
                )

                refresh_device_btn.click(
                    fn=get_device_info,
                    outputs=[device_info]
                )

                gr.Markdown("---")

                gr.Markdown(f'### ⚙️ {t("runtime_settings", "settings")}')

                def _build_device_choices():
                    from lib.device import _has_xpu, _has_directml, _has_mps, _is_rocm
                    import torch
                    choices = []
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
                    return choices

                device_radio = gr.Radio(
                    label=t("compute_device", "settings"),
                    choices=_build_device_choices(),
                    value=config.get("device", "cuda")
                )

                save_settings_btn = gr.Button(
                    f"💾 {t('save_settings', 'settings')}",
                    variant="primary"
                )

                settings_status = gr.Textbox(
                    label=t("status", "settings"),
                    interactive=False
                )

                def save_settings(device):
                    global config
                    config["device"] = device

                    config_path = ROOT_DIR / "configs" / "config.json"
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4, ensure_ascii=False)

                    return t("settings_saved_restart", "settings")

                save_settings_btn.click(
                    fn=save_settings,
                    inputs=[device_radio],
                    outputs=[settings_status]
                )

                gr.Markdown("---")

                gr.Markdown(f"### ℹ️ {t('about', 'settings')}")
                gr.Markdown(t("about_body", "settings"))

                gr.Markdown("---")

                gr.Markdown(t("model_sources", "settings"))

    return app


def _patch_gradio_file_download(blocks):
    """
    Patch Gradio v3 的 /file= 路由，为文件添加 Content-Disposition header，
    使浏览器下载时使用干净的文件名而非完整路径。
    """
    try:
        from starlette.responses import FileResponse
        from urllib.parse import quote
        import fastapi

        def _clean_download_name(response: FileResponse, path_or_url: str) -> str:
            candidates = [
                getattr(response, "filename", None),
                getattr(response, "path", None),
                path_or_url,
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                name = Path(str(candidate)).name
                if not name:
                    continue
                name = re.sub(
                    r"^[A-Za-z]__.*?_gradio_[0-9a-f]{8,}_",
                    "",
                    name,
                    flags=re.IGNORECASE,
                )
                if name:
                    return name
            return "download"

        fastapi_app = getattr(blocks, "server_app", None)
        if fastapi_app is None:
            return

        for route in fastapi_app.routes:
            if hasattr(route, "path") and route.path == "/file={path_or_url:path}":
                original_endpoint = route.endpoint

                async def patched_file(
                    path_or_url: str,
                    request: fastapi.Request,
                    _orig=original_endpoint,
                ):
                    response = await _orig(path_or_url, request=request)
                    if isinstance(response, FileResponse) and "content-disposition" not in response.headers:
                        basename = _clean_download_name(response, path_or_url)
                        encoded = quote(basename)
                        if encoded != basename:
                            cd = f"inline; filename*=utf-8''{encoded}"
                        else:
                            cd = f'inline; filename="{basename}"'
                        response.headers["content-disposition"] = cd
                    return response

                route.endpoint = patched_file
                break
    except Exception as e:
        log.warning(f"Patch Gradio file download failed: {e}")


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """启动 Gradio 界面"""
    app = create_ui()
    app.queue()  # 启用队列以支持进度跟踪
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        prevent_thread_lock=True
    )
    _patch_gradio_file_download(app)
    app.block_thread()


if __name__ == "__main__":
    launch()
