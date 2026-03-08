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
from typing import Optional, Tuple, Dict

from lib.logger import log

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 加载语言包
def load_i18n(lang: str = "zh_CN") -> dict:
    """加载语言包"""
    i18n_path = ROOT_DIR / "i18n" / f"{lang}.json"
    if i18n_path.exists():
        with open(i18n_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 加载配置
def load_config() -> dict:
    """加载配置"""
    config_path = ROOT_DIR / "configs" / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
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

# 全局变量
i18n = load_i18n()
config = normalize_config(load_config())
pipeline = None


def t(key: str, section: str = None) -> str:
    """获取翻译文本"""
    if section:
        return i18n.get(section, {}).get(key, key)
    return i18n.get(key, key)


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
        t("vc_preprocess_direct", "cover"): "direct",
        t("vc_preprocess_uvr_deecho", "cover"): "uvr_deecho",
        t("vc_preprocess_legacy", "cover"): "legacy",
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
            return "下载过程中出现错误，请检查网络连接"
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
    series = sorted({c.get("series", "未知") for c in characters})
    return ["全部"] + series


def get_available_character_list() -> list:
    """获取可下载的角色列表"""
    from tools.character_models import list_available_characters
    return list_available_characters()


def get_available_character_series() -> list:
    """获取可用系列列表"""
    from tools.character_models import list_available_series
    return list_available_series()


def format_character_label(char_info: dict) -> str:
    """格式化角色展示名称：【语言】角色名(中/英/日) · 出处 · 内部名"""
    display = char_info.get("display") or char_info.get("description") or char_info.get("name", "")
    source = char_info.get("source", "未知")
    name = char_info.get("name", "")
    lang_tag = get_character_language_tag(char_info)
    return f"【{lang_tag}】{display}（出自：{source}）[{name}]"


def get_character_language_tag(char_info: dict) -> str:
    """推断语言类型，用于下拉前缀标签"""
    lang = char_info.get("lang")
    if lang:
        return lang
    text = " ".join(
        str(char_info.get(k, "")) for k in ("display", "description", "name")
    ).lower()
    if "韩" in text or "kr" in text or "korean" in text:
        return "韩文"
    if "日" in text or "jp" in text or "japanese" in text:
        return "日文"
    if "中" in text or "cn" in text or "chinese" in text:
        return "中文"
    if "en" in text or "english" in text:
        return "英文"

    source = char_info.get("source", "")
    if source.startswith("Love Live!") or "ホロライブ" in source or "偶像大师" in source or "赛马娘" in source:
        return "日文"
    if "原神" in source or "崩坏" in source or "明日方舟" in source or "碧蓝航线" in source:
        return "中文"
    if "VOCALOID" in source or "Project SEKAI" in source:
        return "日文"
    if "Hololive" in source:
        return "日文"
    if "蔚蓝档案" in source or "绝区零" in source:
        return "日文"
    return "中文"


def get_downloaded_character_choices(series: str = "全部", keyword: str = "") -> list:
    """获取已下载角色的下拉选项"""
    chars = get_downloaded_character_list()
    if series and series != "全部":
        chars = [c for c in chars if c.get("series") == series]
    if keyword:
        kw = keyword.strip().lower()
        if kw:
            chars = [
                c for c in chars
                if kw in c.get("name", "").lower()
                or kw in c.get("display", "").lower()
                or kw in c.get("source", "").lower()
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
    if series and series != "全部":
        chars = [c for c in chars if c.get("series") == series]
    if keyword:
        kw = keyword.strip().lower()
        if kw:
            chars = [
                c for c in chars
                if kw in c.get("name", "").lower()
                or kw in c.get("display", "").lower()
                or kw in c.get("source", "").lower()
            ]
    return [(format_character_label(c), c["name"]) for c in chars]


def _refresh_downloaded_updates(series: str, keyword: str) -> Tuple[Dict, Dict]:
    series_choices = get_downloaded_character_series()
    if series not in series_choices:
        series = "全部"
    return (
        gr.update(choices=series_choices, value=series),
        gr.update(choices=get_downloaded_character_choices(series, keyword))
    )


def download_character(name: str, selected_series: str = "全部", keyword: str = "") -> Tuple[str, Dict, Dict]:
    """下载角色模型"""
    from tools.character_models import download_character_model

    if not name:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return "请选择要下载的角色", choices_update, series_update

    try:
        success = download_character_model(name)
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        if success:
            return (
                f"✅ {name} 模型下载完成",
                choices_update,
                series_update
            )
        else:
            return (
                f"❌ {name} 模型下载失败",
                choices_update,
                series_update
            )
    except Exception as e:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return (
            f"❌ 下载失败: {str(e)}",
            choices_update,
            series_update
        )


def download_all_characters(series: str = "全部", selected_series: str = "全部", keyword: str = "") -> Tuple[str, Dict, Dict]:
    """批量下载角色模型"""
    from tools.character_models import download_all_character_models

    try:
        result = download_all_character_models(series=series)
        ok = result.get("success", [])
        failed = result.get("failed", [])
        status = f"✅ 完成: 成功 {len(ok)} 个"
        if failed:
            status += f"，失败 {len(failed)} 个: {', '.join(failed)}"
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return status, choices_update, series_update
    except Exception as e:
        series_update, choices_update = _refresh_downloaded_updates(selected_series, keyword)
        return f"❌ 批量下载失败: {str(e)}", choices_update, series_update


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
        return *_none6, "请上传歌曲文件"

    if not character_name:
        return *_none6, "请选择角色"

    try:
        from tools.character_models import get_character_model_path
        from infer.cover_pipeline import get_cover_pipeline

        # 获取角色模型路径
        resolved_name = resolve_character_name(character_name)
        model_info = get_character_model_path(resolved_name)
        if model_info is None:
            return *_none6, f"角色模型不存在: {resolved_name}"

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
        karaoke_model = cover_cfg.get("karaoke_model", "mel_band_roformer_karaoke_gabox.ckpt")
        default_vc_preprocess_mode = str(cover_cfg.get("vc_preprocess_mode", "auto"))
        default_source_constraint_mode = str(cover_cfg.get("source_constraint_mode", "auto"))
        vc_label_to_value, vc_value_to_label = get_vc_preprocess_option_maps()
        source_label_to_value, source_value_to_label = get_source_constraint_option_maps()

        vc_preprocess_mode = vc_label_to_value.get(str(vc_preprocess_mode), str(vc_preprocess_mode or default_vc_preprocess_mode).strip().lower())
        if vc_preprocess_mode not in {"auto", "direct", "uvr_deecho", "legacy"}:
            vc_preprocess_mode = default_vc_preprocess_mode
        source_constraint_mode = source_label_to_value.get(str(source_constraint_mode), str(source_constraint_mode or default_source_constraint_mode).strip().lower())
        if source_constraint_mode not in {"auto", "off", "on"}:
            source_constraint_mode = default_source_constraint_mode

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
            output_dir=str(output_dir),
            model_display_name=resolved_name,
            progress_callback=progress_callback
        )

        status_msg = "✅ 翻唱完成!"
        if result.get("all_files_dir"):
            status_msg += f"\n全部文件目录: {result['all_files_dir']}"

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
        return None, None, None, None, None, None, f"❌ 处理失败: {error_msg}"


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
    lines.append(f"PyTorch 版本: {torch.__version__}")

    info = _get_info()
    lines.append(f"可用后端: {', '.join(info['backends'])}")

    for dev in info["devices"]:
        mem = f"{dev['total_memory_gb']} GB" if dev.get("total_memory_gb") else "N/A"
        lines.append(f"GPU: {dev['name']} ({dev['backend']}) - 显存: {mem}")

    if torch.cuda.is_available():
        ver = torch.version.hip if _is_rocm() else torch.version.cuda
        label = "ROCm" if _is_rocm() else "CUDA"
        lines.append(f"{label} 版本: {ver}")

    if not info["devices"]:
        lines.append("未检测到 GPU，将使用 CPU")

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

/* Gradio 3.x 特定选择器样式 */
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

/* Gradio 4.x Dataframe 表头 */
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

/* Gradio 4.x 音频波形 */
.waveform-container,
.audio-container {
    --waveform-color: #ff9800 !important;
    --progress-color: #ff9800 !important;
}
"""


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    with gr.Blocks(
        title=i18n.get("app_title", "RVC AI 翻唱"),
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
            f"# 🎤 {i18n.get('app_title', 'RVC AI 翻唱')}",
            elem_classes=["main-title"]
        )
        gr.Markdown(
            f"<center>{i18n.get('app_description', '基于 RVC v2 的 AI 翻唱系统')}</center>"
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

                gr.Markdown(f"### 🎤 {t('voice_models', 'models')}")
                gr.Markdown(t("voice_models_desc", "models"))

                def get_model_table():
                    from infer.pipeline import list_voice_models
                    weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")
                    models = list_voice_models(str(weights_dir))
                    if not models:
                        return [["(无模型)", "", ""]]
                    return [[m["name"], m["model_path"], m.get("index_path", "无")] for m in models]

                model_table = gr.Dataframe(
                    headers=["模型名称", "模型路径", "索引路径"],
                    value=get_model_table(),
                    interactive=False
                )

                refresh_table_btn = gr.Button(
                    f"🔄 刷新模型列表",
                    variant="secondary"
                )

                refresh_table_btn.click(
                    fn=get_model_table,
                    outputs=[model_table]
                )

            # ===== 歌曲翻唱标签页 =====
            with gr.Tab(t("cover", "tabs")):
                gr.Markdown(f"### 🎵 {t('song_cover', 'cover')}")
                gr.Markdown(
                    """
                    **一键 AI 翻唱**：上传歌曲 → 自动分离人声 → 转换音色 → 混合伴奏 → 输出翻唱

                    **使用步骤：**
                    1. 先下载角色模型（展开下方「下载角色模型」）
                    2. 上传歌曲文件（支持 MP3/WAV/FLAC）
                    3. 选择已下载的角色
                    4. 调整参数后点击「开始翻唱」

                    > ⚠️ 首次运行会自动下载 Mel-Band Roformer 人声分离模型（约 200MB），请耐心等待
                    """
                )

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
                            label="作品/分类",
                            choices=get_downloaded_character_series(),
                            value="全部",
                            interactive=True
                        )

                        downloaded_keyword = gr.Textbox(
                            label="关键词搜索",
                            placeholder="输入角色名/作品名",
                            interactive=True
                        )

                        character_dropdown = gr.Dropdown(
                            label="选择角色",
                            choices=get_downloaded_character_choices("全部", ""),
                            interactive=True,
                            info="括号中的信息为模型训练参数：epochs=训练轮数(越大通常越成熟)，数字+k=训练采样率(如40k=40000Hz)"
                        )

                        with gr.Row():
                            refresh_char_btn = gr.Button(
                                "🔄 刷新",
                                size="sm",
                                variant="secondary"
                            )

                        # 角色下载区域
                        with gr.Accordion("下载角色模型", open=False):
                            series_choices = ["全部"] + get_available_character_series()
                            download_series = gr.Dropdown(
                                label="作品/分类",
                                choices=series_choices,
                                value="全部",
                                interactive=True
                            )

                            download_keyword = gr.Textbox(
                                label="关键词搜索",
                                placeholder="输入角色名/作品名",
                                interactive=True
                            )

                            download_char_dropdown = gr.Dropdown(
                                label="选择角色",
                                choices=get_available_character_choices("全部", ""),
                                interactive=True
                            )

                            download_char_btn = gr.Button(
                                "⬇️ 下载选中角色",
                                variant="primary"
                            )

                            download_all_series_btn = gr.Button(
                                "⬇️ 下载该分类全部",
                                variant="secondary"
                            )

                            download_all_btn = gr.Button(
                                "⬇️ 下载全部角色模型",
                                variant="secondary"
                            )

                            download_char_status = gr.Textbox(
                                label="下载状态",
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
                            info="正数升调，负数降调"
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
                            info="100% 为原始音量"
                        )

                        cover_accompaniment_volume = gr.Slider(
                            label=t("accompaniment_volume", "cover"),
                            minimum=0,
                            maximum=200,
                            value=default_mix["accompaniment_volume"],
                            step=5,
                            info="100% 为原始音量"
                        )

                        cover_reverb = gr.Slider(
                            label=t("vocals_reverb", "cover"),
                            minimum=0,
                            maximum=100,
                            value=default_mix["reverb"],
                            step=5,
                            info="为人声添加混响效果"
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

                downloaded_series.change(
                    fn=update_downloaded_choices,
                    inputs=[downloaded_series, downloaded_keyword],
                    outputs=[character_dropdown]
                )

                downloaded_keyword.change(
                    fn=update_downloaded_choices,
                    inputs=[downloaded_series, downloaded_keyword],
                    outputs=[character_dropdown]
                )

                download_series.change(
                    fn=update_download_choices,
                    inputs=[download_series, download_keyword],
                    outputs=[download_char_dropdown]
                )

                download_keyword.change(
                    fn=update_download_choices,
                    inputs=[download_series, download_keyword],
                    outputs=[download_char_dropdown]
                )

                download_char_btn.click(
                    fn=download_character,
                    inputs=[download_char_dropdown, downloaded_series, downloaded_keyword],
                    outputs=[download_char_status, character_dropdown, downloaded_series]
                )

                download_all_series_btn.click(
                    fn=download_all_characters,
                    inputs=[download_series, downloaded_series, downloaded_keyword],
                    outputs=[download_char_status, character_dropdown, downloaded_series]
                )

                download_all_btn.click(
                    fn=lambda series, keyword: download_all_characters("全部", series, keyword),
                    inputs=[downloaded_series, downloaded_keyword],
                    outputs=[download_char_status, character_dropdown, downloaded_series]
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

                gr.Markdown(f"### ⚙️ 运行设置")

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
                    choices.append(("CPU (较慢)", "cpu"))
                    return choices

                device_radio = gr.Radio(
                    label="计算设备",
                    choices=_build_device_choices(),
                    value=config.get("device", "cuda")
                )

                save_settings_btn = gr.Button(
                    "💾 保存设置",
                    variant="primary"
                )

                settings_status = gr.Textbox(
                    label="状态",
                    interactive=False
                )

                def save_settings(device):
                    global config
                    config["device"] = device

                    config_path = ROOT_DIR / "configs" / "config.json"
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4, ensure_ascii=False)

                    return "✅ 设置已保存，重启后生效"

                save_settings_btn.click(
                    fn=save_settings,
                    inputs=[device_radio],
                    outputs=[settings_status]
                )

                gr.Markdown("---")

                gr.Markdown(f"### ℹ️ {t('about', 'settings')}")
                gr.Markdown(
                    """
                    **RVC AI 翻唱系统**

                    - 基于 RVC v2 + Mel-Band Roformer
                    - 使用 RMVPE 进行高质量 F0 提取
                    - 支持 CUDA GPU 加速

                    [GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
                    """
                )

                gr.Markdown("---")

                gr.Markdown(
                    """
                    ### 📥 角色模型来源

                    以下是本项目角色模型的 HuggingFace 仓库来源，你也可以手动下载模型后放入 `assets/weights/characters/<角色名>/` 目录使用：

                    **Love Live! 系列**
                    - [trioskosmos/rvc_models](https://huggingface.co/trioskosmos/rvc_models) — μ's / Aqours / 虹咲 / Liella! 多角色
                    - [Icchan/LoveLive](https://huggingface.co/Icchan/LoveLive) — 千歌、梨子、绘里、曜
                    - [0xMifune/LoveLive](https://huggingface.co/0xMifune/LoveLive) — 虹咲 / Liella! / 莲之空
                    - [Swordsmagus/Love-Live-RVC](https://huggingface.co/Swordsmagus/Love-Live-RVC) — 花丸、雪菜、小鸟、A-RISE 等
                    - [Zurakichi/RVC](https://huggingface.co/Zurakichi/RVC) — 妮可、彼方、雪菜、花丸
                    - [Phos252/RVCmodels](https://huggingface.co/Phos252/RVCmodels) — 涩谷香音
                    - [ChocoKat/Mari_Ohara](https://huggingface.co/ChocoKat/Mari_Ohara) — 小原鞠莉
                    - [HarunaKasuga/YoshikoTsushima](https://huggingface.co/HarunaKasuga/YoshikoTsushima) — 津岛善子
                    - [thebuddyadrian/RVC_Models](https://huggingface.co/thebuddyadrian/RVC_Models) — 鹿角姐妹

                    **原神 / 崩坏 / 绝区零 (米哈游)**
                    - [makiligon/RVC-Models](https://huggingface.co/makiligon/RVC-Models) — 芙宁娜、绫华、芙卡洛斯
                    - [kohaku12/RVC-MODELS](https://huggingface.co/kohaku12/RVC-MODELS) — 纳西妲、黑塔、流萤、停云、星见雅 等
                    - [jarari/RVC-v2](https://huggingface.co/jarari/RVC-v2) — 芙宁娜(韩语)、银狼(韩语)
                    - [mrmocciai/genshin-impact](https://huggingface.co/mrmocciai/genshin-impact) — 原神 50+ 角色（需手动下载）

                    **VOCALOID**
                    - [javinfamous/infamous_miku_v2](https://huggingface.co/javinfamous/infamous_miku_v2) — 初音未来 (1000 epochs)

                    **Hololive / VTuber**
                    - [megaaziib/my-rvc-models-collection](https://huggingface.co/megaaziib/my-rvc-models-collection) — 佩克拉、樱巫女、大空昴、Kobo、Kaela 等
                    - [Kit-Lemonfoot/kitlemonfoot_rvc_models](https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models) — Hololive JP/EN 多角色

                    **偶像大师 / 赛马娘**
                    - [trioskosmos/rvc_models](https://huggingface.co/trioskosmos/rvc_models) — 神崎兰子、梦见莉亚梦
                    - [makiligon/RVC-Models](https://huggingface.co/makiligon/RVC-Models) — 四条贵音、米浴

                    **Project SEKAI**
                    - [kohaku12/RVC-MODELS](https://huggingface.co/kohaku12/RVC-MODELS) — 草薙宁宁

                    > 💡 手动下载后，将 `.pth` 和 `.index` 文件放入 `assets/weights/characters/<角色名>/` 目录，刷新即可使用。
                    """
                )

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
