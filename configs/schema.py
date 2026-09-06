"""应用配置白名单；未知或已移除的参数必须显式报错。"""
from __future__ import annotations

import re
from infer.contracts import number, integer

TOP_LEVEL_KEYS = {"language", "device", "weights_dir", "output_dir", "cover"}
COVER_KEYS = {
    "separator", "roformer_model", "karaoke_separation", "karaoke_model",
    "karaoke_merge_backing_into_accompaniment", "uvr5_model", "uvr5_agg", "uvr5_format",
    "use_official", "demucs_model", "demucs_shifts", "demucs_overlap", "demucs_split",
    "f0_method", "index_rate", "rms_mix_rate", "protect", "speaker_id", "silence_gate",
    "silence_threshold_db", "silence_smoothing_ms", "silence_min_duration_ms",
    "default_vocals_volume", "default_accompaniment_volume", "default_reverb", "backing_mix",
    "source_constraint_mode", "vc_pipeline_mode",
}


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("应用配置必须是 JSON 对象")
    unknown = set(config) - TOP_LEVEL_KEYS
    cover = config.get("cover", {})
    if not isinstance(cover, dict):
        raise ValueError("cover 配置必须是 JSON 对象")
    unknown.update(f"cover.{key}" for key in set(cover) - COVER_KEYS)
    if unknown:
        raise ValueError(f"存在未知或已移除的配置参数：{', '.join(sorted(unknown))}；请参阅 docs/有效性与模型兼容性.md")
    if config.get("language", "zh_CN") not in {"zh_CN", "en_US"}:
        raise ValueError("language 只支持 zh_CN 或 en_US")
    device = config.get("device", "auto")
    if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|mps|directml|(?:cuda|xpu|privateuseone)(?::\d+)?", device):
        raise ValueError(f"device 无效：{device!r}")
    for key in ("weights_dir", "output_dir"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"{key} 必须是非空路径字符串")
    choices = {
        "separator": {"roformer", "demucs", "uvr5"},
        "f0_method": {"rmvpe", "pm", "fcpe"} if cover.get("use_official", True) or cover.get("vc_pipeline_mode") == "official" else {"rmvpe", "pm", "harvest", "crepe"},
        "vc_pipeline_mode": {"current", "official"},
        "source_constraint_mode": {"auto", "off", "on"},
        "uvr5_format": {"wav"},
    }
    for key, values in choices.items():
        if key in cover and cover[key] not in values:
            raise ValueError(f"cover.{key} 无效：{cover[key]!r}；支持 {sorted(values)}")
    for key in ("karaoke_separation", "karaoke_merge_backing_into_accompaniment", "use_official", "demucs_split", "silence_gate"):
        if key in cover and type(cover[key]) is not bool:
            raise ValueError(f"cover.{key} 必须是布尔值")
    ranges = {"index_rate": (0, 1), "rms_mix_rate": (0, 1), "protect": (0, .5), "backing_mix": (0, 1),
              "demucs_overlap": (0, .99), "default_vocals_volume": (0, 200), "default_accompaniment_volume": (0, 200),
              "default_reverb": (0, 100), "silence_threshold_db": (-120, 0),
              "silence_smoothing_ms": (0, 10000), "silence_min_duration_ms": (0, 60000)}
    for key, limits in ranges.items():
        if key in cover:
            number(cover[key], f"cover.{key}", *limits)
    for key, limits in {"speaker_id": (0, 2**31 - 1), "uvr5_agg": (0, 20), "demucs_shifts": (0, 100)}.items():
        if key in cover:
            integer(cover[key], f"cover.{key}", *limits)
    return config
