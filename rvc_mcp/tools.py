# -*- coding: utf-8 -*-
"""
MCP 工具函数 - 本地人声音色转换（不包含歌曲分离或混音）
"""
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent


def get_config() -> dict:
    """获取配置"""
    config_path = ROOT_DIR / "configs" / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            from configs.schema import validate_config
            validate_config(config)
            return _normalize_config(config)
    return {}


def _normalize_config(config: dict) -> dict:
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
    if "temp_dir" not in config and "temp" in paths:
        config["temp_dir"] = paths["temp"]

    return config


def list_models() -> List[Dict[str, Any]]:
    """
    列出所有可用的语音模型

    Returns:
        List[Dict]: 模型列表，每个模型包含 name, model_path, index_path
    """
    from infer.pipeline import list_voice_models

    config = get_config()
    weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")

    models = list_voice_models(str(weights_dir))
    for model in models:
        model["id"] = Path(model["model_path"]).relative_to(weights_dir).as_posix()
    return models


def convert_voice(
    input_path: str,
    output_path: str,
    model_name: str,
    pitch_shift: int = 0,
    index_ratio: float = 0.5,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    speaker_id: int = 0,
    f0_method: str = "rmvpe",
) -> Dict[str, Any]:
    """使用与 Web 相同的固定版官方入口进行人声音色转换。"""
    try:
        from infer.official_adapter import convert_vocals_official_upstream
        config = get_config()
        models = list_models()
        matches = [model for model in models if model["id"] == model_name]
        if not matches:
            matches = [model for model in models if model["name"] == model_name]
        if len(matches) != 1:
            raise ValueError(f"模型名称必须唯一匹配，实际匹配 {len(matches)} 个：{model_name}；请使用 list_voice_models 返回的 id")
        model = matches[0]
        result = convert_vocals_official_upstream(
            vocals_path=input_path, output_path=output_path,
            model_path=model["model_path"], index_path=model.get("index_path"),
            f0_method=f0_method, pitch_shift=pitch_shift, index_rate=index_ratio,
            rms_mix_rate=rms_mix_rate, protect=protect, speaker_id=speaker_id,
            device=config.get("device", "auto"),
        )
        return {"success": True, "output_path": result, "error": None}
    except Exception as exc:
        return {"success": False, "output_path": None, "error": str(exc)}


def download_model(model_name: str = None) -> Dict[str, Any]:
    """
    下载模型

    Args:
        model_name: 模型名称，为 None 时下载所有必需模型

    Returns:
        Dict: 下载结果
    """
    try:
        from tools.download_models import (
            download_model as dl_model,
            download_required_models,
            MODELS
        )

        if model_name:
            if model_name not in MODELS:
                return {
                    "success": False,
                    "error": f"未知模型: {model_name}"
                }
            success = dl_model(model_name)
        else:
            success = download_required_models()

        return {
            "success": success,
            "error": None if success else "下载失败"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_model_status() -> Dict[str, bool]:
    """
    获取模型下载状态

    Returns:
        Dict: 模型名称 -> 是否已下载
    """
    from tools.download_models import check_all_models, check_default_separator_models, check_upstream_rvc_tree

    status = check_all_models()
    status["official_vc_source_and_transformers_hubert"] = check_upstream_rvc_tree()
    status.update(
        {
            f"default_separator:{name}": exists
            for name, exists in check_default_separator_models().items()
        }
    )
    return status
