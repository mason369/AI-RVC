# -*- coding: utf-8 -*-
"""
MCP 工具函数 - 提供语音转换功能的工具接口
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
            return json.load(f)
    return {}


def list_models() -> List[Dict[str, Any]]:
    """
    列出所有可用的语音模型

    Returns:
        List[Dict]: 模型列表，每个模型包含 name, model_path, index_path
    """
    from infer.pipeline import list_voice_models

    config = get_config()
    weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")

    return list_voice_models(str(weights_dir))


def convert_voice(
    input_path: str,
    output_path: str,
    model_name: str,
    pitch_shift: float = 0,
    index_ratio: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33
) -> Dict[str, Any]:
    """
    执行语音转换

    Args:
        input_path: 输入音频文件路径
        output_path: 输出音频文件路径
        model_name: 模型名称
        pitch_shift: 音调偏移 (半音)
        index_ratio: 索引混合比率
        filter_radius: 中值滤波半径
        rms_mix_rate: 响度混合比率
        protect: 保护系数

    Returns:
        Dict: 转换结果，包含 success, output_path, error
    """
    try:
        from infer.pipeline import VoiceConversionPipeline, list_voice_models

        config = get_config()

        # 查找模型
        weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")
        models = list_voice_models(str(weights_dir))

        model_info = None
        for m in models:
            if m["name"] == model_name:
                model_info = m
                break

        if model_info is None:
            return {
                "success": False,
                "output_path": None,
                "error": f"模型不存在: {model_name}"
            }

        # 初始化管道
        device = config.get("device", "cuda")
        pipeline = VoiceConversionPipeline(device=device)

        # 加载 HuBERT
        hubert_path = ROOT_DIR / config.get("hubert_path", "assets/hubert/hubert_base.pt")
        if not hubert_path.exists():
            return {
                "success": False,
                "output_path": None,
                "error": "HuBERT 模型不存在，请先下载基础模型"
            }
        pipeline.load_hubert(str(hubert_path))

        # 加载 F0 提取器
        rmvpe_path = ROOT_DIR / config.get("rmvpe_path", "assets/rmvpe/rmvpe.pt")
        if not rmvpe_path.exists():
            return {
                "success": False,
                "output_path": None,
                "error": "RMVPE 模型不存在，请先下载基础模型"
            }
        pipeline.load_f0_extractor("rmvpe", str(rmvpe_path))

        # 加载语音模型
        pipeline.load_voice_model(model_info["model_path"])

        # 加载索引
        if model_info.get("index_path"):
            pipeline.load_index(model_info["index_path"])

        # 执行转换
        result_path = pipeline.convert(
            audio_path=input_path,
            output_path=output_path,
            pitch_shift=pitch_shift,
            index_ratio=index_ratio,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect
        )

        return {
            "success": True,
            "output_path": result_path,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "output_path": None,
            "error": str(e)
        }


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
    from tools.download_models import check_all_models
    return check_all_models()
