# -*- coding: utf-8 -*-
"""
模型下载工具 - 自动从 Hugging Face 下载所需模型
"""
import os
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List

# 模型下载配置
MODELS = {
    "HP2_all_vocals.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth",
        "path": "assets/uvr5_weights/HP2_all_vocals.pth",
        "size_mb": 140,
        "description": "UVR5 HP2 vocal model"
    },
    "HP3_all_vocals.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP3_all_vocals.pth",
        "path": "assets/uvr5_weights/HP3_all_vocals.pth",
        "size_mb": 140,
        "description": "UVR5 HP3 vocal model"
    },
    "HP5_only_main_vocal.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth",
        "path": "assets/uvr5_weights/HP5_only_main_vocal.pth",
        "size_mb": 140,
        "description": "UVR5 HP5 main vocal model"
    },
    "VR-DeEchoAggressive.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoAggressive.pth",
        "path": "assets/uvr5_weights/VR-DeEchoAggressive.pth",
        "size_mb": 130,
        "description": "UVR5 de-echo aggressive"
    },
    "VR-DeEchoDeReverb.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoDeReverb.pth",
        "path": "assets/uvr5_weights/VR-DeEchoDeReverb.pth",
        "size_mb": 130,
        "description": "UVR5 de-echo + de-reverb"
    },
    "VR-DeEchoNormal.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth",
        "path": "assets/uvr5_weights/VR-DeEchoNormal.pth",
        "size_mb": 130,
        "description": "UVR5 de-echo normal"
    },
    "onnx_dereverb_By_FoxJoy/vocals.onnx": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
        "path": "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
        "size_mb": 50,
        "description": "UVR5 ONNX dereverb"
    },
    "hubert_base.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "path": "assets/hubert/hubert_base.pt",
        "size_mb": 189,
        "description": "HuBERT 特征提取模型"
    },
    "rmvpe.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "path": "assets/rmvpe/rmvpe.pt",
        "size_mb": 181,
        "description": "RMVPE 音高提取模型"
    },
    "f0G48k.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth",
        "path": "assets/pretrained_v2/f0G48k.pth",
        "size_mb": 55,
        "description": "48kHz 生成器预训练权重"
    },
    "f0D48k.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
        "path": "assets/pretrained_v2/f0D48k.pth",
        "size_mb": 55,
        "description": "48kHz 判别器预训练权重"
    },
    "f0G40k.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
        "path": "assets/pretrained_v2/f0G40k.pth",
        "size_mb": 55,
        "description": "40kHz 生成器预训练权重"
    },
    "f0D40k.pth": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
        "path": "assets/pretrained_v2/f0D40k.pth",
        "size_mb": 55,
        "description": "40kHz 判别器预训练权重"
    }
}

# 必需模型列表
REQUIRED_MODELS = ["hubert_base.pt", "rmvpe.pt", "HP2_all_vocals.pth"]

# Mature DeEcho / DeReverb models downloaded separately
MATURE_DEECHO_MODELS = [
    "VR-DeEchoDeReverb.pth",
    "onnx_dereverb_By_FoxJoy/vocals.onnx",
    "VR-DeEchoNormal.pth",
    "VR-DeEchoAggressive.pth",
]


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """
    下载文件，支持断点续传和进度显示

    Args:
        url: 下载链接
        dest_path: 目标路径
        desc: 进度条描述

    Returns:
        bool: 下载是否成功
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 检查已下载的部分
    resume_pos = 0
    if dest_path.exists():
        resume_pos = dest_path.stat().st_size

    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"
    if "huggingface.co" in url:
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        # 检查是否支持断点续传
        if response.status_code == 416:  # Range not satisfiable
            print(f"  文件已完整下载: {dest_path.name}")
            return True

        if response.status_code not in [200, 206]:
            print(f"  下载失败: HTTP {response.status_code}")
            return False

        # 获取文件总大小
        total_size = int(response.headers.get("content-length", 0))
        if response.status_code == 206:
            total_size += resume_pos

        # 下载模式
        mode = "ab" if resume_pos > 0 else "wb"

        with open(dest_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=resume_pos,
                unit="B",
                unit_scale=True,
                desc=desc or dest_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"  下载错误: {e}")
        return False


def check_model(name: str) -> bool:
    """
    检查模型是否已下载

    Args:
        name: 模型名称

    Returns:
        bool: 模型是否存在
    """
    if name not in MODELS:
        return False

    model_path = get_project_root() / MODELS[name]["path"]
    return model_path.exists()


def download_model(name: str) -> bool:
    """
    下载指定模型

    Args:
        name: 模型名称

    Returns:
        bool: 下载是否成功
    """
    if name not in MODELS:
        print(f"未知模型: {name}")
        return False

    model_info = MODELS[name]
    model_path = get_project_root() / model_info["path"]

    if model_path.exists():
        print(f"模型已存在: {name}")
        return True

    print(f"正在下载: {model_info['description']} ({model_info['size_mb']}MB)")
    return download_file(model_info["url"], model_path, name)


def download_required_models() -> bool:
    """
    下载所有必需模型

    Returns:
        bool: 是否全部下载成功
    """
    print("=" * 50)
    print("检查必需模型...")
    print("=" * 50)

    success = True
    for name in REQUIRED_MODELS:
        if not check_model(name):
            if not download_model(name):
                success = False
        else:
            print(f"[OK] {name} 已存在")

    return success


def download_all_models() -> bool:
    """
    下载所有模型

    Returns:
        bool: 是否全部下载成功
    """
    print("=" * 50)
    print("下载所有模型...")
    print("=" * 50)

    success = True
    for name in MODELS:
        if not check_model(name):
            if not download_model(name):
                success = False
        else:
            print(f"[OK] {name} 已存在")

    return success


def check_all_models() -> Dict[str, bool]:
    """
    检查所有模型状态

    Returns:
        dict: 模型名称 -> 是否存在
    """
    return {name: check_model(name) for name in MODELS}


def get_available_mature_deecho_models() -> List[str]:
    """Return locally available mature DeEcho / DeReverb models."""
    return [name for name in MATURE_DEECHO_MODELS if check_model(name)]


def get_preferred_mature_deecho_model() -> Optional[str]:
    """Return the preferred learned DeEcho model by priority."""
    available = set(get_available_mature_deecho_models())
    for name in MATURE_DEECHO_MODELS:
        if name in available:
            return name
    return None


def download_mature_deecho_models() -> bool:
    """Download mature DeEcho / DeReverb recommended models."""
    print("=" * 50)
    print("Downloading mature DeEcho / DeReverb models...")
    print("=" * 50)

    success = True
    for name in MATURE_DEECHO_MODELS:
        if not check_model(name):
            if not download_model(name):
                success = False
        else:
            print(f"[OK] {name} already exists")

    return success


def print_model_status():
    """打印模型状态"""
    print("=" * 50)
    print("模型状态")
    print("=" * 50)

    status = check_all_models()
    for name, exists in status.items():
        info = MODELS[name]
        mark = "OK" if exists else "MISSING"
        print(f"  {mark} {name}")
        print(f"      {info['description']}")
        print(f"      大小: {info['size_mb']}MB")
        if name in REQUIRED_MODELS:
            print(f"      [必需]")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RVC 模型下载工具")
    parser.add_argument("--check", action="store_true", help="检查模型状态")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--model", type=str, help="下载指定模型")

    args = parser.parse_args()

    if args.check:
        print_model_status()
    elif args.model:
        download_model(args.model)
    elif args.all:
        download_all_models()
    else:
        download_required_models()
