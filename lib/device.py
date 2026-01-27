# -*- coding: utf-8 -*-
"""
设备检测模块 - 自动检测并选择最佳计算设备
"""
import torch
from typing import Tuple


def get_device(preferred: str = "cuda") -> torch.device:
    """
    获取计算设备

    Args:
        preferred: 首选设备 ("cuda" 或 "cpu")

    Returns:
        torch.device: 可用的计算设备
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    获取设备详细信息

    Returns:
        dict: 包含设备信息的字典
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "current_device": "cpu",
        "devices": []
    }

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = "cuda"

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}"
            })

    return info


def print_device_info():
    """打印设备信息到控制台"""
    info = get_device_info()

    print("=" * 50)
    print("设备信息")
    print("=" * 50)

    if info["cuda_available"]:
        print(f"CUDA 可用: 是")
        print(f"GPU 数量: {info['device_count']}")
        for dev in info["devices"]:
            print(f"  [{dev['index']}] {dev['name']}")
            print(f"      显存: {dev['total_memory_gb']} GB")
            print(f"      计算能力: {dev['compute_capability']}")
    else:
        print("CUDA 可用: 否")
        print("将使用 CPU 进行推理")

    print("=" * 50)
