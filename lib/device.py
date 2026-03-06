# -*- coding: utf-8 -*-
"""
设备检测模块 - 自动检测并选择最佳计算设备
支持: CUDA (NVIDIA / AMD ROCm), XPU (Intel Arc via IPEX), DirectML, MPS (Apple), CPU
"""
import torch


def _has_xpu() -> bool:
    """检测 Intel XPU (需要 intel_extension_for_pytorch)"""
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except ImportError:
        return False


def _has_directml() -> bool:
    """检测 DirectML (AMD/Intel on Windows)"""
    try:
        import torch_directml  # noqa: F401
        return True
    except ImportError:
        return False


def _has_mps() -> bool:
    """检测 Apple MPS"""
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False


def _is_rocm() -> bool:
    """检测当前 PyTorch 是否为 ROCm 构建 (AMD GPU)"""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def get_device(preferred: str = "cuda") -> torch.device:
    """
    获取计算设备，按优先级自动回退

    Args:
        preferred: 首选设备 ("cuda", "xpu", "directml", "mps", "cpu")

    Returns:
        torch.device: 可用的计算设备
    """
    p = preferred.lower().strip()

    # 精确匹配请求
    if p in ("cuda", "cuda:0") and torch.cuda.is_available():
        return torch.device("cuda")
    if p in ("xpu", "xpu:0") and _has_xpu():
        return torch.device("xpu")
    if (p == "directml" or p.startswith("privateuseone")) and _has_directml():
        import torch_directml
        return torch_directml.device(torch_directml.default_device())
    if p == "mps" and _has_mps():
        return torch.device("mps")
    if p == "cpu":
        return torch.device("cpu")

    # 自动检测: CUDA (含 ROCm) > XPU > DirectML > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _has_xpu():
        return torch.device("xpu")
    if _has_directml():
        import torch_directml
        return torch_directml.device(torch_directml.default_device())
    if _has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def supports_fp16(device: torch.device) -> bool:
    """判断设备是否支持 FP16 推理"""
    dtype = str(device.type) if hasattr(device, "type") else str(device)
    if dtype == "cuda":
        return True  # CUDA (含 ROCm) 均支持
    if dtype == "xpu":
        return True
    # DirectML / MPS / CPU 不稳定，默认关闭
    return False


def empty_device_cache(device: torch.device = None):
    """清理设备显存缓存（设备无关）"""
    if device is not None:
        dtype = str(device.type) if hasattr(device, "type") else str(device)
    else:
        dtype = None

    if (dtype is None or dtype == "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (dtype is None or dtype == "xpu") and _has_xpu():
        torch.xpu.empty_cache()
    if (dtype is None or dtype == "mps") and _has_mps():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def get_device_info() -> dict:
    """获取设备详细信息"""
    info = {
        "backends": [],
        "current_device": "cpu",
        "devices": []
    }

    # CUDA (NVIDIA 或 AMD ROCm)
    if torch.cuda.is_available():
        backend = "ROCm (AMD)" if _is_rocm() else "CUDA (NVIDIA)"
        info["backends"].append(backend)
        info["current_device"] = "cuda"
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "backend": backend,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            })

    # Intel XPU
    if _has_xpu():
        info["backends"].append("XPU (Intel)")
        if not info["devices"]:
            info["current_device"] = "xpu"
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "backend": "XPU (Intel)",
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            })

    # DirectML
    if _has_directml():
        import torch_directml
        info["backends"].append("DirectML")
        if not info["devices"]:
            info["current_device"] = "directml"
        info["devices"].append({
            "index": 0,
            "backend": "DirectML",
            "name": torch_directml.device_name(0),
            "total_memory_gb": None,
        })

    # MPS
    if _has_mps():
        info["backends"].append("MPS (Apple)")
        if not info["devices"]:
            info["current_device"] = "mps"

    if not info["backends"]:
        info["backends"].append("CPU")

    return info


def print_device_info():
    """打印设备信息到控制台"""
    info = get_device_info()

    print("=" * 50)
    print("设备信息")
    print("=" * 50)
    print(f"可用后端: {', '.join(info['backends'])}")
    print(f"当前设备: {info['current_device']}")

    for dev in info["devices"]:
        mem = f"{dev['total_memory_gb']} GB" if dev.get("total_memory_gb") else "N/A"
        print(f"  [{dev['index']}] {dev['name']} ({dev['backend']}) - 显存: {mem}")

    if not info["devices"]:
        print("  无 GPU 设备，将使用 CPU 进行推理")

    print("=" * 50)
