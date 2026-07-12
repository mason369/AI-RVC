# -*- coding: utf-8 -*-
"""
RVC 安装脚本
自动创建虚拟环境 → 安装依赖 → 启动应用

用法:
  python install.py            # 完整安装并启动
  python install.py --check    # 仅检查依赖
  python install.py --no-run   # 安装但不启动
  python install.py --cpu      # 安装 CPU 版本
  python install.py --backend mps       # Apple MPS
  python install.py --backend directml  # Windows DirectML
"""
import subprocess
import sys
import os
import platform
import json
from pathlib import Path

from lib.console_encoding import configure_console_encoding
from lib.console_i18n import console_print as print

configure_console_encoding()

ROOT_DIR = Path(__file__).parent
VENV_DIR = ROOT_DIR / "venv310"

PYTHON310_CANDIDATES = [
    r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe",
    r"C:\Python310\python.exe",
    r"C:\Program Files\Python310\python.exe",
    r"C:\Program Files (x86)\Python310\python.exe",
]

SUPPORTED_BACKENDS = ("auto", "cpu", "cuda", "rocm", "xpu", "directml", "mps")
BACKEND_SETTINGS = {
    "cpu": {
        "audio_extra": "cpu",
        "runtime_dist": "onnxruntime",
        "conflicting_runtime_dists": ("onnxruntime-gpu", "onnxruntime-directml"),
        "runtime_device": "cpu",
        "torch_install": "cpu",
    },
    "cuda": {
        "audio_extra": "gpu",
        "runtime_dist": "onnxruntime-gpu",
        "conflicting_runtime_dists": ("onnxruntime", "onnxruntime-directml"),
        "runtime_device": "cuda",
        "torch_install": "cuda",
    },
    "rocm": {
        "audio_extra": "cpu",
        "runtime_dist": "onnxruntime",
        "conflicting_runtime_dists": ("onnxruntime-gpu", "onnxruntime-directml"),
        "runtime_device": "cuda",
        "torch_install": "preinstalled",
    },
    "xpu": {
        "audio_extra": "cpu",
        "runtime_dist": "onnxruntime",
        "conflicting_runtime_dists": ("onnxruntime-gpu", "onnxruntime-directml"),
        "runtime_device": "xpu",
        "torch_install": "preinstalled",
    },
    "directml": {
        "audio_extra": "dml",
        "runtime_dist": "onnxruntime-directml",
        "conflicting_runtime_dists": ("onnxruntime", "onnxruntime-gpu"),
        "runtime_device": "directml",
        "torch_install": "preinstalled",
    },
    "mps": {
        "audio_extra": "cpu",
        "runtime_dist": "onnxruntime",
        "conflicting_runtime_dists": ("onnxruntime-gpu", "onnxruntime-directml"),
        "runtime_device": "mps",
        "torch_install": "pypi",
    },
}

PACKAGES = {
    "torch": {"import": "torch", "name": "PyTorch", "pip": "torch"},
    "torchvision": {"import": "torchvision", "name": "torchvision", "pip": "torchvision"},
    "torchaudio": {"import": "torchaudio", "name": "torchaudio", "pip": "torchaudio"},
    "gradio": {
        "import": "gradio",
        "name": "Gradio",
        "pip": "gradio==5.49.1",
        "dist": "gradio",
        "required_version": "5.49.1",
    },
    "librosa": {"import": "librosa", "name": "librosa", "pip": "librosa"},
    "soundfile": {"import": "soundfile", "name": "soundfile", "pip": "soundfile"},
    "av": {"import": "av", "name": "PyAV", "pip": "av"},
    "scipy": {"import": "scipy", "name": "scipy", "pip": "scipy"},
    "numpy": {
        "import": "numpy",
        "name": "numpy",
        "pip": "numpy>=2,<3",
        "dist": "numpy",
        "min_version": "2.0.0",
        "max_exclusive_version": "3.0.0",
    },
    "yaml": {"import": "yaml", "name": "PyYAML", "pip": "PyYAML"},
    "einops": {"import": "einops", "name": "einops", "pip": "einops"},
    "parselmouth": {"import": "parselmouth", "name": "praat-parselmouth", "pip": "praat-parselmouth"},
    "pyworld": {"import": "pyworld", "name": "pyworld", "pip": "pyworld"},
    "torchcrepe": {"import": "torchcrepe", "name": "torchcrepe", "pip": "torchcrepe"},
    "faiss": {"import": "faiss", "name": "faiss-cpu", "pip": "faiss-cpu"},
    "tqdm": {"import": "tqdm", "name": "tqdm", "pip": "tqdm"},
    "requests": {"import": "requests", "name": "requests", "pip": "requests"},
    "dotenv": {"import": "dotenv", "name": "python-dotenv", "pip": "python-dotenv"},
    "colorama": {"import": "colorama", "name": "colorama", "pip": "colorama"},
    "mcp": {"import": "mcp", "name": "mcp", "pip": "mcp"},
    "demucs": {"import": "demucs", "name": "demucs", "pip": "demucs"},
    "audio_separator": {
        "import": "audio_separator",
        "name": "audio-separator",
        "pip": "audio-separator",
        "dist": "audio-separator",
        "required_version": "0.44.1",
    },
    "huggingface_hub": {
        "import": "huggingface_hub",
        "name": "huggingface_hub",
        "pip": "huggingface_hub>=0.19.0,<1.0",
        "dist": "huggingface_hub",
        "min_version": "0.19.0",
        "max_exclusive_version": "1.0",
    },
    "pedalboard": {"import": "pedalboard", "name": "pedalboard", "pip": "pedalboard"},
    "ffmpeg": {"import": "ffmpeg", "name": "ffmpeg-python", "pip": "ffmpeg-python"},
    "fairseq": {
        "import": "fairseq",
        "name": "fairseq",
        "pip": "fairseq==0.12.2",
        "dist": "fairseq",
        "required_version": "0.12.2",
    },
}

# === 虚拟环境 ===

def find_python310():
    """查找系统中的 Python 3.10"""
    if sys.version_info[:2] == (3, 10):
        return sys.executable
    for p in PYTHON310_CANDIDATES:
        if os.path.isfile(p):
            return p
    try:
        r = subprocess.run(
            ["py", "-3.10", "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_venv_python():
    """获取虚拟环境的 Python 路径"""
    if os.name == "nt":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def create_venv():
    """创建 Python 3.10 虚拟环境"""
    venv_py = get_venv_python()
    if os.path.isfile(venv_py):
        r = subprocess.run([venv_py, "--version"], capture_output=True, text=True)
        if r.returncode == 0 and "3.10" in r.stdout:
            print(f"  [OK] 虚拟环境已存在: {VENV_DIR}")
            return True

    py310 = find_python310()
    if not py310:
        print("  [错误] 未找到 Python 3.10")
        print("  下载: https://www.python.org/downloads/release/python-31011/")
        return False

    print(f"  使用 Python: {py310}")
    print(f"  创建虚拟环境: {VENV_DIR}")
    r = subprocess.run([py310, "-m", "venv", str(VENV_DIR)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [错误] 创建失败:\n{r.stderr}")
        return False

    print("  [OK] 虚拟环境创建成功")
    print("  升级 pip ...")
    subprocess.run([venv_py, "-m", "pip", "install", "--upgrade", "pip"],
                   capture_output=True, text=True)
    return True

# === 依赖检查与安装 ===

def check_package(venv_py, import_name):
    """用虚拟环境的 Python 检查包是否已安装"""
    r = subprocess.run(
        [venv_py, "-c", f"import {import_name}"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def get_installed_version(venv_py, distribution_name):
    """返回虚拟环境中已安装发行包版本；无法读取时返回 None。"""
    code = (
        "from importlib.metadata import PackageNotFoundError, version\n"
        f"dist = {distribution_name!r}\n"
        "try:\n"
        "    print(version(dist))\n"
        "except PackageNotFoundError:\n"
        "    raise SystemExit(1)\n"
    )
    r = subprocess.run([venv_py, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _version_parts(version_text):
    parts = []
    for part in str(version_text or "").split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits or 0))
    return tuple(parts)


def _version_at_least(installed, required):
    installed_parts = _version_parts(installed)
    required_parts = _version_parts(required)
    width = max(len(installed_parts), len(required_parts))
    installed_parts += (0,) * (width - len(installed_parts))
    required_parts += (0,) * (width - len(required_parts))
    return installed_parts >= required_parts


def _version_less_than(installed, upper_bound):
    installed_parts = _version_parts(installed)
    upper_parts = _version_parts(upper_bound)
    width = max(len(installed_parts), len(upper_parts))
    installed_parts += (0,) * (width - len(installed_parts))
    upper_parts += (0,) * (width - len(upper_parts))
    return installed_parts < upper_parts


def _version_matches(installed, required):
    return str(installed or "").strip() == str(required or "").strip()


def _version_requirement_text(info):
    required_version = info.get("required_version")
    if required_version:
        return f"== {required_version}"
    requirements = []
    if info.get("min_version"):
        requirements.append(f">= {info['min_version']}")
    if info.get("max_exclusive_version"):
        requirements.append(f"< {info['max_exclusive_version']}")
    return " 且 ".join(requirements)


def detect_cuda_version():
    """检测系统 CUDA 版本，返回对应的 PyTorch index-url"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            # nvidia-smi 存在，尝试获取 CUDA 版本
            r2 = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=10,
            )
            output = r2.stdout
            # 从 nvidia-smi 输出中提取 CUDA Version
            import re
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                if (major, minor) >= (12, 6):
                    return "https://download.pytorch.org/whl/cu126"
                elif (major, minor) >= (12, 4):
                    return "https://download.pytorch.org/whl/cu124"
                elif (major, minor) >= (12, 1):
                    return "https://download.pytorch.org/whl/cu121"
                elif (major, minor) >= (11, 8):
                    return "https://download.pytorch.org/whl/cu118"
                else:
                    return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def resolve_install_backend(requested: str, venv_py: str = None) -> str:
    """Resolve an explicit installer backend; only ``auto`` performs detection."""
    backend = str(requested or "auto").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"不支持的安装后端: {requested!r}")
    if backend != "auto":
        return backend
    if venv_py and os.path.isfile(venv_py):
        probe = (
            "import torch\n"
            "backend = 'cpu'\n"
            "if getattr(torch.version, 'hip', None) and torch.cuda.is_available(): backend = 'rocm'\n"
            "elif hasattr(torch, 'xpu') and torch.xpu.is_available(): backend = 'xpu'\n"
            "elif torch.cuda.is_available(): backend = 'cuda'\n"
            "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): backend = 'mps'\n"
            "else:\n"
            "    try:\n"
            "        import torch_directml\n"
            "        torch.zeros(1).to(torch_directml.device(torch_directml.default_device()))\n"
            "        backend = 'directml'\n"
            "    except Exception:\n"
            "        pass\n"
            "print(backend)\n"
        )
        result = subprocess.run(
            [venv_py, "-c", probe],
            capture_output=True,
            text=True,
        )
        detected = result.stdout.strip().lower() if result.returncode == 0 else ""
        if detected in BACKEND_SETTINGS and detected != "cpu":
            return detected
    if sys.platform == "darwin":
        machine = platform.machine().lower()
        return "mps" if machine in {"arm64", "aarch64"} else "cpu"
    if detect_cuda_version():
        return "cuda"
    return "cpu"


def check_backend_available(venv_py: str, backend: str) -> bool:
    """Verify that the selected accelerator stack is importable and usable."""
    checks = {
        "cpu": "import torch; torch.zeros(1, device='cpu')",
        "cuda": (
            "import torch; "
            "assert torch.version.hip is None, 'ROCm PyTorch cannot satisfy CUDA mode'; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
            "torch.zeros(1, device='cuda')"
        ),
        "rocm": (
            "import torch; "
            "assert torch.version.hip is not None, 'PyTorch is not a ROCm build'; "
            "assert torch.cuda.is_available(), 'ROCm device is unavailable'; "
            "torch.zeros(1, device='cuda')"
        ),
        "xpu": (
            "import torch, intel_extension_for_pytorch; "
            "assert hasattr(torch, 'xpu') and torch.xpu.is_available(), 'XPU is unavailable'; "
            "torch.zeros(1, device='xpu')"
        ),
        "directml": (
            "import torch, torch_directml; "
            "device = torch_directml.device(torch_directml.default_device()); "
            "torch.zeros(1).to(device)"
        ),
        "mps": (
            "import torch; "
            "assert hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(), "
            "'MPS is unavailable'; "
            "torch.zeros(1, device='mps')"
        ),
    }
    result = subprocess.run(
        [venv_py, "-c", checks[backend]],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  [v] 计算后端可用: {backend}")
        return True
    details = (result.stderr or result.stdout).strip().splitlines()
    print(f"  [x] 计算后端不可用: {backend}")
    if details:
        print(f"      {details[-1]}")
    return False


def pip_install(venv_py, package, extra="", index_url=None, no_deps=False, version_spec=""):
    """用虚拟环境的 pip 安装包"""
    target = f"{package}[{extra}]{version_spec}" if extra else f"{package}{version_spec}"
    print(f"  安装 {target} ...")
    cmd = [venv_py, "-m", "pip", "install", target]
    if index_url:
        cmd.extend(["--index-url", index_url])
    if no_deps:
        cmd.append("--no-deps")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [失败] {target}")
        lines = r.stderr.strip().splitlines()
        if lines:
            print(f"  {lines[-1]}")
        return False
    print(f"  [完成] {target}")
    return True


def pip_install_packages(venv_py, packages, index_url=None):
    """Install a compatibility-coupled package set in one resolver transaction."""
    targets = [str(package) for package in packages]
    print(f"  安装 {' '.join(targets)} ...")
    cmd = [venv_py, "-m", "pip", "install", *targets]
    if index_url:
        cmd.extend(["--index-url", index_url])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [失败] {' '.join(targets)}")
        lines = result.stderr.strip().splitlines()
        if lines:
            print(f"  {lines[-1]}")
        return False
    print(f"  [完成] {' '.join(targets)}")
    return True


def check_dependency_consistency(venv_py):
    """Run pip's installed-distribution consistency check without hiding failures."""
    result = subprocess.run(
        [venv_py, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  [v] pip 依赖一致性检查通过")
        return True
    print("  [x] pip 依赖一致性检查失败:")
    details = (result.stdout or result.stderr).strip()
    if details:
        for line in details.splitlines():
            print(f"      {line}")
    return False


def check_all(venv_py):
    """检查所有依赖"""
    print("=" * 50)
    print("RVC 依赖检查")
    print("=" * 50)
    missing = []
    for key, info in PACKAGES.items():
        ok = check_package(venv_py, info["import"])
        status = "OK" if ok else "未安装"
        required_version = info.get("required_version")
        min_version = info.get("min_version")
        max_exclusive_version = info.get("max_exclusive_version")
        if ok and (required_version or min_version or max_exclusive_version):
            installed_version = get_installed_version(
                venv_py,
                info.get("dist", info["pip"]),
            )
            if not installed_version:
                ok = False
                status = f"需更新 (无法读取版本，要求 {_version_requirement_text(info)})"
            elif required_version:
                if _version_matches(installed_version, required_version):
                    status = f"OK ({installed_version})"
                else:
                    ok = False
                    status = (
                        f"需更新 ({installed_version} 不等于 "
                        f"{required_version})"
                    )
            elif (
                (not min_version or _version_at_least(installed_version, min_version))
                and (
                    not max_exclusive_version
                    or _version_less_than(installed_version, max_exclusive_version)
                )
            ):
                status = f"OK ({installed_version})"
            else:
                ok = False
                status = (
                    f"需更新 ({installed_version} 不满足 "
                    f"{_version_requirement_text(info)})"
                )
        mark = "[v]" if ok else "[x]"
        print(f"  {mark} {info['name']:30s} {status}")
        if not ok:
            missing.append(info)
    print("-" * 50)
    if missing:
        print(f"缺少 {len(missing)} 个依赖包")
    else:
        print("所有依赖已安装")
    return missing


def install_all(venv_py, gpu=True, backend=None):
    """安装所有缺失依赖，并保持所选计算后端的运行时组合。"""
    backend = str(backend or ("cuda" if gpu else "cpu")).strip().lower()
    if backend not in BACKEND_SETTINGS:
        raise ValueError(f"不支持的安装后端: {backend!r}")
    settings = BACKEND_SETTINGS[backend]
    missing = check_all(venv_py)

    torch_index_url = None
    if settings["torch_install"] == "cuda":
        cuda_index_url = detect_cuda_version()
        if cuda_index_url:
            print(f"\n  检测到 CUDA，使用 PyTorch 源: {cuda_index_url}")
            torch_index_url = cuda_index_url
        else:
            print("\n  [错误] 未检测到支持的 CUDA，GPU 安装停止。")
            print("  如需 CPU 版 PyTorch，请显式使用 --cpu。")
            return False
    elif settings["torch_install"] == "cpu":
        torch_index_url = "https://download.pytorch.org/whl/cpu"

    runtime_dist = settings["runtime_dist"]
    for conflicting_runtime_dist in settings["conflicting_runtime_dists"]:
        if get_installed_version(venv_py, conflicting_runtime_dist):
            print(
                f"\n  [错误] 检测到冲突的 {conflicting_runtime_dist}。"
                f"当前安装模式只允许 {runtime_dist}，请删除 venv310 后重新安装。"
            )
            return False

    if not get_installed_version(venv_py, runtime_dist):
        audio_separator_info = PACKAGES["audio_separator"]
        if audio_separator_info not in missing:
            missing.append(audio_separator_info)

    if not missing:
        print("\n无需安装，所有依赖已就绪。")
        return (
            check_backend_available(venv_py, backend)
            and check_dependency_consistency(venv_py)
        )

    print(f"\n开始安装 {len(missing)} 个缺失的依赖...\n")
    failed = []
    torch_stack_names = {"torch", "torchvision", "torchaudio"}
    torch_stack_missing = [info for info in missing if info["pip"] in torch_stack_names]
    if torch_stack_missing:
        if settings["torch_install"] == "preinstalled":
            print(
                f"  [错误] {backend} 需要先安装并验证对应的 PyTorch 运行栈；"
                "安装器不会用其他 PyTorch 版本覆盖它"
            )
            failed.extend(info["name"] for info in torch_stack_missing)
        elif not pip_install_packages(
            venv_py,
            ("torch", "torchvision", "torchaudio"),
            index_url=torch_index_url,
        ):
            failed.extend(info["name"] for info in torch_stack_missing)
        missing = [info for info in missing if info["pip"] not in torch_stack_names]

    for info in missing:
        pip_name = info["pip"]
        if pip_name == "audio-separator":
            if info.get("required_version"):
                version_spec = f"=={info['required_version']}"
            else:
                version_spec = f">={info['min_version']}" if info.get("min_version") else ""
            ok = pip_install(
                venv_py,
                pip_name,
                extra=settings["audio_extra"],
                version_spec=version_spec,
            )
        else:
            ok = pip_install(venv_py, pip_name)
        if not ok:
            failed.append(info["name"])

    if not get_installed_version(venv_py, runtime_dist):
        failed.append(runtime_dist)
    if not check_backend_available(venv_py, backend):
        failed.append(f"{backend} backend")
    if not check_dependency_consistency(venv_py):
        failed.append("pip dependency consistency")

    print("\n" + "=" * 50)
    if failed:
        print(f"安装完成，{len(failed)} 个包失败: {', '.join(failed)}")
        return False
    print("所有依赖安装成功!")
    return True


def save_runtime_device(device: str) -> None:
    """把显式安装模式写入运行配置，避免设备被隐式改写。"""
    if device not in {"cpu", "cuda", "xpu", "directml", "mps"}:
        raise ValueError(f"安装脚本不支持写入设备: {device!r}")
    config_path = ROOT_DIR / "configs" / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["device"] = device
    with open(config_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    print(f"已写入运行设备: {device} ({config_path})")


def check_onnx_runtime_variant(venv_py: str, gpu: bool = None, backend: str = None) -> bool:
    """检查当前安装模式只存在对应的 ONNX Runtime 发行包。"""
    backend = str(backend or ("cuda" if gpu else "cpu")).strip().lower()
    if backend not in BACKEND_SETTINGS:
        raise ValueError(f"不支持的安装后端: {backend!r}")
    settings = BACKEND_SETTINGS[backend]
    required_dist = settings["runtime_dist"]
    for conflicting_dist in settings["conflicting_runtime_dists"]:
        conflicting_version = get_installed_version(venv_py, conflicting_dist)
        if conflicting_version:
            print(
                f"  [x] ONNX Runtime 检测到冲突: {conflicting_dist} "
                f"{conflicting_version}；当前模式只允许 {required_dist}"
            )
            return False
    required_version = get_installed_version(venv_py, required_dist)
    if not required_version:
        print(f"  [x] ONNX Runtime 缺少 {required_dist}")
        return False
    print(f"  [v] ONNX Runtime {required_dist} {required_version}")
    return True


def launch_app(venv_py):
    """用虚拟环境启动应用"""
    run_script = str(ROOT_DIR / "run.py")
    print(f"\n启动应用: {run_script}")
    print("=" * 50)
    try:
        subprocess.run([venv_py, run_script], cwd=str(ROOT_DIR))
    except KeyboardInterrupt:
        print("\n已停止")

# === 主入口 ===

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RVC 安装脚本")
    parser.add_argument("--cpu", action="store_true", help="安装 CPU 版本（--backend cpu 的兼容别名）")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default=None,
        help="安装后端: auto/cpu/cuda/rocm/xpu/directml/mps",
    )
    parser.add_argument("--check", action="store_true", help="仅检查依赖")
    parser.add_argument("--no-run", action="store_true", help="安装后不启动")
    args = parser.parse_args()

    if args.cpu and args.backend not in {None, "cpu"}:
        parser.error("--cpu 不能与非 CPU 的 --backend 同时使用")
    requested_backend = "cpu" if args.cpu else (args.backend or "auto")

    print("=" * 50)
    print("RVC 安装程序")
    print("=" * 50)

    # 1. 创建虚拟环境
    print("\n[1/3] 检查虚拟环境")
    if not create_venv():
        sys.exit(1)

    venv_py = get_venv_python()
    backend = resolve_install_backend(requested_backend, venv_py=venv_py)
    settings = BACKEND_SETTINGS[backend]
    print(f"安装后端: {backend}")

    # 2. 安装依赖
    print(f"\n[2/3] 检查依赖")
    if args.check:
        missing = check_all(venv_py)
        runtime_ok = check_onnx_runtime_variant(venv_py, backend=backend)
        backend_ok = check_backend_available(venv_py, backend)
        dependencies_ok = check_dependency_consistency(venv_py)
        if missing or not runtime_ok or not backend_ok or not dependencies_ok:
            sys.exit(1)
        return

    if not install_all(venv_py, backend=backend):
        print("\n部分依赖安装失败，可尝试手动安装。")
        sys.exit(1)

    save_runtime_device(settings["runtime_device"])

    # 3. 启动应用
    if args.no_run:
        print("\n安装完成。运行方式:")
        print(f"  {venv_py} run.py")
        return

    print(f"\n[3/3] 启动应用")
    launch_app(venv_py)


if __name__ == "__main__":
    main()
