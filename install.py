# -*- coding: utf-8 -*-
"""
RVC 安装脚本
自动创建虚拟环境 → 安装依赖 → 启动应用

用法:
  python install.py            # 完整安装并启动
  python install.py --check    # 仅检查依赖
  python install.py --no-run   # 安装但不启动
  python install.py --cpu      # 安装 CPU 版本
"""
import subprocess
import sys
import os
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


def install_all(venv_py, gpu=True):
    """安装所有缺失的依赖"""
    missing = check_all(venv_py)

    # 检测 CUDA 版本
    cuda_index_url = None
    if gpu:
        cuda_index_url = detect_cuda_version()
        if cuda_index_url:
            print(f"\n  检测到 CUDA，使用 PyTorch 源: {cuda_index_url}")
        else:
            print("\n  [错误] 未检测到支持的 CUDA，GPU 安装停止。")
            print("  如需 CPU 版 PyTorch，请显式使用 --cpu。")
            return False

    runtime_dist = "onnxruntime-gpu" if gpu else "onnxruntime"
    conflicting_runtime_dist = "onnxruntime" if gpu else "onnxruntime-gpu"
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
        return check_dependency_consistency(venv_py)

    print(f"\n开始安装 {len(missing)} 个缺失的依赖...\n")
    failed = []
    torch_stack_names = {"torch", "torchvision", "torchaudio"}
    torch_stack_missing = [info for info in missing if info["pip"] in torch_stack_names]
    if torch_stack_missing:
        torch_index_url = (
            cuda_index_url
            if gpu and cuda_index_url
            else "https://download.pytorch.org/whl/cpu"
        )
        if not pip_install_packages(
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
                extra="gpu" if gpu else "cpu",
                version_spec=version_spec,
            )
        else:
            ok = pip_install(venv_py, pip_name)
        if not ok:
            failed.append(info["name"])

    if not get_installed_version(venv_py, runtime_dist):
        failed.append(runtime_dist)
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
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"安装脚本不支持写入设备: {device!r}")
    config_path = ROOT_DIR / "configs" / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["device"] = device
    with open(config_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    print(f"已写入运行设备: {device} ({config_path})")


def check_onnx_runtime_variant(venv_py: str, gpu: bool) -> bool:
    """检查当前安装模式只存在对应的 ONNX Runtime 发行包。"""
    required_dist = "onnxruntime-gpu" if gpu else "onnxruntime"
    conflicting_dist = "onnxruntime" if gpu else "onnxruntime-gpu"
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
    parser.add_argument("--cpu", action="store_true", help="安装 CPU 版本")
    parser.add_argument("--check", action="store_true", help="仅检查依赖")
    parser.add_argument("--no-run", action="store_true", help="安装后不启动")
    args = parser.parse_args()

    print("=" * 50)
    print("RVC 安装程序")
    print("=" * 50)

    # 1. 创建虚拟环境
    print("\n[1/3] 检查虚拟环境")
    if not create_venv():
        sys.exit(1)

    venv_py = get_venv_python()

    # 2. 安装依赖
    print(f"\n[2/3] 检查依赖")
    if args.check:
        missing = check_all(venv_py)
        runtime_ok = check_onnx_runtime_variant(venv_py, gpu=not args.cpu)
        dependencies_ok = check_dependency_consistency(venv_py)
        if missing or not runtime_ok or not dependencies_ok:
            sys.exit(1)
        return

    gpu = not args.cpu
    if not install_all(venv_py, gpu=gpu):
        print("\n部分依赖安装失败，可尝试手动安装。")
        sys.exit(1)

    save_runtime_device("cuda" if gpu else "cpu")

    # 3. 启动应用
    if args.no_run:
        print("\n安装完成。运行方式:")
        print(f"  {venv_py} run.py")
        return

    print(f"\n[3/3] 启动应用")
    launch_app(venv_py)


if __name__ == "__main__":
    main()
