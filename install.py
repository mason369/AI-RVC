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
from pathlib import Path

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
    "torchaudio": {"import": "torchaudio", "name": "torchaudio", "pip": "torchaudio"},
    "gradio": {"import": "gradio", "name": "Gradio", "pip": "gradio==3.50.2"},
    "librosa": {"import": "librosa", "name": "librosa", "pip": "librosa"},
    "soundfile": {"import": "soundfile", "name": "soundfile", "pip": "soundfile"},
    "av": {"import": "av", "name": "PyAV", "pip": "av"},
    "scipy": {"import": "scipy", "name": "scipy", "pip": "scipy"},
    "numpy": {
        "import": "numpy",
        "name": "numpy",
        "pip": "numpy<2,>=1.23.0",
        "dist": "numpy",
        "min_version": "1.23.0",
        "max_exclusive_version": "2.0.0",
    },
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
        "min_version": "0.44.1",
    },
    "huggingface_hub": {"import": "huggingface_hub", "name": "huggingface_hub", "pip": "huggingface_hub"},
    "pedalboard": {"import": "pedalboard", "name": "pedalboard", "pip": "pedalboard"},
    "ffmpeg": {"import": "ffmpeg", "name": "ffmpeg-python", "pip": "ffmpeg-python"},
    "fairseq": {"import": "fairseq", "name": "fairseq", "pip": "fairseq==0.12.2"},
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
                if major >= 12 and minor >= 6:
                    return "https://download.pytorch.org/whl/cu126"
                elif major >= 12 and minor >= 4:
                    return "https://download.pytorch.org/whl/cu124"
                elif major >= 12 and minor >= 1:
                    return "https://download.pytorch.org/whl/cu121"
                elif major >= 11 and minor >= 8:
                    return "https://download.pytorch.org/whl/cu118"
                else:
                    # CUDA 版本太旧，回退 CPU
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
        # fairseq 依赖冲突时尝试 --no-deps 回退
        if not no_deps and "ResolutionImpossible" in r.stderr:
            print(f"  [依赖冲突] 尝试 --no-deps 安装 {target} ...")
            return pip_install(venv_py, package, extra=extra, index_url=index_url, no_deps=True)
        print(f"  [失败] {target}")
        lines = r.stderr.strip().splitlines()
        if lines:
            print(f"  {lines[-1]}")
        return False
    print(f"  [完成] {target}")
    return True


def check_all(venv_py):
    """检查所有依赖"""
    print("=" * 50)
    print("RVC 依赖检查")
    print("=" * 50)
    missing = []
    for key, info in PACKAGES.items():
        ok = check_package(venv_py, info["import"])
        status = "OK" if ok else "未安装"
        min_version = info.get("min_version")
        max_exclusive_version = info.get("max_exclusive_version")
        if ok and (min_version or max_exclusive_version):
            installed_version = get_installed_version(
                venv_py,
                info.get("dist", info["pip"]),
            )
            if not installed_version:
                ok = False
                requirements = []
                if min_version:
                    requirements.append(f">= {min_version}")
                if max_exclusive_version:
                    requirements.append(f"< {max_exclusive_version}")
                status = f"需更新 (无法读取版本，要求 {' 且 '.join(requirements)})"
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
                requirements = []
                if min_version:
                    requirements.append(f">= {min_version}")
                if max_exclusive_version:
                    requirements.append(f"< {max_exclusive_version}")
                status = f"需更新 ({installed_version} 不满足 {' 且 '.join(requirements)})"
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
    if not missing:
        print("\n无需安装，所有依赖已就绪。")
        return True

    # 检测 CUDA 版本
    cuda_index_url = None
    if gpu:
        cuda_index_url = detect_cuda_version()
        if cuda_index_url:
            print(f"\n  检测到 CUDA，使用 PyTorch 源: {cuda_index_url}")
        else:
            print("\n  未检测到 CUDA，将安装 CPU 版 PyTorch")

    print(f"\n开始安装 {len(missing)} 个缺失的依赖...\n")
    failed = []
    for info in missing:
        pip_name = info["pip"]
        if pip_name in ("torch", "torchaudio"):
            if gpu and cuda_index_url:
                ok = pip_install(venv_py, pip_name, index_url=cuda_index_url)
            else:
                ok = pip_install(venv_py, pip_name, index_url="https://download.pytorch.org/whl/cpu")
        elif pip_name == "audio-separator":
            version_spec = f">={info['min_version']}" if info.get("min_version") else ""
            ok = pip_install(
                venv_py,
                pip_name,
                extra="gpu" if gpu else "cpu",
                version_spec=version_spec,
            )
            if ok:
                # audio-separator 0.31+ declares numpy>=2, while the current
                # Gradio 3.x UI stack is pinned to numpy 1.x. The separator
                # model table and runtime imports work with numpy 1.26, so
                # restore the app-compatible numpy line after separator install.
                ok = pip_install(venv_py, "numpy<2,>=1.23.0")
        else:
            ok = pip_install(venv_py, pip_name)
        if not ok:
            failed.append(info["name"])

    print("\n" + "=" * 50)
    if failed:
        print(f"安装完成，{len(failed)} 个包失败: {', '.join(failed)}")
        return False
    print("所有依赖安装成功!")
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
        check_all(venv_py)
        return

    gpu = not args.cpu
    if not install_all(venv_py, gpu=gpu):
        print("\n部分依赖安装失败，可尝试手动安装。")
        sys.exit(1)

    # 3. 启动应用
    if args.no_run:
        print("\n安装完成。运行方式:")
        print(f"  {venv_py} run.py")
        return

    print(f"\n[3/3] 启动应用")
    launch_app(venv_py)


if __name__ == "__main__":
    main()
