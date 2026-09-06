"""Activate upstream NVIDIA hooks for indirect CUDA Toolkit dependencies."""
from importlib.metadata import distributions

from _pyinstaller_hooks_contrib.utils.nvidia_cuda import infer_hiddenimports_from_requirements
from PyInstaller.utils.hooks import PY_DYLIB_PATTERNS, collect_dynamic_libs

# Torch 2.11 also obtains CUDA libraries through cuda-toolkit extras. The upstream
# torch hook only inspects direct requirements, so activate its existing per-library
# hooks for every NVIDIA runtime actually installed in this isolated build environment.
hiddenimports = sorted(infer_hiddenimports_from_requirements([
    distribution.metadata["Name"] for distribution in distributions()
]))
if "nvidia.cuda_nvrtc" not in hiddenimports:
    raise RuntimeError("CUDA 发行环境缺少 nvidia.cuda_nvrtc 依赖元数据")

# Include dynamically loaded companions and plugins, including names such as
# nvshmem_bootstrap_uid.so.3 which do not start with "lib".
binaries = collect_dynamic_libs('nvidia', search_patterns=PY_DYLIB_PATTERNS + ['*.so.*'])
bindepend_symlink_suppression = ['**/nvidia/**/*.so*']
