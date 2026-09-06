"""Require both pinned upstream runtimes before and after portable bundling."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import upstream_runtime
from lib.console_i18n import console_print as print
from lib.console_i18n import localize_console_message


def validate_upstream_sources(project_root: Path) -> int:
    manifest_path = Path(upstream_runtime.__file__).with_name("upstream_source_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for capability in upstream_runtime.REVISIONS:
        root = upstream_runtime.runtime_root(project_root, capability)
        problems = upstream_runtime._packaged_source_problems(root, capability)
        if problems:
            raise RuntimeError(f"发行资源 {capability} 源码校验失败: " + "；".join(problems))
        checked += len(manifest[capability]["files"])
    return checked


def validate_cuda_runtime(runtime_root: Path, cuda_version: str = "12.8") -> int:
    """Compile with the distributed NVRTC pair; no GPU or driver is required."""
    major, minor = map(int, cuda_version.split(".")[:2])
    windows = sys.platform == "win32"
    library_name = f"nvrtc64_{major}0_0.dll" if windows else f"libnvrtc.so.{major}"
    builtins_name = f"nvrtc-builtins64_{major}{minor}.dll" if windows else f"libnvrtc-builtins.so.{major}.{minor}"
    locations = (runtime_root / "torch/lib", runtime_root / "nvidia/cuda_nvrtc/lib", runtime_root)
    libraries = {path.resolve() for directory in locations if (path := directory / library_name).is_file()}
    if len(libraries) != 1:
        raise RuntimeError(f"发行包必须包含唯一的 CUDA NVRTC 主库: {library_name}")
    library_path = libraries.pop()
    builtins_path = library_path.with_name(builtins_name)
    if not builtins_path.is_file():
        raise RuntimeError(f"发行包缺少匹配的 CUDA NVRTC 内置库: {builtins_path}")

    dll_directory = os.add_dll_directory(str(library_path.parent)) if windows else None
    program = ctypes.c_void_p()
    nvrtc = None
    try:
        # NVRTC loads this companion by name. Keep the matching absolute library
        # loaded, as torch does on Windows, without consulting a host CUDA Toolkit.
        builtins_library = ctypes.CDLL(str(builtins_path))
        nvrtc = ctypes.CDLL(str(library_path))
        signatures = {
            "nvrtcVersion": [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
            "nvrtcCreateProgram": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p],
            "nvrtcCompileProgram": [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
            "nvrtcGetProgramLogSize": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
            "nvrtcGetProgramLog": [ctypes.c_void_p, ctypes.c_void_p],
            "nvrtcGetPTXSize": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
            "nvrtcGetPTX": [ctypes.c_void_p, ctypes.c_void_p],
            "nvrtcDestroyProgram": [ctypes.POINTER(ctypes.c_void_p)],
        }
        for name, arguments in signatures.items():
            function = getattr(nvrtc, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        actual_major, actual_minor = ctypes.c_int(), ctypes.c_int()
        status = nvrtc.nvrtcVersion(ctypes.byref(actual_major), ctypes.byref(actual_minor))
        if status or (actual_major.value, actual_minor.value) != (major, minor):
            raise RuntimeError(f"发行包 CUDA NVRTC 版本不符: {actual_major.value}.{actual_minor.value}; {status}")
        source = b'extern "C" __global__ void verify(float *x) { x[0] = sinf(x[0]) + sqrtf(x[0]); }'
        status = nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"verify.cu", 0, None, None)
        if status:
            raise RuntimeError(f"CUDA NVRTC 创建编译任务失败: {status}")
        options = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_75")
        status = nvrtc.nvrtcCompileProgram(program, 1, options)
        if status:
            log_size = ctypes.c_size_t()
            log_status = nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
            if log_status or not log_size.value:
                raise RuntimeError(f"CUDA NVRTC 编译和日志读取失败: {status}; {log_status}")
            log = ctypes.create_string_buffer(log_size.value)
            log_status = nvrtc.nvrtcGetProgramLog(program, log)
            if log_status:
                raise RuntimeError(f"CUDA NVRTC 日志读取失败: {log_status}")
            raise RuntimeError("CUDA NVRTC 编译失败: " + log.value.decode("utf-8"))
        ptx_size = ctypes.c_size_t()
        status = nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
        if status or ptx_size.value <= 1:
            raise RuntimeError(f"CUDA NVRTC 编译产物无效: {status}; {ptx_size.value}")
        ptx = ctypes.create_string_buffer(ptx_size.value)
        status = nvrtc.nvrtcGetPTX(program, ptx)
        if status or b".entry verify" not in ptx.value:
            raise RuntimeError(f"CUDA NVRTC 编译产物缺少验收内核: {status}")
        return len(ptx.value)
    finally:
        if program.value and nvrtc is not None:
            destroy_status = nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
            if destroy_status:
                raise RuntimeError(f"CUDA NVRTC 释放编译任务失败: {destroy_status}")
        if dll_directory is not None:
            dll_directory.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=localize_console_message("校验发行包内全部固定版本 VC/UVR5 源码"))
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--cuda", metavar="MAJOR.MINOR", help=localize_console_message("额外验证包内 CUDA NVRTC 编译库"))
    args = parser.parse_args()
    count = validate_upstream_sources(args.project_root.resolve())
    print(f"已核对 VC/UVR5 固定源码，共 {count} 个文件")
    if args.cuda:
        ptx_bytes = validate_cuda_runtime(args.project_root.resolve(), args.cuda)
        print(f"CUDA NVRTC 实际编译通过，产物 {ptx_bytes} 字节")


if __name__ == "__main__":
    main()
