# -*- coding: utf-8 -*-
"""
模型下载工具 - 自动从 Hugging Face 下载所需模型
"""
import os
import hashlib
import shutil
import subprocess
import requests
import logging as _logging
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.console_i18n import console_print as print
from tools import upstream_runtime

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

# Pin base assets to one upstream commit, including exact bytes and LFS SHA-256.
from tools.base_model_manifest import REVISION as BASE_REVISION, FILES as BASE_FILES
for _model in MODELS.values():
    _filename = _model['url'].split('/resolve/main/')[1]
    _asset = BASE_FILES[_filename]
    _model.update(sha256=_asset['sha256'], size_bytes=_asset['size'])
    _model['size_mb'] = round(_asset['size'] / 1024**2, 1)
    _model['url'] = _model['url'].replace('/resolve/main/', f'/resolve/{BASE_REVISION}/')

# 必需模型列表
REQUIRED_MODELS = ["hubert_base.pt", "rmvpe.pt", "HP2_all_vocals.pth"]

# Mature DeEcho / DeReverb models downloaded separately
MATURE_DEECHO_MODELS = [
    "VR-DeEchoDeReverb.pth",
    "onnx_dereverb_By_FoxJoy/vocals.onnx",
    "VR-DeEchoNormal.pth",
    "VR-DeEchoAggressive.pth",
]

UPSTREAM_RVC_REPO_URL = upstream_runtime.REPOSITORY
UPSTREAM_RVC_REVISION = upstream_runtime.REVISIONS["vc"]
UPSTREAM_RVC_DIR = f"_official_rvc_runtime/{UPSTREAM_RVC_REVISION}"
UPSTREAM_RVC_REQUIRED_FILES = list(upstream_runtime.REQUIRED_FILES["vc"])


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def download_file(url: str, dest_path: Path, desc: str = None, *, expected_sha256: str = None) -> bool:
    """Download to a partial file; propagate HTTP/integrity failures to callers."""
    from tools.http_download import download_http
    download_http(url, dest_path, desc, expected_sha256)
    return True


def check_model(name: str) -> bool:
    """Check the pinned size and SHA-256; a partial file is never installed."""
    if name not in MODELS:
        return False
    info = MODELS[name]
    path = get_project_root() / info['path']
    if not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != info['size_bytes']:
        return False
    # File timestamps can repeat on coarse-resolution filesystems. Re-read the
    # content so a same-size overwrite cannot inherit an earlier valid result.
    from tools.model_assets import file_sha256
    return file_sha256(path) == info['sha256']


def download_model(name: str) -> bool:
    """Prepare one verified asset without overwriting corrupt existing files."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    info = MODELS[name]
    path = get_project_root() / info['path']
    if check_model(name):
        print(f"[OK] {name}")
        return True
    if path.exists():
        raise RuntimeError(f"Model integrity check failed: {path}; preserve or remove this file before downloading again")
    return download_file(info['url'], path, name, expected_sha256=info['sha256'])


def _download_hf_file(repo_id: str, filename: str, model_dir: Path) -> Path:
    """Download a Hugging Face file into a local directory."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("请先安装 huggingface_hub，才能下载默认分离模型") from exc

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"下载默认分离模型失败: repo={repo_id}, file={filename}"
        ) from exc
    return Path(downloaded)


def get_default_separator_asset_paths(
    root_dir: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    """Return required files for the current default separation route."""
    from infer.separator import (
        KARAOKE_SOTA_MODELS,
        LEAP_XE_VOCALS_MODEL,
        LEAP_INSTRUMENTAL_MODEL,
        ROFORMER_DEREVERB_DEFAULT_MODEL,
        _CUSTOM_AUDIO_SEPARATOR_MODELS,
    )

    project_root = Path(root_dir) if root_dir is not None else get_project_root()
    model_dir = project_root / "assets" / "separator_models"
    leap_spec = _CUSTOM_AUDIO_SEPARATOR_MODELS[LEAP_XE_VOCALS_MODEL]
    assets: Dict[str, List[Path]] = {
        "Leap XE 90 vocals": [
            model_dir / leap_spec["model_filename"],
            model_dir / leap_spec["config_filename"],
            *(
                [model_dir / leap_spec["runtime_config_filename"]]
                if "runtime_config_filename" in leap_spec
                else []
            ),
        ],
    }
    for label, model_name in (
        ("Leap Instrumental 62 bands", LEAP_INSTRUMENTAL_MODEL),
        ("RoFormer De-Reverb Stereo", ROFORMER_DEREVERB_DEFAULT_MODEL),
    ):
        spec = _CUSTOM_AUDIO_SEPARATOR_MODELS[model_name]
        asset_dir = model_dir / spec["local_subdir"]
        assets[label] = [asset_dir / spec[key] for key in (
            "model_filename", "config_filename", "runtime_config_filename",
        )]
    for model_name in KARAOKE_SOTA_MODELS:
        spec = _CUSTOM_AUDIO_SEPARATOR_MODELS[model_name]
        assets[f"MVSep 9205 {model_name}"] = [
            model_dir / spec["model_filename"],
            model_dir / spec["config_filename"],
        ]
    return assets


def get_missing_default_separator_model_files(
    root_dir: Optional[Path] = None,
) -> List[str]:
    """List missing files for the default hybrid SOTA separator route."""
    project_root = Path(root_dir) if root_dir is not None else get_project_root()
    missing = []
    for label, paths in get_default_separator_asset_paths(project_root).items():
        for path in paths:
            if not path.exists():
                try:
                    display_path = path.relative_to(project_root)
                except ValueError:
                    display_path = path
                missing.append(f"{label}: {display_path}")
    return missing


def check_required_default_separator_models(root_dir: Optional[Path] = None) -> bool:
    """Return whether the current default separation models are present."""
    return not get_missing_default_separator_model_files(root_dir)


def check_default_separator_models(root_dir: Optional[Path] = None) -> Dict[str, bool]:
    """Return status by default separator model group."""
    project_root = Path(root_dir) if root_dir is not None else get_project_root()
    return {
        label: all(path.exists() for path in paths)
        for label, paths in get_default_separator_asset_paths(project_root).items()
    }


def download_default_separator_models(root_dir: Optional[Path] = None) -> bool:
    """Download the default hybrid SOTA, MVSep 9205, and De-Reverb assets."""
    from infer.separator import (
        KARAOKE_SOTA_MODELS,
        LEAP_XE_VOCALS_MODEL,
        LEAP_INSTRUMENTAL_MODEL,
        ROFORMER_DEREVERB_DEFAULT_MODEL,
        _install_custom_audio_separator_models,
    )

    project_root = Path(root_dir) if root_dir is not None else get_project_root()
    model_dir = project_root / "assets" / "separator_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("准备默认分离模型...")
    print("=" * 50)

    success = True
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        print(f"[ERROR] audio-separator 不可用，无法下载 Leap XE / MVSep 9205 / De-Reverb: {exc}")
        return False

    separator = Separator(
        log_level=_logging.WARNING,
        info_only=True,
        output_dir=str(project_root / "temp" / "separator_download"),
        model_file_dir=str(model_dir),
    )
    _install_custom_audio_separator_models(separator)
    for model_name in [
        LEAP_XE_VOCALS_MODEL,
        LEAP_INSTRUMENTAL_MODEL,
        *KARAOKE_SOTA_MODELS,
        ROFORMER_DEREVERB_DEFAULT_MODEL,
    ]:
        try:
            separator.download_model_files(model_name)
            print(f"[OK] 分离模型: {model_name}")
        except Exception as exc:
            print(f"[ERROR] 分离模型下载失败: {model_name}; {exc}")
            success = False

    missing = get_missing_default_separator_model_files(project_root)
    if missing:
        print("[ERROR] 默认分离模型仍有缺失:")
        for item in missing:
            print(f"  - {item}")
        success = False

    return success


def get_upstream_rvc_root(root_dir: Optional[Path] = None, *, capability: str = "vc") -> Path:
    """Return the isolated pinned source directory for the requested capability."""
    project_root = Path(root_dir) if root_dir is not None else get_project_root()
    return upstream_runtime.runtime_root(project_root, capability)


def get_missing_upstream_rvc_files(root_dir: Optional[Path] = None) -> List[str]:
    """Report source/API/version and Transformers HuBERT readiness problems."""
    project_root = Path(root_dir) if root_dir is not None else get_project_root()
    official_root = get_upstream_rvc_root(project_root)
    return upstream_runtime.source_problems(official_root) + upstream_runtime.hubert_problems(project_root)


def check_upstream_rvc_tree(root_dir: Optional[Path] = None) -> bool:
    """Return whether the vendored official RVC tree is ready for default VC."""
    return not get_missing_upstream_rvc_files(root_dir)


def ensure_upstream_rvc_tree(
    root_dir: Optional[Path] = None,
    *,
    clone_timeout_sec: int = 900,
    capability: str = "vc",
) -> Path:
    """
    Ensure the vendored official RVC source tree exists.

    This is required by the default quality route. It does not fall back to
    another VC backend: if the tree cannot be prepared, the caller should stop.
    """
    project_root = Path(root_dir) if root_dir is not None else get_project_root()
    official_root = upstream_runtime.ensure_source(project_root, capability, clone_timeout_sec)
    if capability == "vc":
        upstream_runtime.ensure_transformers_hubert(project_root)
    print(f"[OK] 固定版本官方 {capability} 已就绪: {official_root}")
    return official_root


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

    if not download_default_separator_models():
        success = False

    try:
        ensure_upstream_rvc_tree()
    except Exception as e:
        print(f"[ERROR] 内置官方 RVC 准备失败: {e}")
        success = False

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

    if not download_default_separator_models():
        success = False

    try:
        ensure_upstream_rvc_tree()
    except Exception as e:
        print(f"[ERROR] 内置官方 RVC 准备失败: {e}")
        success = False

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

    print("=" * 50)
    print("默认分离模型状态")
    print("=" * 50)
    separator_status = check_default_separator_models()
    for name, exists in separator_status.items():
        mark = "OK" if exists else "MISSING"
        print(f"  {mark} {name}")
    missing_separator = get_missing_default_separator_model_files()
    if missing_separator:
        print("  缺失文件:")
        for item in missing_separator:
            print(f"    - {item}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RVC 模型下载工具")
    parser.add_argument("--check", action="store_true", help="检查模型状态")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--model", type=str, help="下载指定模型")
    parser.add_argument("--official-rvc", action="store_true", help="准备内置官方 RVC 源码")
    parser.add_argument("--separator", action="store_true", help="准备默认混合 SOTA / MVSep 分离模型")

    args = parser.parse_args()

    if args.check:
        print_model_status()
        missing_official = get_missing_upstream_rvc_files()
        if missing_official:
            print("[MISSING] 内置官方 RVC: " + ", ".join(missing_official))
        else:
            print("[OK] 内置官方 RVC")
        ready = all(check_model(name) for name in REQUIRED_MODELS) and all(check_default_separator_models().values()) and not missing_official
        sys.exit(0 if ready else 1)
    elif args.official_rvc:
        ensure_upstream_rvc_tree()
    elif args.separator:
        sys.exit(0 if download_default_separator_models() else 1)
    elif args.model:
        sys.exit(0 if download_model(args.model) else 1)
    elif args.all:
        sys.exit(0 if download_all_models() else 1)
    else:
        sys.exit(0 if download_required_models() else 1)
