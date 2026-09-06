# -*- coding: utf-8 -*-
"""Subprocess runner for vendored upstream RVC conversion."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.console_i18n import console_print as print
from lib.device import get_device
from lib.upstream_imports import activate_upstream_packages
from lib.upstream_audio_output import preserve_float_vc_output, preserve_unvoiced_f0
from infer.contracts import inspect_checkpoint, validate_state_dict, read_index, retrieve_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendored upstream RVC VC")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--vocals-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--f0-method", required=True)
    parser.add_argument("--pitch-shift", type=int, required=True)
    parser.add_argument("--index-path", default="")
    parser.add_argument("--index-rate", type=float, required=True)
    parser.add_argument("--rms-mix-rate", type=float, required=True)
    parser.add_argument("--protect", type=float, required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def configure_selected_device(config_module, selected_device):
    """Select upstream's precision profile before its Config singleton is constructed."""
    dtype, memory = torch.float32, 0.0
    if selected_device.type == "cuda" and torch.version.hip is None:
        index = selected_device.index if selected_device.index is not None else torch.cuda.current_device()
        actual, dtype, _sm, memory = config_module.get_device_dtype_sm(index)
        if actual.type != "cuda":
            raise RuntimeError(f"所选设备 {selected_device} 不满足固定版官方 RVC 的 GPU 要求")
        selected_device = actual
    config_module.infer_device = selected_device
    config_module.infer_dtype = dtype
    config_module.infer_gpu_mem = memory
    config_module.CUDA_GRAPH_AVAILABLE = config_module.configure_cuda_graph(selected_device)
    config = config_module.Config()
    config.device = selected_device
    config.dml = str(selected_device).startswith("privateuseone")
    if config.dtype != dtype or config.is_half != (dtype == torch.float16):
        raise RuntimeError("官方 RVC 设备与精度配置不一致")
    return config


def save_vc_result(result, output_path: Path) -> None:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(f"官方 VC 返回协议无效: {type(result).__name__}")
    info, payload = result
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise RuntimeError(f"官方 VC 未返回音频: {info}")
    sr, audio = payload
    if sr is None or audio is None:
        raise RuntimeError(f"官方 VC 失败: {info}")
    audio = np.asarray(audio)
    if not isinstance(sr, (int, np.integer)) or sr <= 0 or audio.ndim not in (1, 2) or not audio.size or not np.isfinite(audio).all():
        raise RuntimeError(f"官方 VC 返回无效采样率或音频: {info}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".wav", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        sf.write(temporary, audio, int(sr), subtype="FLOAT" if audio.dtype.kind == "f" else "PCM_16")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    selected_device = get_device(args.device)
    official_root = Path(args.official_root).resolve()
    if not (official_root / "infer" / "vc" / "modules.py").is_file():
        raise FileNotFoundError(f"新版官方 VC 模块不存在: {official_root}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    activate_upstream_packages(official_root)
    os.chdir(official_root)
    sys.argv = [sys.argv[0]]

    # Offline clips have changing lengths and each shape is consumed once.
    # Upstream's graph cache repeats inference during capture and keeps a
    # private memory pool for every shape. Select eager execution up front;
    # this changes neither model precision nor the inference/DSP algorithm.
    os.environ["RVC_CUDA_GRAPH"] = "0"
    import configs.config as config_module  # type: ignore
    config = configure_selected_device(config_module, selected_device)
    from infer.vc.modules import VC  # type: ignore
    vc = VC(config)
    vc.get_vc(args.sid)
    contract = inspect_checkpoint(vc.cpt, args.sid)
    validate_state_dict(vc.net_g, vc.cpt["weight"])
    if contract.uses_f0:
        preserve_unvoiced_f0(vc.pipeline)
    if args.index_rate > 0:
        preserve_float_vc_output(vc.pipeline,
            index_loader=partial(read_index, feature_dim=contract.feature_dim, min_vectors=8),
            index_retriever=retrieve_features)
    else:
        preserve_float_vc_output(vc.pipeline)

    speaker_count = int(vc.cpt["weight"]["emb_g.weight"].shape[0])
    if not 0 <= args.speaker_id < speaker_count:
        raise ValueError(f"说话人 ID 越界: id={args.speaker_id}, count={speaker_count}")
    started = time.perf_counter()
    result = vc.vc_single(
        args.speaker_id,
        args.vocals_path,
        args.pitch_shift,
        args.f0_method,
        args.index_path,
        args.index_rate,
        0,
        args.rms_mix_rate,
        args.protect,
    )
    save_vc_result(result, Path(args.output_path))
    print("转换成功。", flush=True)
    if args.index_path:
        print(f"索引：{args.index_path}", flush=True)
    print(f"转换及保存实测耗时：{time.perf_counter() - started:.2f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
