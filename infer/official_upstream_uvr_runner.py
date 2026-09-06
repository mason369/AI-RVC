# -*- coding: utf-8 -*-
"""Subprocess runner for vendored upstream UVR5 separation."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendored upstream UVR5 separation")
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--save-root-vocal", required=True)
    parser.add_argument("--save-root-ins", required=True)
    parser.add_argument("--agg", type=int, required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.format != "wav" or not 0 <= args.agg <= 20:
        raise ValueError("UVR5 只接受 WAV 浮点输出，agg 必须在 0～20 之间")
    application_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(application_root))
    from lib.upstream_audio_output import preserve_float_uvr_output
    from lib.device import get_device, supports_fp16
    from lib.upstream_imports import activate_upstream_packages
    from infer.separator import _ensure_separator_pcm_wav
    device = get_device(args.device)
    is_half = supports_fp16(device)
    # Call the pinned model directly: upstream's batch wrapper catches model
    # errors and retries via PCM16. Neither behavior is part of this contract.
    import tempfile
    input_workspace = tempfile.TemporaryDirectory(prefix="rvc_uvr_float_")
    input_files = list(Path(args.input_dir).iterdir())
    if not input_files or any(not item.is_file() for item in input_files):
        raise ValueError("UVR5 输入目录必须包含音频文件")
    normalized_inputs = []
    for item in input_files:
        normalized = _ensure_separator_pcm_wav(str(item), Path(input_workspace.name) / "decode")
        normalized_inputs.append(normalized)
    official_root = Path(args.official_root).resolve()
    if not (official_root / "infer/modules/uvr5/modules.py").is_file():
        raise FileNotFoundError(f"固定版 UVR5 模块不存在: {official_root}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    activate_upstream_packages(official_root)
    os.chdir(official_root)
    sys.argv = [sys.argv[0]]
    from infer.modules.uvr5 import vr
    preserve_float_uvr_output(vr)
    model_class = vr.AudioPreDeEcho if "DeEcho" in args.model_name else vr.AudioPre
    model_path = official_root / "assets/uvr5_weights" / f"{args.model_name}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"UVR5 VR 推理权重不存在：{model_path}")
    model = model_class(agg=args.agg, model_path=str(model_path), device=device, is_half=is_half)
    for normalized in normalized_inputs:
        model._path_audio_(normalized, args.save_root_ins, args.save_root_vocal, args.format,
                           is_hp3="HP3" in args.model_name)
    input_workspace.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
