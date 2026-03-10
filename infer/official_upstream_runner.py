# -*- coding: utf-8 -*-
"""Subprocess runner for vendored upstream RVC conversion."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vendored upstream RVC VC")
    parser.add_argument("--sid", required=True)
    parser.add_argument("--vocals-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--f0-method", required=True)
    parser.add_argument("--pitch-shift", type=int, required=True)
    parser.add_argument("--index-path", default="")
    parser.add_argument("--index-rate", type=float, required=True)
    parser.add_argument("--filter-radius", type=int, required=True)
    parser.add_argument("--rms-mix-rate", type=float, required=True)
    parser.add_argument("--protect", type=float, required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    official_root = repo_root / "_official_rvc"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sys.path.insert(0, str(official_root))
    os.chdir(official_root)
    sys.argv = [sys.argv[0]]

    from configs.config import Config  # type: ignore
    from infer.modules.vc.modules import VC  # type: ignore

    config = Config()
    vc = VC(config)
    vc.get_vc(args.sid)

    spk_max = 1
    try:
        if getattr(vc, "cpt", None) is not None:
            spk_max = int(vc.cpt["config"][-3])
    except Exception:
        spk_max = 1
    spk_id = max(0, min(max(1, spk_max) - 1, int(args.speaker_id)))

    info, (sr, audio) = vc.vc_single(
        spk_id,
        args.vocals_path,
        args.pitch_shift,
        None,
        args.f0_method,
        args.index_path,
        "",
        args.index_rate,
        args.filter_radius,
        0,
        args.rms_mix_rate,
        args.protect,
    )
    if sr is None or audio is None:
        raise RuntimeError(info)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)

    match = re.search(
        r"Time:\s*npy:\s*([0-9.]+)s,\s*f0:\s*([0-9.]+)s,\s*infer:\s*([0-9.]+)s\.",
        str(info),
        re.IGNORECASE | re.MULTILINE,
    )
    print("转换成功。", flush=True)
    if args.index_path:
        print(f"索引：{args.index_path}", flush=True)
    if match:
        print(
            f"耗时：npy {match.group(1)}s，f0 {match.group(2)}s，推理 {match.group(3)}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
