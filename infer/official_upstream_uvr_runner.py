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
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--save-root-vocal", required=True)
    parser.add_argument("--save-root-ins", required=True)
    parser.add_argument("--agg", type=int, required=True)
    parser.add_argument("--format", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    official_root = repo_root / "_official_rvc"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sys.path.insert(0, str(official_root))
    os.chdir(official_root)
    sys.argv = [sys.argv[0]]

    from infer.modules.uvr5.modules import uvr  # type: ignore

    for progress_info in uvr(
        args.model_name,
        args.input_dir,
        args.save_root_vocal,
        [],
        args.save_root_ins,
        args.agg,
        args.format,
    ):
        if progress_info:
            print(progress_info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
