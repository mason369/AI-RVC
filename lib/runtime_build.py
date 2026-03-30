from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict


ROOT_DIR = Path(__file__).resolve().parent.parent
WATCHED_FILES = (
    "run.py",
    "app.py",
    "lib/ffmpeg_runtime.py",
    "ui/app.py",
    "infer/cover_pipeline.py",
    "infer/official_adapter.py",
    "infer/quality_policy.py",
)


def _existing_watched_files() -> list[Path]:
    files: list[Path] = []
    for relative_path in WATCHED_FILES:
        path = ROOT_DIR / relative_path
        if path.exists():
            files.append(path)
    return files


def _compute_runtime_build_info() -> Dict[str, str]:
    files = _existing_watched_files()
    if not files:
        return {
            "timestamp": "unknown",
            "source": "unknown",
        }

    latest = max(files, key=lambda path: path.stat().st_mtime)
    timestamp = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    try:
        source = latest.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        source = str(latest)
    return {
        "timestamp": timestamp,
        "source": source,
    }


RUNTIME_BUILD_INFO = _compute_runtime_build_info()


def get_runtime_build_label() -> str:
    info = RUNTIME_BUILD_INFO
    return f"当前运行代码标记: {info['timestamp']} ({info['source']})"


def get_runtime_build_short_label() -> str:
    info = RUNTIME_BUILD_INFO
    return f"{info['timestamp']} ({info['source']})"
