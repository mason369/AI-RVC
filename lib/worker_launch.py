"""Explicit worker dispatch for Python and PyInstaller executables."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKERS = {
    "vc": "infer/official_upstream_runner.py",
    "uvr5": "infer/official_upstream_uvr_runner.py",
    "mcp-convert": "rvc_mcp/worker.py",
    "mcp-server": "rvc_mcp/server.py",
}


def build_worker_command(name: str, *arguments: str) -> list[str]:
    if name not in WORKERS:
        raise ValueError(f"未知内部工作进程: {name!r}")
    if getattr(sys, "frozen", False):
        return [sys.executable, "--internal-worker", name, *arguments]
    if name.startswith("mcp-"):
        module = "rvc_mcp.worker" if name == "mcp-convert" else "rvc_mcp.server"
        return [sys.executable, "-m", module, *arguments]
    return [sys.executable, str(ROOT / WORKERS[name]), *arguments]


def dispatch_worker(name: str, arguments: list[str]) -> None:
    if name not in WORKERS:
        raise ValueError(f"未知内部工作进程: {name!r}")
    source = ROOT / WORKERS[name]
    if not source.is_file():
        raise FileNotFoundError(f"内部工作进程源码缺失: {source}")
    sys.argv = [str(source), *arguments]
    runpy.run_path(str(source), run_name="__main__")
