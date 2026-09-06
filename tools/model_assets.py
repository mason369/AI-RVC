"""Pinned model assets: verify content and expose download failures."""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

_DOWNLOAD_LOCK = threading.Lock()
_LAST_DOWNLOAD = 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, expected_sha256: str) -> Path:
    path = Path(path)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"模型资源 SHA-256 不匹配: {path}; expected={expected_sha256}, actual={actual}。"
            "已停止加载，请检查该文件；不会覆盖现有资源或自动更换模型。"
        )
    return path


def download_pinned_asset(
    repo_id: str, filename: str, local_dir: Path, revision: str, sha256: str,
) -> Path:
    """Use an immutable HF revision; never silently replace a corrupt local file."""
    global _LAST_DOWNLOAD
    from huggingface_hub import hf_hub_download

    if len(revision) != 40 or len(sha256) != 64:
        raise ValueError("固定模型资源必须提供完整提交号和 SHA-256")
    destination = Path(local_dir) / filename
    with _DOWNLOAD_LOCK:
        if destination.is_file():
            return verify_asset(destination, sha256)
        delay = 2.0 - (time.monotonic() - _LAST_DOWNLOAD)
        if delay > 0:
            time.sleep(delay)
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision,
                local_dir=str(local_dir),
            )
        except Exception as exc:
            raise RuntimeError(
                f"固定模型下载失败: {repo_id}@{revision}/{filename}: {exc}"
            ) from exc
        finally:
            _LAST_DOWNLOAD = time.monotonic()
        return verify_asset(Path(downloaded), sha256)
