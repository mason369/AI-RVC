from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import MutableMapping, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent


def get_runtime_root(root_dir: Path | str | None = None) -> Path:
    if root_dir is not None:
        return Path(root_dir).expanduser()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return ROOT_DIR


def _normalize_path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _iter_ffmpeg_bin_candidates(root_dir: Path) -> list[Path]:
    candidates = [
        root_dir / "tools" / "ffmpeg" / "bin",
        root_dir / "tools" / "ffmpeg",
    ]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        meipass_root = Path(meipass)
        candidates.extend(
            [
                meipass_root / "tools" / "ffmpeg" / "bin",
                meipass_root / "tools" / "ffmpeg",
            ]
        )

    try:
        import imageio_ffmpeg

        candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve().parent)
    except Exception:
        pass

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize_path_key(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def get_ffmpeg_bin_dir(root_dir: Path | str | None = None) -> Optional[Path]:
    runtime_root = get_runtime_root(root_dir)
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    for candidate in _iter_ffmpeg_bin_candidates(runtime_root):
        if (candidate / ffmpeg_name).exists():
            return candidate

    found = shutil.which("ffmpeg")
    if found:
        return Path(found).resolve().parent

    return None


def _check_executable_runs(executable: Path, label: str) -> None:
    try:
        result = subprocess.run(
            [str(executable), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        raise RuntimeError(f"{label} 无法启动: {executable} ({exc})") from exc

    if result.returncode != 0:
        details = "\n".join(
            part.strip()
            for part in (result.stdout, result.stderr)
            if part and part.strip()
        )
        if not details:
            details = f"退出码: {result.returncode}"
        raise RuntimeError(f"{label} 无法正常运行: {executable}\n{details}")


def configure_ffmpeg_runtime(
    root_dir: Path | str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Optional[Path]:
    env = os.environ if env is None else env
    bin_dir = get_ffmpeg_bin_dir(root_dir=root_dir)
    if bin_dir is None:
        return None

    bin_dir_str = str(bin_dir)
    current_path = env.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    normalized_entries = {_normalize_path_key(entry) for entry in path_entries}
    if _normalize_path_key(bin_dir_str) not in normalized_entries:
        env["PATH"] = bin_dir_str if not current_path else bin_dir_str + os.pathsep + current_path

    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_path = bin_dir / ffmpeg_name
    if ffmpeg_path.exists():
        _check_executable_runs(ffmpeg_path, "ffmpeg")
        env.setdefault("FFMPEG_BINARY", str(ffmpeg_path))
        env.setdefault("IMAGEIO_FFMPEG_EXE", str(ffmpeg_path))

    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    ffprobe_path = bin_dir / ffprobe_name
    if ffprobe_path.exists():
        _check_executable_runs(ffprobe_path, "ffprobe")
        env.setdefault("FFPROBE_BINARY", str(ffprobe_path))
    else:
        resolved_ffprobe = shutil.which("ffprobe", path=env.get("PATH", ""))
        if resolved_ffprobe is None:
            raise RuntimeError("未找到 ffprobe。请安装 ffmpeg/ffprobe，或将 ffprobe 放入 tools/ffmpeg/bin。")
        _check_executable_runs(Path(resolved_ffprobe), "ffprobe")
        env.setdefault("FFPROBE_BINARY", str(Path(resolved_ffprobe)))

    return bin_dir
