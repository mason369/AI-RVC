"""Prepare the two executables used by local and CI portable builds."""
import argparse
import os
import shutil
import stat
from pathlib import Path

from lib.ffmpeg_runtime import _check_executable_runs


def bundle_ffmpeg(destination: Path, staged: Path | None = None) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for tool in ("ffmpeg", "ffprobe"):
        name = tool + (".exe" if os.name == "nt" else "")
        if staged is not None:
            source = staged / name
        elif tool == "ffmpeg":
            import imageio_ffmpeg
            source = Path(imageio_ffmpeg.get_ffmpeg_exe())
        else:
            executable = shutil.which(name)
            if not executable:
                raise FileNotFoundError(name)
            source = Path(executable)

        if os.name == "nt":
            choco = Path(os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey"))
            if source.parent.resolve() == (choco / "bin").resolve():
                source = choco / "lib/ffmpeg/tools/ffmpeg/bin" / name
        target = destination / name
        if source.resolve(strict=True) != target.resolve():
            shutil.copy2(source, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        _check_executable_runs(target, tool)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(bundle_ffmpeg(parser.parse_args().destination))
