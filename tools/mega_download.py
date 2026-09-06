"""Download public MEGA links with pinned, author-published Megatools builds."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from tools.model_assets import verify_asset

VERSION = '1.11.5.20250706'
BUILDS = {
    'win64': ('zip', 'bd1269e50d9c45e369c14e287c1754c9506e1b11efcc0cd4bb95d460c9d782b5',
              'megatools.exe', '150089e479162b069a809c620925fecf7dae586169a90f404ab67ec3eb75829d'),
    'linux-x86_64': ('tar.gz', 'f6f6f22cb1d3c166c88d85eb669acfe7741d9ce28f7fb94693980bd37894832a',
                     'megatools', '7b9fdd85a608166012a4b163b56e95a60b1ff8886c28c3c5ce0980f42cff88be'),
}


def build_key() -> str | None:
    if platform.machine().lower() not in ('amd64', 'x86_64'):
        return None
    return 'win64' if sys.platform == 'win32' else 'linux-x86_64' if sys.platform.startswith('linux') else None


def download_supported() -> bool:
    return build_key() is not None or shutil.which('megatools') is not None


def prepare_megatools(root: Path) -> Path:
    key = build_key()
    if key is None:
        executable = shutil.which('megatools')
        if executable is None:
            raise RuntimeError('当前平台需要先安装 Megatools 才能下载 Mega 公开链接')
        return Path(executable)
    suffix, archive_hash, program, program_hash = BUILDS[key]
    name = f'megatools-{VERSION}-{key}'
    directory = Path(root) / 'assets' / 'tools' / name
    executable = directory / program
    documents = ('LICENSE.TXT', 'DEPS.TXT', 'CHECKSUMS.TXT') if suffix == 'zip' else ('LICENSE', 'DEPS', 'CHECKSUMS')
    if executable.exists():
        if not all((directory / filename).is_file() for filename in documents):
            raise RuntimeError('Megatools 安装不完整，缺少许可证或依赖说明')
        return verify_asset(executable, program_hash)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f'{name}.{suffix}'
    if not archive.exists():
        import requests
        partial = archive.with_name(archive.name + '.partial')
        with requests.get(f'https://xff.cz/megatools/builds/builds/{archive.name}',
                          stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
        verify_asset(partial, archive_hash)
        partial.rename(archive)
    verify_asset(archive, archive_hash)
    if suffix == 'zip':
        with zipfile.ZipFile(archive) as stream:
            contents = {filename: stream.read(f'{name}/{filename}') for filename in (*documents, program)}
    else:
        with tarfile.open(archive) as stream:
            contents = {}
            for filename in (*documents, program):
                member = next((m for m in stream.getmembers() if m.isfile() and Path(m.name).name == filename), None)
                if member is None:
                    raise RuntimeError(f'Megatools 发布包缺少必需文件：{filename}')
                contents[filename] = stream.extractfile(member).read()
    for filename, content in contents.items():
        (directory / filename).write_bytes(content)
    if suffix != 'zip':
        executable.chmod(0o755)
    return verify_asset(executable, program_hash)


def download_public_file(url: str, directory: Path, root: Path) -> Path:
    executable = prepare_megatools(root)
    directory.mkdir(parents=True, exist_ok=False)
    result = subprocess.run([str(executable), 'dl', '--no-progress', '--path', str(directory), url],
                            capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=1800,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
    if result.returncode:
        details = (result.stderr or result.stdout).strip().replace(url, '[Mega public link]')
        raise RuntimeError(f'Mega 公开链接下载失败：exit={result.returncode}；{details}')
    files = [path for path in directory.iterdir() if path.is_file()]
    if len(files) != 1 or files[0].stat().st_size == 0:
        raise RuntimeError('Mega 公开链接未返回唯一的非空模型文件')
    return files[0]
