"""HTTP downloads with validated ranges and atomic publication (RFC 9110)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from tqdm import tqdm

_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def download_http(url: str, destination: Path, description: str | None = None,
                  expected_sha256: str | None = None) -> Path:
    """Keep incomplete bytes in .part; existing published files survive failures."""
    global _LAST_REQUEST
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    metadata = partial.with_name(partial.name + '.json')
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValueError('Download URL must use HTTP or HTTPS')
    identity = hashlib.sha256(url.encode()).hexdigest()
    with _LOCK:
        previous = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
        offset = partial.stat().st_size if partial.exists() else 0
        validator = previous.get('etag')
        if previous.get('url_hash') != identity or not (expected_sha256 or validator):
            offset = 0
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
            if validator:
                headers['If-Range'] = validator
        if parsed.hostname == 'huggingface.co' and parsed.scheme == 'https':
            token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
            if token:
                headers['Authorization'] = f'Bearer {token}'
        delay = 2.0 - (time.monotonic() - _LAST_REQUEST)
        if delay > 0:
            time.sleep(delay)
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
                if response.status_code not in {200, 206}:
                    raise RuntimeError(f'Download failed: HTTP {response.status_code}; file={destination.name}')
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise RuntimeError('Download returned an unexpected Content-Encoding')
                length = response.headers.get('Content-Length')
                length = int(length) if length is not None else None
                total = length
                if response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match:
                        raise RuntimeError('Download returned an invalid Content-Range')
                    start, end, total = map(int, match.groups())
                    if start != offset or end < start or end != total - 1 or (length is not None and length != end - start + 1):
                        raise RuntimeError('Download range does not match the requested remaining bytes')
                    if validator and response.headers.get('ETag') not in {None, validator}:
                        raise RuntimeError('Download ETag changed during resume')
                else:
                    # Range ignored or If-Range changed: a 200 response is the full file.
                    offset = 0
                etag = response.headers.get('ETag')
                if etag and etag.startswith('W/'):
                    etag = None
                metadata.write_text(json.dumps({'url_hash': identity, 'etag': etag}), encoding='utf-8')
                with partial.open('ab' if offset else 'wb') as stream, tqdm(total=total, initial=offset, unit='B', unit_scale=True, desc=description or destination.name) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            progress.update(len(chunk))
                actual_size = partial.stat().st_size
                if actual_size == 0 or (total is not None and actual_size != total):
                    raise RuntimeError(f'Download size mismatch: expected={total}, actual={actual_size}')
                if expected_sha256:
                    from tools.model_assets import verify_asset
                    verify_asset(partial, expected_sha256)
                os.replace(partial, destination)
                metadata.unlink()
                return destination
        finally:
            _LAST_REQUEST = time.monotonic()
