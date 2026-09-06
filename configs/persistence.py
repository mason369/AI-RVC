"""Validate before replacing configuration; preserve unrelated on-disk settings."""
import json
import os
import tempfile
import threading
from pathlib import Path

from configs.schema import validate_config

_LOCK = threading.Lock()


def update_config(path: Path, updates: dict, *, replace: bool = False) -> dict:
    # Replace the real file atomically, preserving container volume symlinks.
    path = Path(path).resolve()
    with _LOCK:
        current = json.loads(path.read_text(encoding='utf-8')) if path.exists() and not replace else {}
        updated = validate_config({**current, **updates})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(updated, stream, ensure_ascii=False, indent=4)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return updated
