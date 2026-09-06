"""Content inventory for portable releases; changed or missing files fail verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.console_i18n import console_print as print


def inventory(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / "PACKAGE-MANIFEST.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if not files:
        raise ValueError("发行目录为空")
    return {"files": files, "count": len(files), "bytes": sum(item["bytes"] for item in files.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成或核对发行包完整性清单")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = args.directory / "PACKAGE-MANIFEST.json"
    actual = inventory(args.directory)
    if args.verify:
        if json.loads(manifest.read_text(encoding="utf-8")) != actual:
            raise RuntimeError("发行包内容与清单不一致，禁止发布")
    else:
        manifest.write_text(json.dumps(actual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已核对 {actual['count']} 个文件，共 {actual['bytes']} 字节")


if __name__ == "__main__":
    main()
