"""Require both pinned upstream runtimes before and after portable bundling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import upstream_runtime
from lib.console_i18n import console_print as print
from lib.console_i18n import localize_console_message


def validate_upstream_sources(project_root: Path) -> int:
    manifest_path = Path(upstream_runtime.__file__).with_name("upstream_source_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for capability in upstream_runtime.REVISIONS:
        root = upstream_runtime.runtime_root(project_root, capability)
        problems = upstream_runtime._packaged_source_problems(root, capability)
        if problems:
            raise RuntimeError(f"发行资源 {capability} 源码校验失败: " + "；".join(problems))
        checked += len(manifest[capability]["files"])
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=localize_console_message("校验发行包内全部固定版本 VC/UVR5 源码"))
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    count = validate_upstream_sources(args.project_root.resolve())
    print(f"已核对 VC/UVR5 固定源码，共 {count} 个文件")


if __name__ == "__main__":
    main()
