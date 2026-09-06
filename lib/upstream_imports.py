"""Bind pinned upstream packages without removing bundled dependency paths."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def activate_upstream_packages(root: Path) -> None:
    """Use the vendored packages inside an isolated worker, including frozen apps."""
    root = root.resolve()
    names = ("configs", "i18n", "infer", "tools")
    specs = {}
    for name in names:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"固定版官方源码包不存在: {directory}")
        initializer = directory / "__init__.py"
        if initializer.is_file():
            spec = importlib.util.spec_from_file_location(
                name, initializer, submodule_search_locations=[str(directory)])
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载固定版官方源码包: {directory}")
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(directory)]
        specs[name] = spec

    # A later regular package can override an earlier namespace package. Bind
    # these names explicitly rather than removing _MEIPASS (which also owns
    # PyInstaller's finder for every bundled third-party module).
    for name in list(sys.modules):
        if name.split(".", 1)[0] in names:
            del sys.modules[name]
    for name, spec in specs.items():
        sys.modules[name] = importlib.util.module_from_spec(spec)
    for name, spec in specs.items():
        if spec.origin is not None:
            spec.loader.exec_module(sys.modules[name])
    sys.path.insert(0, str(root))
