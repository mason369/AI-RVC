# -*- coding: utf-8 -*-
"""Core library exports without eager optional-dependency imports."""

from importlib import import_module
from typing import Any

__all__ = ["load_audio", "save_audio", "get_device", "get_device_info", "empty_device_cache", "supports_fp16"]

_EXPORT_MODULES = {
    "load_audio": "lib.audio",
    "save_audio": "lib.audio",
    "get_device": "lib.device",
    "get_device_info": "lib.device",
    "empty_device_cache": "lib.device",
    "supports_fp16": "lib.device",
}


def __getattr__(name: str) -> Any:
    """Load public audio/device helpers only when the caller requests them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
