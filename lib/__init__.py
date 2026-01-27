# -*- coding: utf-8 -*-
"""
核心库模块
"""
from .audio import load_audio, save_audio
from .device import get_device, get_device_info

__all__ = ["load_audio", "save_audio", "get_device", "get_device_info"]
