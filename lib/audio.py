# -*- coding: utf-8 -*-
"""
音频处理模块 - 加载、保存和处理音频文件
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """
    加载音频文件并重采样

    Args:
        path: 音频文件路径
        sr: 目标采样率 (默认 16000)

    Returns:
        np.ndarray: 音频数据 (float32, 单声道)
    """
    audio, orig_sr = librosa.load(path, sr=None, mono=True)

    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    return audio.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sr: int = 48000):
    """
    保存音频到文件

    Args:
        path: 输出文件路径
        audio: 音频数据
        sr: 采样率 (默认 48000)
    """
    # 确保音频在 [-1, 1] 范围内
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr)


def soft_clip(
    audio: np.ndarray,
    threshold: float = 0.9,
    ceiling: float = 0.99,
) -> np.ndarray:
    """
    使用平滑软削波抑制峰值，尽量保留主体响度。

    Args:
        audio: 输入音频
        threshold: 开始压缩的阈值
        ceiling: 软削波上限

    Returns:
        np.ndarray: 处理后的音频
    """
    audio = np.asarray(audio, dtype=np.float32)

    if threshold <= 0:
        raise ValueError("threshold 必须大于 0")
    if ceiling <= threshold:
        raise ValueError("ceiling 必须大于 threshold")

    result = audio.copy()
    abs_audio = np.abs(result)
    mask = abs_audio > threshold
    if not np.any(mask):
        return result

    overshoot = (abs_audio[mask] - threshold) / (ceiling - threshold + 1e-8)
    compressed = threshold + (ceiling - threshold) * np.tanh(overshoot)
    result[mask] = np.sign(result[mask]) * compressed
    return result.astype(np.float32, copy=False)


def soft_clip_array(
    audio: np.ndarray,
    threshold: float = 0.9,
    ceiling: float = 0.99,
) -> np.ndarray:
    """软削波数组版本，支持单声道/多声道。"""
    return soft_clip(audio, threshold=threshold, ceiling=ceiling)


def get_audio_info(path: str) -> dict:
    """
    获取音频文件信息

    Args:
        path: 音频文件路径

    Returns:
        dict: 音频信息
    """
    info = sf.info(path)
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format
    }


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    音频响度归一化

    Args:
        audio: 输入音频
        target_db: 目标响度 (dB)

    Returns:
        np.ndarray: 归一化后的音频
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)


def trim_silence(audio: np.ndarray, sr: int = 16000,
                 top_db: int = 30) -> np.ndarray:
    """
    去除音频首尾静音

    Args:
        audio: 输入音频
        sr: 采样率
        top_db: 静音阈值 (dB)

    Returns:
        np.ndarray: 去除静音后的音频
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed
