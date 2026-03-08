# -*- coding: utf-8 -*-
"""
混音模块 - 人声与伴奏混合
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional

from lib.audio import soft_clip_array

try:
    from lib.logger import log
except ImportError:
    log = None

try:
    from pedalboard import Pedalboard, Reverb, Compressor, Gain
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False


def _probe_sample_rate(path: str, fallback: int = 44100) -> int:
    """Probe sample rate from file metadata."""
    try:
        return int(sf.info(path).samplerate)
    except Exception:
        return int(fallback)


def load_audio_for_mix(path: str, target_sr: Optional[int] = None) -> tuple:
    """
    加载音频用于混音。

    Args:
        path: 音频路径
        target_sr: 目标采样率；为 None 时保持原始采样率

    Returns:
        tuple: (audio_data, sample_rate)
    """
    if log:
        log.detail(f"加载音频: {Path(path).name}")

    audio, sr = librosa.load(path, sr=target_sr, mono=False)

    if audio.ndim == 1:
        audio = np.stack([audio, audio])
        if log:
            log.detail("单声道已扩展为双声道")

    if log:
        log.detail(f"音频形状: {audio.shape}, 采样率: {sr}Hz")

    return audio, sr


def apply_reverb(
    audio: np.ndarray,
    sr: int,
    room_size: float = 0.3,
    wet_level: float = 0.2,
) -> np.ndarray:
    """对人声应用混响效果。"""
    if not PEDALBOARD_AVAILABLE:
        if log:
            log.warning("Pedalboard 不可用，跳过混响处理")
        return audio

    if log:
        log.detail(f"应用混响: room_size={room_size}, wet_level={wet_level}")

    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    board = Pedalboard([
        Reverb(room_size=room_size, wet_level=wet_level, dry_level=1.0 - wet_level)
    ])
    processed = board(audio, sr)

    if log:
        log.detail("混响处理完成")

    return processed


def adjust_audio_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """将音频裁切/补零到目标长度。"""
    current_length = audio.shape[-1]

    if current_length == target_length:
        return audio
    if current_length > target_length:
        return audio[..., :target_length]

    pad_amount = target_length - current_length
    if audio.ndim == 1:
        return np.pad(audio, (0, pad_amount))
    return np.pad(audio, ((0, 0), (0, pad_amount)))


def mix_vocals_and_accompaniment(
    vocals_path: str,
    accompaniment_path: str,
    output_path: str,
    vocals_volume: float = 1.0,
    accompaniment_volume: float = 1.0,
    reverb_amount: float = 0.0,
    target_sr: Optional[int] = None,
) -> str:
    """
    混合人声和伴奏。

    Args:
        vocals_path: 人声音频路径
        accompaniment_path: 伴奏音频路径
        output_path: 输出路径
        vocals_volume: 人声音量 (0-2)
        accompaniment_volume: 伴奏音量 (0-2)
        reverb_amount: 人声混响量 (0-1)
        target_sr: 目标采样率；None 时自动采用两轨中更高采样率

    Returns:
        str: 输出文件路径
    """
    if target_sr is None or target_sr <= 0:
        vocals_sr = _probe_sample_rate(vocals_path)
        accompaniment_sr = _probe_sample_rate(accompaniment_path)
        target_sr = max(vocals_sr, accompaniment_sr)

    if log:
        log.progress("开始混音处理...")
        log.audio(f"人声文件: {Path(vocals_path).name}")
        log.audio(f"伴奏文件: {Path(accompaniment_path).name}")
        log.config(f"人声音量: {vocals_volume}, 伴奏音量: {accompaniment_volume}")
        log.config(f"混响量: {reverb_amount}, 目标采样率: {target_sr}Hz")

    if log:
        log.detail("加载人声音频...")
    vocals, sr = load_audio_for_mix(vocals_path, target_sr)

    if log:
        log.detail("加载伴奏音频...")
    accompaniment, _ = load_audio_for_mix(accompaniment_path, target_sr)

    if reverb_amount > 0 and PEDALBOARD_AVAILABLE:
        if log:
            log.progress("应用人声混响...")
        vocals = apply_reverb(vocals, sr, room_size=0.4, wet_level=reverb_amount)
    elif reverb_amount > 0 and log:
        log.warning("Pedalboard 不可用，跳过混响")

    vocals = soft_clip_array(vocals * vocals_volume, threshold=0.85, ceiling=0.95)
    accompaniment = soft_clip_array(
        accompaniment * accompaniment_volume,
        threshold=0.85,
        ceiling=0.95,
    )

    vocals_len = vocals.shape[-1]
    accompaniment_len = accompaniment.shape[-1]
    target_len = max(vocals_len, accompaniment_len)

    if target_len <= 0:
        raise ValueError("混音失败：音频长度为 0")

    if log:
        log.detail(f"人声长度: {vocals_len}, 伴奏长度: {accompaniment_len}")
        if vocals_len != accompaniment_len:
            log.detail(f"长度不一致，已补齐到最长长度: {target_len}")

    vocals = adjust_audio_length(vocals, target_len)
    accompaniment = adjust_audio_length(accompaniment, target_len)

    if log:
        log.progress("混合音轨...")
    mixed = vocals + accompaniment

    max_val = float(np.max(np.abs(mixed)))
    if log:
        log.detail(f"混合后峰值: {max_val:.4f}")

    mixed = soft_clip_array(mixed, threshold=0.90, ceiling=0.98)
    if log:
        final_peak = float(np.max(np.abs(mixed)))
        log.detail(f"软削波后峰值: {final_peak:.4f}")

    if mixed.ndim == 2:
        mixed = mixed.T

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if log:
        log.progress(f"保存混音文件: {output_path}")

    sf.write(output_path, mixed, sr)

    output_size = Path(output_path).stat().st_size
    duration = target_len / sr

    if log:
        log.success("混音完成")
        log.audio(f"输出时长: {duration:.2f}秒")
        log.audio(f"输出大小: {output_size / 1024 / 1024:.2f} MB")

    return output_path


def check_pedalboard_available() -> bool:
    """检查 pedalboard 是否可用。"""
    return PEDALBOARD_AVAILABLE
