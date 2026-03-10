# -*- coding: utf-8 -*-
"""
音频后处理模块 - 齿音和呼吸音处理
基于研究文献的最佳实践
"""
import numpy as np
from scipy import signal
from typing import Optional


def detect_sibilance_frames(audio: np.ndarray, sr: int, threshold_db: float = -20.0) -> np.ndarray:
    """
    检测齿音帧 (s, sh, ch, z 等高频辅音)

    参考: "Managing Sibilance" - Sound on Sound
    齿音主要集中在 4-10kHz 频段

    Args:
        audio: 音频数据
        sr: 采样率
        threshold_db: 高频能量阈值 (dB)

    Returns:
        布尔数组，True 表示齿音帧
    """
    # 设计高通滤波器提取高频成分 (4-10kHz)
    nyquist = sr / 2
    low_freq = 4000 / nyquist
    high_freq = min(10000 / nyquist, 0.99)

    # 带通滤波器
    sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos')
    high_freq_audio = signal.sosfilt(sos, audio)

    # 计算帧能量
    frame_length = int(0.02 * sr)  # 20ms 帧
    hop_length = int(0.01 * sr)    # 10ms 跳跃

    n_frames = 1 + (len(audio) - frame_length) // hop_length
    high_energy = np.zeros(n_frames)
    total_energy = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio):
            break

        # 高频能量
        high_energy[i] = np.sum(high_freq_audio[start:end] ** 2)
        # 总能量
        total_energy[i] = np.sum(audio[start:end] ** 2)

    # 计算高频能量比例
    high_ratio = np.zeros_like(high_energy)
    mask = total_energy > 1e-10
    high_ratio[mask] = high_energy[mask] / total_energy[mask]

    # 转换为 dB
    high_energy_db = 10 * np.log10(high_energy + 1e-10)

    # 齿音检测：高频能量高且高频比例大
    is_sibilance = (high_energy_db > threshold_db) & (high_ratio > 0.3)

    return is_sibilance


def reduce_sibilance(audio: np.ndarray, sr: int, reduction_db: float = 6.0) -> np.ndarray:
    """
    减少齿音 (De-essing)

    参考: "Advanced Sibilance Control" - Mike's Mix Master
    使用多频段动态压缩技术

    Args:
        audio: 音频数据
        sr: 采样率
        reduction_db: 齿音衰减量 (dB)

    Returns:
        处理后的音频
    """
    # 检测齿音帧
    sibilance_frames = detect_sibilance_frames(audio, sr)

    if not np.any(sibilance_frames):
        return audio

    # 计算衰减增益曲线（在时域应用，避免频段分离的相位问题）
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)

    gain_curve = np.ones(len(audio))
    reduction_factor = 10 ** (-reduction_db / 20)

    for i, is_sib in enumerate(sibilance_frames):
        if is_sib:
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break

            # 平滑过渡
            fade_in = np.linspace(1.0, reduction_factor, frame_length // 4)
            sustain = np.full(frame_length // 2, reduction_factor)
            fade_out = np.linspace(reduction_factor, 1.0, frame_length // 4)
            envelope = np.concatenate([fade_in, sustain, fade_out])

            # 应用增益
            gain_curve[start:start+len(envelope)] = np.minimum(
                gain_curve[start:start+len(envelope)],
                envelope
            )

    # 直接在时域应用增益（避免频段分离）
    result = audio * gain_curve

    return result


def detect_breath_frames(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> np.ndarray:
    """
    检测呼吸音帧

    呼吸音特征：
    - 低能量
    - 宽频噪声
    - 通常在乐句之间

    Args:
        audio: 音频数据
        sr: 采样率
        threshold_db: 能量阈值 (dB)

    Returns:
        布尔数组，True 表示呼吸音帧
    """
    frame_length = int(0.02 * sr)  # 20ms
    hop_length = int(0.01 * sr)    # 10ms

    n_frames = 1 + (len(audio) - frame_length) // hop_length
    is_breath = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio):
            break

        frame = audio[start:end]

        # 计算能量
        energy = np.sum(frame ** 2)
        energy_db = 10 * np.log10(energy + 1e-10)

        # 计算频谱平坦度 (噪声特征)
        fft = np.abs(np.fft.rfft(frame))
        geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
        arithmetic_mean = np.mean(fft)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

        # 呼吸音：低能量 + 高频谱平坦度
        is_breath[i] = (energy_db < threshold_db) and (spectral_flatness > 0.5)

    return is_breath


def reduce_breath_noise(audio: np.ndarray, sr: int, reduction_db: float = 12.0) -> np.ndarray:
    """
    减少呼吸音噪声

    参考: "How to REALLY Clean Vocals" - Waves

    Args:
        audio: 音频数据
        sr: 采样率
        reduction_db: 呼吸音衰减量 (dB)

    Returns:
        处理后的音频
    """
    # 检测呼吸音帧
    breath_frames = detect_breath_frames(audio, sr)

    if not np.any(breath_frames):
        return audio

    # 计算衰减增益曲线
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)

    gain_curve = np.ones(len(audio))
    reduction_factor = 10 ** (-reduction_db / 20)

    for i, is_breath in enumerate(breath_frames):
        if is_breath:
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break

            # 平滑过渡，避免咔嗒声
            fade_length = frame_length // 4
            fade_in = np.linspace(1.0, reduction_factor, fade_length)
            sustain = np.full(frame_length - 2 * fade_length, reduction_factor)
            fade_out = np.linspace(reduction_factor, 1.0, fade_length)
            envelope = np.concatenate([fade_in, sustain, fade_out])

            # 应用增益
            gain_curve[start:start+len(envelope)] = np.minimum(
                gain_curve[start:start+len(envelope)],
                envelope
            )

    # 应用增益曲线
    result = audio * gain_curve

    return result


def apply_vocal_cleanup(
    audio: np.ndarray,
    sr: int,
    reduce_sibilance_enabled: bool = True,
    reduce_breath_enabled: bool = True,
    sibilance_reduction_db: float = 4.0,
    breath_reduction_db: float = 8.0
) -> np.ndarray:
    """
    应用完整的人声清理处理

    Args:
        audio: 音频数据
        sr: 采样率
        reduce_sibilance_enabled: 是否减少齿音
        reduce_breath_enabled: 是否减少呼吸音
        sibilance_reduction_db: 齿音衰减量 (dB)
        breath_reduction_db: 呼吸音衰减量 (dB)

    Returns:
        处理后的音频
    """
    result = audio.copy()

    # 减少呼吸音（先处理，因为能量更低）
    if reduce_breath_enabled:
        result = reduce_breath_noise(result, sr, breath_reduction_db)

    # 减少齿音
    if reduce_sibilance_enabled:
        result = reduce_sibilance(result, sr, sibilance_reduction_db)

    return result
