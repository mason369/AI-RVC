# -*- coding: utf-8 -*-
"""
Vocoder伪影修复 - 针对呼吸音电音和长音撕裂
基于RVC社区反馈和研究文献
"""
import numpy as np
from scipy import signal
from typing import Optional


def fix_phase_discontinuity(audio: np.ndarray, sr: int, chunk_boundaries: Optional[list] = None) -> np.ndarray:
    """
    修复相位不连续导致的撕裂

    参考: "Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding" (arXiv:2601.14472)
    Vocoder在长音时会产生相位不连续，导致撕裂

    Args:
        audio: 音频数据
        sr: 采样率
        chunk_boundaries: 分块边界位置（样本索引）

    Returns:
        修复后的音频
    """
    # 使用希尔伯特变换提取瞬时相位
    analytic_signal = signal.hilbert(audio)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # 检测相位跳变
    phase_diff = np.diff(instantaneous_phase)
    phase_diff_threshold = np.percentile(np.abs(phase_diff), 99) * 2

    # 找到相位跳变点
    discontinuities = np.where(np.abs(phase_diff) > phase_diff_threshold)[0]

    if len(discontinuities) == 0:
        return audio

    # 修复每个不连续点
    result = audio.copy()
    for disc_idx in discontinuities:
        # 在不连续点周围应用平滑
        window_size = int(0.01 * sr)  # 10ms窗口
        start = max(0, disc_idx - window_size // 2)
        end = min(len(audio), disc_idx + window_size // 2)

        if end - start < 10:
            continue

        # 使用汉宁窗平滑
        window = signal.windows.hann(end - start)
        # 只平滑幅度，保留相位
        result[start:end] = result[start:end] * window

    return result


def reduce_breath_electric_noise(audio: np.ndarray, sr: int, f0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    减少呼吸音中的电音

    参考: GitHub Issue #65 "Artefacting when speech has breath"
    问题: Vocoder在F0=0的区域会产生电子噪声

    Args:
        audio: 音频数据
        sr: 采样率
        f0: F0序列（可选，用于定位呼吸音）

    Returns:
        处理后的音频
    """
    # 检测低能量区域（可能是呼吸音）
    frame_length = int(0.02 * sr)  # 20ms
    hop_length = int(0.01 * sr)    # 10ms

    n_frames = 1 + (len(audio) - frame_length) // hop_length

    # 计算每帧的能量和频谱平坦度
    energy = np.zeros(n_frames)
    spectral_flatness = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio):
            break

        frame = audio[start:end]

        # 能量
        energy[i] = np.sum(frame ** 2)

        # 频谱平坦度（噪声特征）
        fft = np.abs(np.fft.rfft(frame))
        if np.sum(fft) > 1e-10:
            geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
            arithmetic_mean = np.mean(fft)
            spectral_flatness[i] = geometric_mean / (arithmetic_mean + 1e-10)

    # 归一化能量
    energy_db = 10 * np.log10(energy + 1e-10)
    energy_threshold = np.percentile(energy_db, 20)  # 低能量阈值

    # 检测呼吸音：低能量 + 高频谱平坦度
    is_breath = (energy_db < energy_threshold) & (spectral_flatness > 0.6)

    # 如果提供了F0，使用F0=0来辅助判断
    if f0 is not None and len(f0) > 0:
        # F0对齐到音频帧
        f0_per_audio_frame = len(f0) / n_frames
        for i in range(n_frames):
            f0_idx = int(i * f0_per_audio_frame)
            if f0_idx < len(f0) and f0[f0_idx] == 0:
                is_breath[i] = True

    # 对呼吸音区域应用降噪
    result = audio.copy()

    for i in range(n_frames):
        if is_breath[i]:
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break

            # 使用频谱门限降噪
            frame = audio[start:end]

            # FFT
            fft = np.fft.rfft(frame)
            magnitude = np.abs(fft)
            phase = np.angle(fft)

            # 频谱门限：去除低于阈值的频率成分
            threshold = np.percentile(magnitude, 70)  # 保留30%的能量
            magnitude = np.where(magnitude > threshold, magnitude, magnitude * 0.1)

            # 重建
            fft_cleaned = magnitude * np.exp(1j * phase)
            frame_cleaned = np.fft.irfft(fft_cleaned, n=len(frame))

            # 平滑过渡
            fade_length = min(hop_length // 2, len(frame) // 4)
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)

            frame_cleaned[:fade_length] *= fade_in
            frame_cleaned[-fade_length:] *= fade_out

            # 混合
            result[start:end] = frame * 0.3 + frame_cleaned * 0.7

    return result


def stabilize_sustained_notes(audio: np.ndarray, sr: int, f0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    稳定长音，防止撕裂

    参考: "Mel Spectrogram Inversion with Stable Pitch" - Apple Research
    长音时vocoder容易产生相位漂移

    Args:
        audio: 音频数据
        sr: 采样率
        f0: F0序列（用于检测长音）

    Returns:
        稳定后的音频
    """
    if f0 is None or len(f0) == 0:
        return audio

    # 检测长音区域（F0稳定且持续时间长）
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)

    # F0对齐到音频帧
    n_audio_frames = 1 + (len(audio) - frame_length) // hop_length
    f0_per_audio_frame = len(f0) / n_audio_frames

    is_sustained = np.zeros(n_audio_frames, dtype=bool)

    # 检测F0稳定的区域
    window_size = 20  # 200ms窗口
    for i in range(window_size, n_audio_frames - window_size):
        f0_idx = int(i * f0_per_audio_frame)
        if f0_idx >= len(f0):
            break

        # 获取窗口内的F0
        f0_window_start = max(0, f0_idx - window_size)
        f0_window_end = min(len(f0), f0_idx + window_size)
        f0_window = f0[f0_window_start:f0_window_end]

        # 过滤F0=0
        f0_voiced = f0_window[f0_window > 0]

        if len(f0_voiced) > window_size * 0.8:  # 80%有声
            # 计算F0稳定性
            f0_std = np.std(f0_voiced)
            f0_mean = np.mean(f0_voiced)

            # F0变化小于5%认为是长音
            if f0_std / (f0_mean + 1e-6) < 0.05:
                is_sustained[i] = True

    # 对长音区域应用相位稳定
    result = audio.copy()

    i = 0
    while i < n_audio_frames:
        if is_sustained[i]:
            # 找到长音区域的起止
            start_frame = i
            while i < n_audio_frames and is_sustained[i]:
                i += 1
            end_frame = i

            # 转换为样本索引
            start_sample = start_frame * hop_length
            end_sample = min(end_frame * hop_length + frame_length, len(audio))

            if end_sample - start_sample < frame_length:
                continue

            # 提取长音段
            sustained_segment = audio[start_sample:end_sample]

            # 使用LPC分析提取稳定的谐波结构
            # 简化版本：使用低通滤波平滑幅度包络
            envelope = np.abs(signal.hilbert(sustained_segment))

            # 平滑包络
            b, a = signal.butter(2, 50 / (sr / 2), btype='low')
            smoothed_envelope = signal.filtfilt(b, a, envelope)

            # 应用平滑包络
            if np.max(envelope) > 1e-6:
                result[start_sample:end_sample] = sustained_segment * (smoothed_envelope / (envelope + 1e-6))

        i += 1

    return result


def apply_vocoder_artifact_fix(
    audio: np.ndarray,
    sr: int,
    f0: Optional[np.ndarray] = None,
    chunk_boundaries: Optional[list] = None,
    fix_phase: bool = True,
    fix_breath: bool = True,
    fix_sustained: bool = True
) -> np.ndarray:
    """
    应用完整的vocoder伪影修复

    Args:
        audio: 音频数据
        sr: 采样率
        f0: F0序列
        chunk_boundaries: 分块边界
        fix_phase: 是否修复相位不连续
        fix_breath: 是否修复呼吸音电音
        fix_sustained: 是否稳定长音

    Returns:
        修复后的音频
    """
    result = audio.copy()

    # 1. 修复相位不连续（长音撕裂）
    if fix_phase:
        result = fix_phase_discontinuity(result, sr, chunk_boundaries)

    # 2. 减少呼吸音电音
    if fix_breath:
        result = reduce_breath_electric_noise(result, sr, f0)

    # 3. 稳定长音
    if fix_sustained:
        result = stabilize_sustained_notes(result, sr, f0)

    return result
