# -*- coding: utf-8 -*-
"""
频谱后处理模块 - 修复vocoder伪影和破音
基于最新研究文献
"""
import numpy as np
from scipy import signal
from typing import Optional


def detect_artifacts(audio: np.ndarray, sr: int, frame_length: int = 2048) -> np.ndarray:
    """
    检测音频中的伪影帧

    参考: "Understanding AI Voice Artifacts" - Sonarworks
    伪影通常表现为频谱异常和突变

    Args:
        audio: 音频数据
        sr: 采样率
        frame_length: STFT帧长

    Returns:
        布尔数组，True表示伪影帧
    """
    # STFT分析
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=frame_length, noverlap=frame_length//2)

    # 计算频谱特征
    magnitude = np.abs(Zxx)

    # 1. 检测频谱突变
    spec_diff = np.diff(magnitude, axis=1)
    spec_diff_norm = np.linalg.norm(spec_diff, axis=0)
    threshold = np.percentile(spec_diff_norm, 95)
    sudden_changes = spec_diff_norm > threshold * 2

    # 2. 检测频谱异常（过高的高频能量）
    high_freq_idx = len(f) // 2  # 高频部分
    high_freq_energy = np.sum(magnitude[high_freq_idx:, :], axis=0)
    total_energy = np.sum(magnitude, axis=0) + 1e-10
    high_freq_ratio = high_freq_energy / total_energy
    abnormal_spectrum = high_freq_ratio > 0.3

    # 合并检测结果
    is_artifact = sudden_changes | abnormal_spectrum

    # 扩展到音频样本
    hop_length = frame_length // 2
    artifact_samples = np.zeros(len(audio), dtype=bool)
    for i, is_art in enumerate(is_artifact):
        if is_art:
            start = i * hop_length
            end = start + frame_length
            artifact_samples[start:end] = True

    return artifact_samples


def spectral_smoothing(audio: np.ndarray, sr: int, smoothing_factor: float = 0.3) -> np.ndarray:
    """
    频谱平滑 - 减少vocoder伪影

    参考: "A Conditional Diffusion Model for Singing Voice Synthesis" (arXiv:2506.21478)
    通过频谱域平滑减少失真

    Args:
        audio: 音频数据
        sr: 采样率
        smoothing_factor: 平滑强度 (0-1)

    Returns:
        平滑后的音频
    """
    frame_length = 2048
    hop_length = frame_length // 4

    # STFT
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)

    # 频谱平滑
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # 时间维度平滑（减少突变）
    kernel_time = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel_time /= kernel_time.sum()

    smoothed_mag = np.copy(magnitude)
    for freq_idx in range(magnitude.shape[0]):
        smoothed_mag[freq_idx, :] = np.convolve(
            magnitude[freq_idx, :],
            kernel_time,
            mode='same'
        )

    # 混合原始和平滑频谱
    mixed_mag = magnitude * (1 - smoothing_factor) + smoothed_mag * smoothing_factor

    # 重建
    Zxx_smoothed = mixed_mag * np.exp(1j * phase)
    _, audio_smoothed = signal.istft(Zxx_smoothed, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)

    # 确保长度一致
    if len(audio_smoothed) > len(audio):
        audio_smoothed = audio_smoothed[:len(audio)]
    elif len(audio_smoothed) < len(audio):
        audio_smoothed = np.pad(audio_smoothed, (0, len(audio) - len(audio_smoothed)), mode='edge')

    return audio_smoothed


def harmonic_enhancement(audio: np.ndarray, sr: int, f0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    谐波增强 - 修复高音失真

    参考: "Robust Zero-Shot Singing Voice Conversion with Additive Synthesis" (arXiv:2504.05686)
    使用加性合成增强谐波结构

    Args:
        audio: 音频数据
        sr: 采样率
        f0: F0序列（可选）

    Returns:
        增强后的音频
    """
    frame_length = 2048
    hop_length = frame_length // 4

    # STFT
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # 如果提供了F0，增强谐波
    if f0 is not None and len(f0) > 0:
        # F0对齐到STFT帧
        f0_per_frame = len(f0) / magnitude.shape[1]

        for t_idx in range(magnitude.shape[1]):
            f0_idx = int(t_idx * f0_per_frame)
            if f0_idx < len(f0) and f0[f0_idx] > 0:
                fundamental = f0[f0_idx]

                # 增强前5个谐波
                for harmonic in range(1, 6):
                    harmonic_freq = fundamental * harmonic
                    if harmonic_freq < sr / 2:
                        # 找到最接近的频率bin
                        freq_idx = int(harmonic_freq / (sr / frame_length))
                        if freq_idx < len(f):
                            # 增强该频率附近的能量
                            magnitude[max(0, freq_idx-1):min(len(f), freq_idx+2), t_idx] *= 1.1

    # 重建
    Zxx_enhanced = magnitude * np.exp(1j * phase)
    _, audio_enhanced = signal.istft(Zxx_enhanced, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)

    # 确保长度一致
    if len(audio_enhanced) > len(audio):
        audio_enhanced = audio_enhanced[:len(audio)]
    elif len(audio_enhanced) < len(audio):
        audio_enhanced = np.pad(audio_enhanced, (0, len(audio) - len(audio_enhanced)), mode='edge')

    return audio_enhanced


def transient_preservation(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    瞬态保护 - 保留辅音的清晰度

    参考: "TOWARDS REAL-WORLD ROBUST AND EXPRESSIVE ZERO-SHOT SINGING VOICE CONVERSION" (arXiv:2510.20677)

    Args:
        audio: 音频数据
        sr: 采样率

    Returns:
        处理后的音频
    """
    # 检测瞬态（辅音、打击音）
    # 使用包络检测
    envelope = np.abs(signal.hilbert(audio))

    # 计算包络导数
    envelope_diff = np.diff(envelope, prepend=envelope[0])

    # 瞬态检测：包络快速上升
    threshold = np.percentile(np.abs(envelope_diff), 95)
    is_transient = np.abs(envelope_diff) > threshold

    # 扩展瞬态区域
    kernel = np.ones(int(0.01 * sr))  # 10ms
    is_transient = signal.convolve(is_transient.astype(float), kernel, mode='same') > 0.5

    # 在瞬态区域保留更多原始信号
    # （这里假设audio是处理后的，需要与原始信号混合）
    # 实际使用时需要传入原始信号

    return audio


def apply_spectral_postprocessing(
    audio: np.ndarray,
    sr: int,
    f0: Optional[np.ndarray] = None,
    enable_smoothing: bool = True,
    enable_harmonic_enhancement: bool = True,
    smoothing_factor: float = 0.2
) -> np.ndarray:
    """
    应用完整的频谱后处理

    Args:
        audio: 音频数据
        sr: 采样率
        f0: F0序列（可选）
        enable_smoothing: 是否启用频谱平滑
        enable_harmonic_enhancement: 是否启用谐波增强
        smoothing_factor: 平滑强度

    Returns:
        处理后的音频
    """
    result = audio.copy()

    # 1. 频谱平滑（减少伪影）
    if enable_smoothing:
        result = spectral_smoothing(result, sr, smoothing_factor)

    # 2. 谐波增强（修复高音）
    if enable_harmonic_enhancement and f0 is not None:
        result = harmonic_enhancement(result, sr, f0)

    return result
