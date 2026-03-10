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
    amplitude = np.abs(analytic_signal)

    # 检测相位跳变
    phase_diff = np.diff(instantaneous_phase)
    phase_diff_threshold = np.percentile(np.abs(phase_diff), 99) * 2.5

    # 找到相位跳变点
    discontinuities = np.where(np.abs(phase_diff) > phase_diff_threshold)[0]

    if len(discontinuities) == 0:
        return audio

    # 修复每个不连续点
    result = audio.copy()
    phase_corrected = instantaneous_phase.copy()

    for disc_idx in discontinuities:
        # 计算相位跳变量
        phase_jump = phase_diff[disc_idx]

        # 在不连续点之后应用相位校正（累积补偿）
        correction_length = min(int(0.02 * sr), len(phase_corrected) - disc_idx - 1)  # 20ms
        if correction_length > 0:
            # 线性过渡相位校正
            correction_curve = np.linspace(phase_jump, 0, correction_length)
            phase_corrected[disc_idx + 1:disc_idx + 1 + correction_length] -= correction_curve

    # 用校正后的相位重建信号
    corrected_signal = amplitude * np.exp(1j * phase_corrected)
    result = np.real(corrected_signal).astype(np.float32)

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
    # 第一步：去除DC偏移和极低频噪声（0-80Hz）
    # 这是vocoder常见的低频泄漏问题
    from scipy import signal as scipy_signal

    # 设计高通滤波器：80Hz截止
    nyquist = sr / 2
    cutoff = 80 / nyquist

    # 使用4阶Butterworth高通滤波器
    sos = scipy_signal.butter(4, cutoff, btype='highpass', output='sos')
    audio = scipy_signal.sosfilt(sos, audio)

    # 第二步：检测和清理宽频噪声（原有逻辑）
    # 检测低能量区域（可能是呼吸音）
    frame_length = int(0.02 * sr)  # 20ms
    hop_length = int(0.01 * sr)    # 10ms

    n_frames = 1 + (len(audio) - frame_length) // hop_length

    # 计算每帧的能量和频谱平坦度
    energy = np.zeros(n_frames)
    spectral_flatness = np.zeros(n_frames)
    high_freq_ratio = np.zeros(n_frames)  # 新增：高频能量占比

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

            # 计算高频能量占比（4kHz以上）
            freqs = np.fft.rfftfreq(len(frame), 1/sr)
            high_freq_mask = freqs >= 4000
            high_freq_energy = np.sum(fft[high_freq_mask] ** 2)
            total_freq_energy = np.sum(fft ** 2)
            high_freq_ratio[i] = high_freq_energy / (total_freq_energy + 1e-10)

    # 归一化能量
    energy_db = 10 * np.log10(energy + 1e-10)

    # 自适应底噪检测：
    # 1. 计算能量分布的统计特征
    # 2. 使用最低5%作为候选底噪区域
    # 3. 在候选区域中,根据频谱特征进一步筛选

    # 候选底噪区域：最低5%能量
    candidate_threshold = np.percentile(energy_db, 5)

    # 在候选区域中,检测真正的底噪
    # 底噪类型1：宽频噪声（频谱平坦度 > 0.35）
    # 底噪类型2：高频电流声（高频占比 > 0.15）
    is_candidate = energy_db < candidate_threshold
    is_wideband_noise = is_candidate & (spectral_flatness > 0.35)
    is_highfreq_noise = is_candidate & (high_freq_ratio > 0.15)

    # 合并两种类型的底噪
    is_noise = is_wideband_noise | is_highfreq_noise

    # 如果检测到的底噪帧数太少(<1%),说明音频本身很纯净,不需要处理
    noise_ratio = is_noise.sum() / len(is_noise)
    if noise_ratio < 0.01:
        return audio

    # 如果提供了F0，使用F0=0来辅助判断
    if f0 is not None and len(f0) > 0:
        # F0对齐到音频帧
        f0_per_audio_frame = len(f0) / n_frames
        for i in range(n_frames):
            if not is_noise[i]:
                continue

            f0_idx = int(i * f0_per_audio_frame)
            if f0_idx < len(f0):
                # 如果F0>0，说明有音高，不是底噪
                if f0[f0_idx] > 0:
                    is_noise[i] = False

    # 使用is_noise替代is_breath，更准确地描述我们要处理的内容
    is_breath = is_noise

    # 根据底噪比例动态调整清理强度
    # 底噪越多，说明vocoder质量越差，需要更激进的清理
    if noise_ratio < 0.05:
        # 底噪很少(1-5%)，温和清理
        spectral_threshold_percentile = 85  # 保留15%
        magnitude_attenuation = 0.2  # 衰减到20%
        mix_ratio = 0.5  # 50%清理
    elif noise_ratio < 0.15:
        # 底噪中等(5-15%)，中等清理
        spectral_threshold_percentile = 90  # 保留10%
        magnitude_attenuation = 0.1  # 衰减到10%
        mix_ratio = 0.7  # 70%清理
    else:
        # 底噪很多(>15%)，激进清理
        spectral_threshold_percentile = 95  # 保留5%
        magnitude_attenuation = 0.05  # 衰减到5%
        mix_ratio = 0.85  # 85%清理

    # 对底噪区域应用降噪
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
            freqs = np.fft.rfftfreq(len(frame), 1/sr)

            # 检测这一帧是高频噪声还是宽频噪声
            high_freq_mask = freqs >= 4000
            high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)
            total_freq_energy = np.sum(magnitude ** 2)
            frame_high_ratio = high_freq_energy / (total_freq_energy + 1e-10)

            if frame_high_ratio > 0.15:
                # 高频电流声：专门衰减高频部分
                magnitude[high_freq_mask] *= 0.05  # 高频衰减到5%
                # 中频(1-4kHz)温和衰减
                mid_freq_mask = (freqs >= 1000) & (freqs < 4000)
                magnitude[mid_freq_mask] *= 0.3
            else:
                # 宽频噪声：使用原有的频谱门限
                threshold = np.percentile(magnitude, spectral_threshold_percentile)
                magnitude = np.where(magnitude > threshold, magnitude, magnitude * magnitude_attenuation)

            # 重建
            fft_cleaned = magnitude * np.exp(1j * phase)
            frame_cleaned = np.fft.irfft(fft_cleaned, n=len(frame))

            # 平滑过渡
            fade_length = min(hop_length // 2, len(frame) // 4)
            if fade_length > 0:
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)

                frame_cleaned[:fade_length] *= fade_in
                frame_cleaned[-fade_length:] *= fade_out

            # 动态混合比例
            result[start:end] = frame * (1 - mix_ratio) + frame_cleaned * mix_ratio

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

            # 使用低通滤波平滑幅度包络（而非除法）
            envelope = np.abs(signal.hilbert(sustained_segment))

            # 平滑包络
            b, a = signal.butter(2, 50 / (sr / 2), btype='low')
            smoothed_envelope = signal.filtfilt(b, a, envelope)

            # 计算增益调整（避免除法放大噪声）
            # 只在包络变化剧烈的地方应用平滑
            envelope_variation = np.abs(envelope - smoothed_envelope)
            variation_threshold = np.percentile(envelope_variation, 75)

            # 创建混合掩码：变化大的地方用平滑包络，变化小的地方保持原样
            blend_mask = np.clip(envelope_variation / (variation_threshold + 1e-6), 0, 1)

            # 计算目标包络
            target_envelope = smoothed_envelope * blend_mask + envelope * (1 - blend_mask)

            # 应用包络调整（使用乘法而非除法）
            if np.max(envelope) > 1e-6:
                gain = target_envelope / (envelope + 1e-6)
                # 限制增益范围，避免放大噪声
                gain = np.clip(gain, 0.5, 2.0)
                result[start_sample:end_sample] = sustained_segment * gain

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
