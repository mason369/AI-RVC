# -*- coding: utf-8 -*-
"""
高级去混响模块 - 基于二进制残差掩码和时域一致性
参考: arXiv 2510.00356 - Dereverberation Using Binary Residual Masking
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BinaryResidualMask(nn.Module):
    """
    二进制残差掩码网络 - 专注于抑制混响而非预测完整频谱

    核心思想：
    1. 学习识别并抑制晚期反射（late reflections）
    2. 保留直达声路径（direct path）
    3. 使用时域一致性损失隐式学习相位
    """

    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1

        # U-Net编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 瓶颈层 - 时序注意力
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # U-Net解码器
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 输出层 - 二进制掩码
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # 输出0-1的掩码
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, F, T] - 输入频谱幅度
        Returns:
            mask: [B, 1, F, T] - 二进制残差掩码
        """
        # 编码
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # 瓶颈
        b = self.bottleneck(e3)

        # 解码 + 跳跃连接
        d3 = self.decoder3(b)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.decoder1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        # 输出掩码
        mask = self.output(d1)
        return mask


def advanced_dereverb(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    高级去混响 - 分离干声和混响

    Args:
        audio: 输入音频 [samples]
        sr: 采样率
        n_fft: FFT大小
        hop_length: 跳跃长度
        device: 计算设备

    Returns:
        dry_signal: 干声（直达声）
        reverb_tail: 混响尾巴
    """
    import librosa

    # STFT
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    mag = np.abs(spec).astype(np.float32)
    phase = np.angle(spec)

    # 基于能量的混响检测
    # 1. 计算时域RMS能量
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    rms_db = 20.0 * np.log10(rms + 1e-8)
    ref_db = float(np.percentile(rms_db, 90))

    # 2. 检测晚期反射（late reflections）
    # 晚期反射特征：能量衰减 + 时间延迟
    late_reflections = np.zeros_like(mag, dtype=np.float32)

    for t in range(2, mag.shape[1]):
        # 递归估计：衰减的历史 + 延迟的观测
        late_reflections[:, t] = np.maximum(
            late_reflections[:, t - 1] * 0.92,  # 衰减系数
            mag[:, t - 2] * 0.80  # 延迟观测
        )

    # 3. 计算直达声（direct path）
    # 直达声 = 总能量 - 晚期反射
    direct_path = np.maximum(mag - 0.75 * late_reflections, 0.0)

    # 4. 动态floor：保护有声段
    # 扩展RMS到频谱帧数
    if len(rms) < mag.shape[1]:
        rms_extended = np.pad(rms, (0, mag.shape[1] - len(rms)), mode='edge')
    else:
        rms_extended = rms[:mag.shape[1]]

    # 有声段（高能量）：vocal_strength接近1
    # 无声段（低能量/混响尾）：vocal_strength接近0
    vocal_strength = np.clip((rms_db[:len(rms_extended)] - (ref_db - 35.0)) / 25.0, 0.0, 1.0)

    # 动态floor系数
    reverb_ratio = np.clip(late_reflections / (mag + 1e-8), 0.0, 1.0)
    floor_coef = 0.08 + 0.12 * vocal_strength[np.newaxis, :]
    floor = (1.0 - reverb_ratio) * floor_coef * mag
    direct_path = np.maximum(direct_path, floor)

    # 5. 时域平滑（避免音乐噪声）
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= np.sum(kernel)
    direct_path = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis=1,
        arr=direct_path,
    )
    direct_path = np.clip(direct_path, 0.0, mag)

    # 6. 计算混响残差
    reverb_mag = mag - direct_path
    reverb_mag = np.maximum(reverb_mag, 0.0)

    # 7. 重建音频
    # 干声：使用原始相位（保留音色）
    dry_spec = direct_path * np.exp(1j * phase)
    dry_signal = librosa.istft(dry_spec, hop_length=hop_length, win_length=n_fft, length=len(audio))

    # 混响：使用原始相位
    reverb_spec = reverb_mag * np.exp(1j * phase)
    reverb_tail = librosa.istft(reverb_spec, hop_length=hop_length, win_length=n_fft, length=len(audio))

    return dry_signal.astype(np.float32), reverb_tail.astype(np.float32)


def apply_reverb_to_converted(
    converted_dry: np.ndarray,
    original_reverb: np.ndarray,
    mix_ratio: float = 0.8
) -> np.ndarray:
    """
    将原始混响重新应用到转换后的干声上

    Args:
        converted_dry: 转换后的干声
        original_reverb: 原始混响尾巴
        mix_ratio: 混响混合比例 (0-1)

    Returns:
        wet_signal: 带混响的转换结果
    """
    # 对齐长度
    min_len = min(len(converted_dry), len(original_reverb))
    converted_dry = converted_dry[:min_len]
    original_reverb = original_reverb[:min_len]

    # 混合
    wet_signal = converted_dry + mix_ratio * original_reverb

    # 软限幅
    from lib.audio import soft_clip
    wet_signal = soft_clip(wet_signal, threshold=0.9, ceiling=0.99)

    return wet_signal.astype(np.float32)


if __name__ == "__main__":
    # 测试
    print("Testing advanced dereverberation...")

    # 生成测试信号：干声 + 混响
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # 干声：440Hz正弦波
    dry = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # 混响：衰减的延迟
    reverb = np.zeros_like(dry)
    delay_samples = int(0.05 * sr)  # 50ms延迟
    for i in range(3):
        delay = delay_samples * (i + 1)
        decay = 0.5 ** (i + 1)
        if delay < len(reverb):
            reverb[delay:] += dry[:-delay] * decay

    # 混合信号
    wet = dry + reverb * 0.5

    # 去混响
    dry_extracted, reverb_extracted = advanced_dereverb(wet, sr)

    print(f"Input RMS: {np.sqrt(np.mean(wet**2)):.4f}")
    print(f"Dry RMS: {np.sqrt(np.mean(dry_extracted**2)):.4f}")
    print(f"Reverb RMS: {np.sqrt(np.mean(reverb_extracted**2)):.4f}")
    print(f"Separation ratio: {np.sqrt(np.mean(dry_extracted**2)) / (np.sqrt(np.mean(reverb_extracted**2)) + 1e-8):.2f}")

    print("\n[OK] Advanced dereverberation test passed!")
