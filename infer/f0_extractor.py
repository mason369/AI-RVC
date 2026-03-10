# -*- coding: utf-8 -*-
"""
F0 (基频) 提取模块 - 支持多种提取方法
"""
import numpy as np
import torch
from typing import Optional, Literal

# F0 提取方法类型
F0Method = Literal["rmvpe", "pm", "harvest", "crepe", "hybrid"]


class F0Extractor:
    """F0 提取器基类"""

    def __init__(self, sample_rate: int = 16000, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = 50
        self.f0_max = 1100

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """提取 F0，子类需实现此方法"""
        raise NotImplementedError


class PMExtractor(F0Extractor):
    """Parselmouth (Praat) F0 提取器 - 速度快"""

    def extract(self, audio: np.ndarray) -> np.ndarray:
        import parselmouth

        time_step = self.hop_length / self.sample_rate
        sound = parselmouth.Sound(audio, self.sample_rate)

        pitch = sound.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.6,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max
        )

        f0 = pitch.selected_array["frequency"]
        f0[f0 == 0] = np.nan

        return f0


class HarvestExtractor(F0Extractor):
    """PyWorld Harvest F0 提取器 - 质量较好"""

    def extract(self, audio: np.ndarray) -> np.ndarray:
        import pyworld

        audio = audio.astype(np.float64)
        f0, _ = pyworld.harvest(
            audio,
            self.sample_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.hop_length / self.sample_rate * 1000
        )

        return f0


class CrepeExtractor(F0Extractor):
    """TorchCrepe F0 提取器 - 深度学习方法"""

    def __init__(self, sample_rate: int = 16000, hop_length: int = 160,
                 device: str = "cuda"):
        super().__init__(sample_rate, hop_length)
        self.device = device

    def extract(self, audio: np.ndarray) -> np.ndarray:
        import torchcrepe

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)

        f0, _ = torchcrepe.predict(
            audio_tensor,
            self.sample_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            model="full",
            batch_size=512,
            device=self.device,
            return_periodicity=True
        )

        f0 = f0.squeeze(0).cpu().numpy()
        return f0


class RMVPEExtractor(F0Extractor):
    """RMVPE F0 提取器 - 质量最高 (推荐)"""

    def __init__(self, model_path: str, sample_rate: int = 16000,
                 hop_length: int = 160, device: str = "cuda"):
        super().__init__(sample_rate, hop_length)
        self.device = device
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """加载 RMVPE 模型"""
        if self.model is not None:
            return

        from models.rmvpe import RMVPE

        self.model = RMVPE(self.model_path, device=self.device)
        print(f"RMVPE 模型已加载: {self.device}")

    def extract(self, audio: np.ndarray) -> np.ndarray:
        self.load_model()

        # RMVPE 需要 16kHz 输入
        f0 = self.model.infer_from_audio(audio, thred=0.01)

        return f0


def get_f0_extractor(method: F0Method, device: str = "cuda",
                     rmvpe_path: str = None, crepe_threshold: float = 0.05) -> F0Extractor:
    """
    获取 F0 提取器实例

    Args:
        method: 提取方法 ("rmvpe", "pm", "harvest", "crepe", "hybrid")
        device: 计算设备
        rmvpe_path: RMVPE 模型路径 (rmvpe/hybrid 方法需要)
        crepe_threshold: CREPE置信度阈值 (仅hybrid方法使用)

    Returns:
        F0Extractor: 提取器实例
    """
    if method == "rmvpe":
        if rmvpe_path is None:
            raise ValueError("RMVPE 方法需要指定模型路径")
        return RMVPEExtractor(rmvpe_path, device=device)
    elif method == "hybrid":
        if rmvpe_path is None:
            raise ValueError("Hybrid 方法需要指定RMVPE模型路径")
        return HybridF0Extractor(rmvpe_path, device=device, crepe_threshold=crepe_threshold)
    elif method == "pm":
        return PMExtractor()
    elif method == "harvest":
        return HarvestExtractor()
    elif method == "crepe":
        return CrepeExtractor(device=device)
    else:
        raise ValueError(f"未知的 F0 提取方法: {method}")


class HybridF0Extractor(F0Extractor):
    """混合F0提取器 - RMVPE主导 + CREPE高精度补充"""

    def __init__(self, rmvpe_path: str, sample_rate: int = 16000,
                 hop_length: int = 160, device: str = "cuda",
                 crepe_threshold: float = 0.05):
        super().__init__(sample_rate, hop_length)
        self.device = device
        self.rmvpe = RMVPEExtractor(rmvpe_path, sample_rate, hop_length, device)
        self.crepe = None  # 延迟加载
        self.crepe_threshold = crepe_threshold

    def _load_crepe(self):
        """延迟加载CREPE模型"""
        if self.crepe is None:
            try:
                self.crepe = CrepeExtractor(self.sample_rate, self.hop_length, self.device)
            except ImportError:
                print("警告: torchcrepe未安装，混合F0将仅使用RMVPE")
                self.crepe = False

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        混合提取F0：
        1. 使用RMVPE作为主要方法（快速、稳定）
        2. 在RMVPE不稳定的区域使用CREPE补充（高精度）
        """
        # 提取RMVPE F0
        f0_rmvpe = self.rmvpe.extract(audio)

        # 如果CREPE不可用，直接返回RMVPE结果
        self._load_crepe()
        if self.crepe is False:
            return f0_rmvpe

        # 提取CREPE F0和置信度
        import torchcrepe
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        f0_crepe, confidence = torchcrepe.predict(
            audio_tensor,
            self.sample_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            model="full",
            batch_size=512,
            device=self.device,
            return_periodicity=True
        )
        f0_crepe = f0_crepe.squeeze(0).cpu().numpy()
        confidence = confidence.squeeze(0).cpu().numpy()

        # 对齐长度
        min_len = min(len(f0_rmvpe), len(f0_crepe), len(confidence))
        f0_rmvpe = f0_rmvpe[:min_len]
        f0_crepe = f0_crepe[:min_len]
        confidence = confidence[:min_len]

        # 检测RMVPE不稳定区域
        # 1. F0跳变过大（超过3个半音）
        f0_diff = np.abs(np.diff(f0_rmvpe, prepend=f0_rmvpe[0]))
        semitone_diff = np.abs(12 * np.log2((f0_rmvpe + 1e-6) / (np.roll(f0_rmvpe, 1) + 1e-6)))
        semitone_diff[0] = 0
        unstable_jump = semitone_diff > 3.0

        # 2. CREPE置信度高但RMVPE给出F0=0
        unstable_unvoiced = (f0_rmvpe < 1e-3) & (confidence > self.crepe_threshold)

        # 3. RMVPE和CREPE差异过大（超过2个半音）且CREPE置信度高
        f0_ratio = (f0_crepe + 1e-6) / (f0_rmvpe + 1e-6)
        semitone_gap = np.abs(12 * np.log2(f0_ratio))
        unstable_diverge = (semitone_gap > 2.0) & (confidence > self.crepe_threshold * 1.5)

        # 合并不稳定区域
        unstable_mask = unstable_jump | unstable_unvoiced | unstable_diverge

        # 扩展不稳定区域（前后各2帧）以平滑过渡
        kernel = np.ones(5, dtype=bool)
        unstable_mask = np.convolve(unstable_mask, kernel, mode='same')

        # 混合F0：不稳定区域使用CREPE，其他区域使用RMVPE
        f0_hybrid = f0_rmvpe.copy()
        f0_hybrid[unstable_mask] = f0_crepe[unstable_mask]

        # 平滑过渡边界
        for i in range(1, len(f0_hybrid) - 1):
            if unstable_mask[i] != unstable_mask[i-1]:
                # 边界处使用加权平均
                w = 0.5
                f0_hybrid[i] = w * f0_rmvpe[i] + (1-w) * f0_crepe[i]

        return f0_hybrid


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    """
    音调偏移

    Args:
        f0: 原始 F0
        semitones: 偏移半音数 (正数升调，负数降调)

    Returns:
        np.ndarray: 偏移后的 F0
    """
    factor = 2 ** (semitones / 12)
    f0_shifted = f0 * factor
    return f0_shifted
