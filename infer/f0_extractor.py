# -*- coding: utf-8 -*-
"""
F0 (基频) 提取模块 - 支持多种提取方法
"""
import numpy as np
import torch
from typing import Optional, Literal

# F0 提取方法类型
F0Method = Literal["rmvpe", "pm", "harvest", "crepe"]


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
        f0 = self.model.infer_from_audio(audio, thred=0.03)

        return f0


def get_f0_extractor(method: F0Method, device: str = "cuda",
                     rmvpe_path: str = None) -> F0Extractor:
    """
    获取 F0 提取器实例

    Args:
        method: 提取方法 ("rmvpe", "pm", "harvest", "crepe")
        device: 计算设备
        rmvpe_path: RMVPE 模型路径 (仅 rmvpe 方法需要)

    Returns:
        F0Extractor: 提取器实例
    """
    if method == "rmvpe":
        if rmvpe_path is None:
            raise ValueError("RMVPE 方法需要指定模型路径")
        return RMVPEExtractor(rmvpe_path, device=device)
    elif method == "pm":
        return PMExtractor()
    elif method == "harvest":
        return HarvestExtractor()
    elif method == "crepe":
        return CrepeExtractor(device=device)
    else:
        raise ValueError(f"未知的 F0 提取方法: {method}")


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
