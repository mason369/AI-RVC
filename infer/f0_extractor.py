# -*- coding: utf-8 -*-
"""
F0 extraction helpers used by the local VC pipeline.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from infer.quality_policy import build_conservative_crepe_fill_mask


F0Method = Literal["rmvpe", "pm", "harvest", "crepe", "hybrid"]


class F0Extractor:
    """Base F0 extractor."""

    def __init__(self, sample_rate: int = 16000, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = 50
        self.f0_max = 1100

    def extract(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PMExtractor(F0Extractor):
    """Praat/Parselmouth extractor."""

    def extract(self, audio: np.ndarray) -> np.ndarray:
        import parselmouth

        time_step = self.hop_length / self.sample_rate
        sound = parselmouth.Sound(audio, self.sample_rate)
        pitch = sound.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.6,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max,
        )
        f0 = pitch.selected_array["frequency"]
        f0[f0 == 0] = np.nan
        return f0


class HarvestExtractor(F0Extractor):
    """PyWorld harvest extractor."""

    def extract(self, audio: np.ndarray) -> np.ndarray:
        import pyworld

        audio = audio.astype(np.float64)
        f0, _ = pyworld.harvest(
            audio,
            self.sample_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.hop_length / self.sample_rate * 1000,
        )
        return f0


class CrepeExtractor(F0Extractor):
    """TorchCrepe extractor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 160,
        device: str = "cuda",
    ):
        super().__init__(sample_rate, hop_length)
        self.device = device

    def extract(self, audio: np.ndarray) -> np.ndarray:
        import torchcrepe

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        f0, _ = torchcrepe.predict(
            audio_tensor,
            self.sample_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            model="full",
            batch_size=512,
            device=self.device,
            return_periodicity=True,
        )
        return f0.squeeze(0).cpu().numpy()


class RMVPEExtractor(F0Extractor):
    """RMVPE extractor."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        hop_length: int = 160,
        device: str = "cuda",
    ):
        super().__init__(sample_rate, hop_length)
        self.device = device
        self.model = None
        self.model_path = model_path

    def load_model(self) -> None:
        if self.model is not None:
            return

        from models.rmvpe import RMVPE

        self.model = RMVPE(self.model_path, device=self.device)
        print(f"RMVPE model loaded: {self.device}")

    def extract(self, audio: np.ndarray) -> np.ndarray:
        self.load_model()
        return self.model.infer_from_audio(audio, thred=0.01)


def get_f0_extractor(
    method: F0Method,
    device: str = "cuda",
    rmvpe_path: str | None = None,
    crepe_threshold: float = 0.05,
) -> F0Extractor:
    """Create an F0 extractor."""

    if method == "rmvpe":
        if rmvpe_path is None:
            raise ValueError("RMVPE requires a model path")
        return RMVPEExtractor(rmvpe_path, device=device)
    if method == "hybrid":
        if rmvpe_path is None:
            raise ValueError("Hybrid requires an RMVPE model path")
        return HybridF0Extractor(
            rmvpe_path,
            device=device,
            crepe_threshold=crepe_threshold,
        )
    if method == "pm":
        return PMExtractor()
    if method == "harvest":
        return HarvestExtractor()
    if method == "crepe":
        return CrepeExtractor(device=device)
    raise ValueError(f"Unknown F0 method: {method}")


class HybridF0Extractor(F0Extractor):
    """
    Conservative hybrid extractor.

    RMVPE remains the primary estimator. CREPE is only allowed to repair short,
    high-confidence dropouts that sit inside already-voiced context.
    """

    def __init__(
        self,
        rmvpe_path: str,
        sample_rate: int = 16000,
        hop_length: int = 160,
        device: str = "cuda",
        crepe_threshold: float = 0.05,
        max_fill_ratio: float = 0.02,
        max_fill_frames: int = 320,
        context_radius: int = 6,
    ):
        super().__init__(sample_rate, hop_length)
        self.device = device
        self.rmvpe = RMVPEExtractor(rmvpe_path, sample_rate, hop_length, device)
        self.crepe = None
        self.crepe_threshold = float(crepe_threshold)
        self.max_fill_ratio = float(max_fill_ratio)
        self.max_fill_frames = int(max_fill_frames)
        self.context_radius = int(context_radius)

    def _load_crepe(self) -> None:
        if self.crepe is not None:
            return

        try:
            self.crepe = CrepeExtractor(
                self.sample_rate,
                self.hop_length,
                self.device,
            )
        except ImportError:
            print("Warning: torchcrepe is unavailable, hybrid falls back to RMVPE")
            self.crepe = False

    def extract(self, audio: np.ndarray) -> np.ndarray:
        f0_rmvpe = self.rmvpe.extract(audio)

        self._load_crepe()
        if self.crepe is False:
            return f0_rmvpe

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
            return_periodicity=True,
        )
        f0_crepe = f0_crepe.squeeze(0).cpu().numpy()
        confidence = confidence.squeeze(0).cpu().numpy()

        min_len = min(len(f0_rmvpe), len(f0_crepe), len(confidence))
        if min_len <= 0:
            return f0_rmvpe

        f0_rmvpe = np.asarray(f0_rmvpe[:min_len], dtype=np.float32)
        f0_crepe = np.asarray(f0_crepe[:min_len], dtype=np.float32)
        confidence = np.asarray(confidence[:min_len], dtype=np.float32)

        fill_mask = build_conservative_crepe_fill_mask(
            f0_rmvpe=f0_rmvpe,
            f0_crepe=f0_crepe,
            confidence=confidence,
            confidence_threshold=self.crepe_threshold,
            max_ratio=self.max_fill_ratio,
            max_frames=self.max_fill_frames,
            context_radius=self.context_radius,
        )

        if not np.any(fill_mask):
            return f0_rmvpe

        f0_hybrid = f0_rmvpe.copy()
        f0_hybrid[fill_mask] = f0_crepe[fill_mask]
        return f0_hybrid


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    """Shift F0 by semitones."""

    factor = 2 ** (semitones / 12)
    return f0 * factor
