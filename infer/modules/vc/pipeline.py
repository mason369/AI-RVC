import os
import sys
import traceback
import logging

logger = logging.getLogger(__name__)

from functools import lru_cache
from time import time as ttime

import faiss
import librosa
import numpy as np
import parselmouth
import pyworld
import torch
import torch.nn.functional as F
import torchcrepe
from scipy import signal
from typing import Optional

now_dir = os.getcwd()
sys.path.append(now_dir)

# 导入彩色日志
try:
    from lib.logger import log
except ImportError:
    log = None

from lib.audio import soft_clip
from infer.quality_policy import compute_breath_preserving_energy_gates

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    gain = torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))
    # Reduced upper clamp: 4.0x over-amplifies noise in quiet sections,
    # producing buzzy/electronic artifacts. 2.0x is sufficient for RMS matching.
    gain = torch.clamp(gain, 0.3, 2.0)
    data2 *= gain.numpy()
    return data2


def repair_f0(
    f0: np.ndarray,
    max_gap: int = 6,
    mask: Optional[np.ndarray] = None,
    min_mask_ratio: float = 0.6,
) -> np.ndarray:
    """Fill short unvoiced gaps in F0 to reduce crack/tearing artifacts."""
    if f0 is None or len(f0) == 0:
        return f0
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32, copy=False)
    voiced = f0 > 0
    if voiced.sum() < 2:
        return f0

    if mask is not None:
        mask = mask.astype(bool, copy=False)
        if len(mask) < len(f0):
            mask = np.pad(mask, (0, len(f0) - len(mask)), mode="edge")
        else:
            mask = mask[: len(f0)]

    x = np.arange(len(f0))
    interp = np.interp(x, x[voiced], f0[voiced])

    zero_idx = np.where(~voiced)[0]
    if zero_idx.size == 0:
        return f0

    run_start = zero_idx[0]
    prev = zero_idx[0]
    for idx in zero_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        run_end = prev
        run_len = run_end - run_start + 1
        if run_len <= max_gap and run_start > 0 and run_end < len(f0) - 1:
            if mask is None or (mask[run_start : run_end + 1].mean() >= min_mask_ratio):
                f0[run_start : run_end + 1] = interp[run_start : run_end + 1]
        run_start = idx
        prev = idx
    run_end = prev
    run_len = run_end - run_start + 1
    if run_len <= max_gap and run_start > 0 and run_end < len(f0) - 1:
        if mask is None or (mask[run_start : run_end + 1].mean() >= min_mask_ratio):
            f0[run_start : run_end + 1] = interp[run_start : run_end + 1]

    return f0


def _normalize_rmvpe_hybrid_mode(mode: Optional[str]) -> str:
    """Normalize user-facing hybrid mode aliases to internal fallback modes."""
    normalized = str(mode or "off").strip().lower()
    if normalized in {"", "off", "none", "strict", "official", "rmvpe_strict", "rmvpe-strict", "raw", "rmvpe"}:
        return "off"
    if normalized in {
        "fallback",
        "smart",
        "rmvpe+fallback",
        "rmvpe_fallback",
        "rmvpe-fallback",
        "hybrid_fallback",
        "hybrid-fallback",
        "hybrid",
        "auto",
        "harvest",
        "harvest_fallback",
        "harvest-fallback",
    }:
        return "fallback"
    return normalized


def _build_protect_mix_curve(pitchf: torch.Tensor, protect: float) -> torch.Tensor:
    """Create a smooth protect curve for voiced/unvoiced transitions."""
    protect = float(np.clip(protect, 0.0, 1.0))
    if protect >= 1.0:
        return torch.ones_like(pitchf, dtype=torch.float32)

    voiced = (pitchf > 0).detach().float().cpu().numpy()
    if voiced.ndim == 2:
        voiced_curve = voiced[0]
    else:
        voiced_curve = voiced.reshape(-1)

    smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    smooth_kernel /= np.sum(smooth_kernel)
    voiced_curve = np.convolve(voiced_curve, smooth_kernel, mode="same")
    voiced_curve = np.convolve(voiced_curve, smooth_kernel, mode="same")
    voiced_curve = np.clip(voiced_curve, 0.0, 1.0)

    mix_curve = protect + (1.0 - protect) * voiced_curve
    mix_curve = torch.from_numpy(mix_curve.astype(np.float32)).to(pitchf.device)
    if pitchf.ndim == 2:
        mix_curve = mix_curve.unsqueeze(0)
    return mix_curve


def _compute_energy_mask(
    audio: np.ndarray,
    hop_length: int,
    frame_length: int = 1024,
    threshold_db: float = -50.0,
) -> np.ndarray:
    """Return frames considered voiced based on RMS energy."""
    if audio is None or len(audio) == 0:
        return np.zeros(0, dtype=bool)
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    rms_db = 20 * np.log10(rms + 1e-6)
    ref_db = np.percentile(rms_db, 95)
    gate_db = ref_db + threshold_db
    return rms_db >= gate_db


def _compute_harvest_f0(
    audio: np.ndarray,
    sr: int,
    f0_min: float,
    f0_max: float,
    frame_period: float = 10.0,
) -> np.ndarray:
    """Compute Harvest F0 for fallback filling."""
    audio = audio.astype(np.double, copy=False)
    f0, t = pyworld.harvest(
        audio,
        fs=sr,
        f0_ceil=f0_max,
        f0_floor=f0_min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, sr)
    return f0


def _compute_crepe_f0(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    f0_min: float,
    f0_max: float,
    device: str,
    periodicity_threshold: float = 0.1,
    return_periodicity: bool = False,
) -> np.ndarray:
    """Compute CREPE F0 for fallback filling."""
    audio_tensor = torch.tensor(np.copy(audio))[None].float()
    f0, pd = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length,
        f0_min,
        f0_max,
        "full",
        batch_size=512,
        device=device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0 = f0[0].cpu().numpy()
    pd = pd[0].cpu().numpy()
    if periodicity_threshold is not None:
        f0[pd < periodicity_threshold] = 0
    if return_periodicity:
        return f0, pd
    return f0


def _stabilize_f0(
    f0: np.ndarray,
    max_semitones: float = 6.0,
    window: int = 2,
    octave_fix: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Stabilize F0 by correcting octave errors and extreme jumps."""
    if f0 is None or len(f0) == 0:
        return f0, 0, 0
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32, copy=True)
    voiced_idx = np.where(f0 > 0)[0]
    if voiced_idx.size < 3:
        return f0, 0, 0
    win = max(1, int(window))
    max_semi = float(max_semitones)
    eps = 1e-6
    octave_fix_count = 0
    outlier_count = 0

    for i in voiced_idx:
        start = max(0, i - win)
        end = min(len(f0), i + win + 1)
        neighbors = f0[start:end]
        neighbors = neighbors[neighbors > 0]
        if neighbors.size < 3:
            continue
        med = float(np.median(neighbors))
        if med <= 0:
            continue

        if octave_fix:
            ratio = f0[i] / (med + eps)
            if 1.9 < ratio < 2.1:
                f0[i] = f0[i] * 0.5
                octave_fix_count += 1
            elif 0.48 < ratio < 0.52:
                f0[i] = f0[i] * 2.0
                octave_fix_count += 1

        if max_semi > 0:
            semi_diff = 12.0 * abs(np.log2((f0[i] + eps) / (med + eps)))
            if semi_diff > max_semi:
                f0[i] = med
                outlier_count += 1

    return f0, octave_fix_count, outlier_count


def _limit_f0_slope(
    f0: np.ndarray,
    max_semitones: float = 8.0,
) -> tuple[np.ndarray, int]:
    """Limit frame-to-frame pitch jumps to reduce harsh transitions."""
    if f0 is None or len(f0) == 0:
        return f0, 0
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32, copy=True)
    max_semi = float(max_semitones)
    if max_semi <= 0:
        return f0, 0
    max_ratio = 2 ** (max_semi / 12.0)
    min_ratio = 1.0 / max_ratio
    changed = 0
    prev = None
    for i in range(len(f0)):
        if f0[i] <= 0:
            continue
        if prev is None:
            prev = f0[i]
            continue
        ratio = f0[i] / (prev + 1e-6)
        if ratio > max_ratio:
            f0[i] = prev * max_ratio
            changed += 1
        elif ratio < min_ratio:
            f0[i] = prev * min_ratio
            changed += 1
        prev = f0[i]
    return f0, changed


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.disable_chunking = bool(getattr(config, "disable_chunking", False))
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device
        self.f0_min = float(getattr(config, "f0_min", 50))
        self.f0_max = float(getattr(config, "f0_max", 1100))
        if self.f0_max <= self.f0_min:
            self.f0_max = max(self.f0_min + 1.0, 1100.0)
        self.rmvpe_threshold = float(getattr(config, "rmvpe_threshold", 0.02))
        self.f0_energy_threshold_db = float(getattr(config, "f0_energy_threshold_db", -50))
        self.f0_hybrid_mode = _normalize_rmvpe_hybrid_mode(
            getattr(config, "f0_hybrid_mode", "off")
        )
        self.rmvpe_strict_modes = {
            "",
            "off",
            "none",
            "strict",
            "official",
            "rmvpe_strict",
            "rmvpe-strict",
        }
        self.rmvpe_fallback_modes = {
            "fallback",
            "smart",
            "rmvpe+fallback",
            "rmvpe_fallback",
            "rmvpe-fallback",
            "hybrid_fallback",
            "hybrid-fallback",
        }
        self.crepe_pd_threshold = float(getattr(config, "crepe_pd_threshold", 0.1))
        self.crepe_force_ratio = float(getattr(config, "crepe_force_ratio", 0.05))
        self.crepe_replace_semitones = float(getattr(config, "crepe_replace_semitones", 0.0))
        self.unvoiced_feature_gate_floor = float(
            getattr(config, "unvoiced_feature_gate_floor", 0.28)
        )
        self.breath_active_margin_db = float(
            getattr(config, "breath_active_margin_db", 52.0)
        )
        self.f0_fallback_context_radius = int(getattr(config, "f0_fallback_context_radius", 24))
        self.f0_fallback_repair_gap = int(getattr(config, "f0_fallback_repair_gap", 12))
        self.f0_fallback_post_gap = int(getattr(config, "f0_fallback_post_gap", 10))
        self.f0_fallback_use_crepe = bool(getattr(config, "f0_fallback_use_crepe", True))
        self.f0_fallback_crepe_max_ratio = float(getattr(config, "f0_fallback_crepe_max_ratio", 0.02))
        self.f0_fallback_crepe_max_frames = int(getattr(config, "f0_fallback_crepe_max_frames", 320))
        self.f0_stabilize = bool(getattr(config, "f0_stabilize", False))
        self.f0_stabilize_window = int(getattr(config, "f0_stabilize_window", 2))
        self.f0_stabilize_max_semitones = float(
            getattr(config, "f0_stabilize_max_semitones", 6.0)
        )
        self.f0_stabilize_octave = bool(getattr(config, "f0_stabilize_octave", True))
        self.f0_rate_limit = bool(getattr(config, "f0_rate_limit", False))
        self.f0_rate_limit_semitones = float(
            getattr(config, "f0_rate_limit_semitones", 8.0)
        )
        if self.crepe_force_ratio < 0:
            self.crepe_force_ratio = 0.0
        if self.crepe_pd_threshold < 0:
            self.crepe_pd_threshold = 0.0
        if self.crepe_replace_semitones < 0:
            self.crepe_replace_semitones = 0.0
        if self.unvoiced_feature_gate_floor < 0.05:
            self.unvoiced_feature_gate_floor = 0.05
        if self.unvoiced_feature_gate_floor > 1.0:
            self.unvoiced_feature_gate_floor = 1.0
        if self.breath_active_margin_db < 1.0:
            self.breath_active_margin_db = 1.0
        if self.f0_fallback_context_radius < 1:
            self.f0_fallback_context_radius = 1
        if self.f0_fallback_repair_gap < 0:
            self.f0_fallback_repair_gap = 0
        if self.f0_fallback_post_gap < 0:
            self.f0_fallback_post_gap = 0
        if self.f0_fallback_crepe_max_ratio < 0:
            self.f0_fallback_crepe_max_ratio = 0.0
        if self.f0_fallback_crepe_max_frames < 0:
            self.f0_fallback_crepe_max_frames = 0
        if self.f0_stabilize_window < 1:
            self.f0_stabilize_window = 1
        if self.f0_stabilize_max_semitones < 0:
            self.f0_stabilize_max_semitones = 0.0
        if self.f0_rate_limit_semitones < 0:
            self.f0_rate_limit_semitones = 0.0

        if log:
            log.detail(f"Pipeline初始化: 目标采样率={tgt_sr}Hz")
            log.detail(f"设备: {self.device}, 半精度: {self.is_half}")
            log.detail(f"x_pad={self.x_pad}, x_query={self.x_query}, x_center={self.x_center}, x_max={self.x_max}")
            log.detail(f"禁用分段: {self.disable_chunking}")
            log.detail(f"F0范围: {self.f0_min}-{self.f0_max}Hz, RMVPE阈值: {self.rmvpe_threshold}")
            log.detail(
                f"F0混合: {self.f0_hybrid_mode}, CREPE阈值: {self.crepe_pd_threshold}, "
                f"强制比率: {self.crepe_force_ratio}, 替换阈值(半音): {self.crepe_replace_semitones}"
            )
            log.detail(
                f"气声门控: 特征地板={self.unvoiced_feature_gate_floor:.2f}, "
                f"激活边界=ref-{self.breath_active_margin_db:.1f}dB"
            )
            log.detail(
                f"F0兜底: 上下文半径={self.f0_fallback_context_radius}, "
                f"预修补长度={self.f0_fallback_repair_gap}, 后修补长度={self.f0_fallback_post_gap}, "
                f"CREPE兜底={self.f0_fallback_use_crepe}, "
                f"CREPE最大占比={self.f0_fallback_crepe_max_ratio:.2%}, "
                f"CREPE最大帧数={self.f0_fallback_crepe_max_frames}"
            )
            log.detail(
                "RMVPE兜底: "
                f"{'on' if self.f0_hybrid_mode in self.rmvpe_fallback_modes else 'off'}"
            )
            log.detail(
                f"F0稳定器: {self.f0_stabilize}, 窗口: {self.f0_stabilize_window}, "
                f"最大跳变(半音): {self.f0_stabilize_max_semitones}, "
                f"八度修正: {self.f0_stabilize_octave}"
            )
            log.detail(
                f"F0限速: {self.f0_rate_limit}, 最大跳变/帧(半音): {self.f0_rate_limit_semitones}"
            )

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        inp_f0=None,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = self.f0_min
        f0_max = self.f0_max
        # Mel quantization range MUST match training (50-1100Hz) regardless of
        # extraction range, otherwise pitch embedding indices shift and the
        # model produces degraded output on all notes.
        f0_mel_min = 1127 * np.log(1 + 50.0 / 700)
        f0_mel_max = 1127 * np.log(1 + 1100.0 / 700)

        if log:
            log.progress(f"提取F0: 方法={f0_method}")
            log.detail(f"时间步长: {time_step:.2f}ms, F0范围: {f0_min}-{f0_max}Hz")
            log.detail(f"音频长度: {len(x)} 样本, p_len: {p_len}")

        # Normalize direct hybrid requests to the conservative RMVPE fallback path.
        if f0_method == "hybrid":
            f0_method = "rmvpe"
            original_hybrid_mode = self.f0_hybrid_mode
            if self.f0_hybrid_mode not in self.rmvpe_fallback_modes:
                self.f0_hybrid_mode = "fallback"
            restore_hybrid_mode = True
        else:
            restore_hybrid_mode = False

        if f0_method == "pm":
            if log:
                log.detail("使用Parselmouth提取F0...")
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
            if log:
                log.detail(f"PM F0提取完成: shape={f0.shape}")
        elif f0_method == "harvest":
            if log:
                log.detail("使用PyWorld Harvest提取F0...")
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
                if log:
                    log.detail(f"应用中值滤波: radius={filter_radius}")
            if log:
                log.detail(f"Harvest F0提取完成: shape={f0.shape}")
        elif f0_method == "crepe":
            if log:
                log.detail("使用CREPE提取F0...")
            model = "full"
            # Pick a batch size that doesn't cause memory errors on your gpu
            batch_size = 512
            if log:
                log.detail(f"CREPE模型: {model}, batch_size: {batch_size}")
            # Compute pitch using first gpu
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
            if log:
                log.detail(f"CREPE F0提取完成: shape={f0.shape}")
        elif f0_method == "rmvpe":
            if self.f0_hybrid_mode in ("crepe", "crepe_only", "crepe-only"):
                if log:
                    log.detail("使用CREPE全量F0 (质量优先)...")
                f0 = _compute_crepe_f0(
                    x,
                    self.sr,
                    self.window,
                    f0_min,
                    f0_max,
                    self.device,
                    periodicity_threshold=self.crepe_pd_threshold,
                )
                if log:
                    log.detail(f"CREPE F0提取完成: shape={f0.shape}")
            else:
                if log:
                    log.detail("使用RMVPE提取F0...")
                if not hasattr(self, "model_rmvpe"):
                    from infer.lib.rmvpe import RMVPE

                    rmvpe_path = "%s/rmvpe.pt" % os.environ["rmvpe_root"]
                    logger.info(
                        "Loading rmvpe model,%s" % rmvpe_path
                    )
                    if log:
                        log.model(f"加载RMVPE模型: {rmvpe_path}")
                    self.model_rmvpe = RMVPE(
                        rmvpe_path,
                        is_half=self.is_half,
                        device=self.device,
                    )
                    if log:
                        log.success("RMVPE模型加载完成")
                # Slightly lower threshold to reduce short unvoiced dropouts
                f0 = self.model_rmvpe.infer_from_audio(x, thred=self.rmvpe_threshold)
                if log:
                    log.detail(f"RMVPE F0提取完成: shape={f0.shape}")

                if "privateuseone" in str(self.device):  # clean ortruntime memory
                    del self.model_rmvpe.model
                    del self.model_rmvpe
                    logger.info("Cleaning ortruntime memory")
                    if log:
                        log.detail("清理ONNX Runtime内存")

                if self.f0_hybrid_mode in ("rmvpe+crepe", "rmvpe_crepe", "hybrid", "rmvpe-crepe"):
                    if log:
                        log.detail("启用RMVPE+CREPE混合F0 (质量优先)...")
                    crepe_f0, crepe_pd = _compute_crepe_f0(
                        x,
                        self.sr,
                        self.window,
                        f0_min,
                        f0_max,
                        self.device,
                        periodicity_threshold=self.crepe_pd_threshold,
                        return_periodicity=True,
                    )
                    if len(crepe_f0) < len(f0):
                        crepe_f0 = np.pad(crepe_f0, (0, len(f0) - len(crepe_f0)), mode="edge")
                        crepe_pd = np.pad(crepe_pd, (0, len(f0) - len(crepe_pd)), mode="edge")
                    else:
                        crepe_f0 = crepe_f0[: len(f0)]
                        crepe_pd = crepe_pd[: len(f0)]

                    crepe_mask = crepe_f0 > 0
                    drop_ratio = float(np.sum(f0 <= 0)) / max(len(f0), 1)
                    replace_mask = (f0 <= 0) & crepe_mask
                    if drop_ratio >= self.crepe_force_ratio:
                        replace_mask = crepe_mask

                    if self.crepe_replace_semitones > 0:
                        both_voiced = (f0 > 0) & crepe_mask
                        if np.any(both_voiced):
                            diff_semi = np.zeros_like(f0, dtype=np.float32)
                            diff_semi[both_voiced] = np.abs(
                                12.0
                                * np.log2(
                                    (f0[both_voiced] + 1e-6) / (crepe_f0[both_voiced] + 1e-6)
                                )
                            )
                            replace_mask |= both_voiced & (diff_semi >= self.crepe_replace_semitones)

                    replaced = int(np.sum(replace_mask))
                    f0[replace_mask] = crepe_f0[replace_mask]
                    if log:
                        log.detail(
                            f"CREPE混合完成: 掉线比率={drop_ratio:.2%}, "
                            f"替换帧={replaced}/{len(f0)}"
                        )

        f0 *= pow(2, f0_up_key / 12)
        if log:
            log.detail(f"应用音调偏移: {f0_up_key} 半音, 倍率: {pow(2, f0_up_key / 12):.4f}")
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            if log:
                log.detail("应用自定义F0曲线...")
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        else:
            use_rmvpe_fallback = (
                f0_method == "rmvpe"
                and self.f0_hybrid_mode not in self.rmvpe_strict_modes
                and self.f0_hybrid_mode in self.rmvpe_fallback_modes
            )

            if use_rmvpe_fallback:
                energy_mask = _compute_energy_mask(
                    x, hop_length=self.window, threshold_db=self.f0_energy_threshold_db
                )
                if energy_mask.size > 0:
                    if len(energy_mask) < len(f0):
                        energy_mask = np.pad(
                            energy_mask, (0, len(f0) - len(energy_mask)), mode="edge"
                        )
                    else:
                        energy_mask = energy_mask[: len(f0)]
                else:
                    energy_mask = None

                # Repair short unvoiced gaps only when fallback mode is explicitly enabled.
                f0 = repair_f0(
                    f0,
                    max_gap=self.f0_fallback_repair_gap,
                    mask=energy_mask,
                )

                # Conservative F0 fallback:
                # only fill dropouts that are surrounded by voiced context.
                if energy_mask is not None:
                    voiced_seed = f0 > 0
                    if np.any(voiced_seed):
                        idx = np.arange(len(f0))
                        left_seen = np.where(voiced_seed, idx, -10**9)
                        left_seen = np.maximum.accumulate(left_seen)
                        right_seen = np.where(voiced_seed, idx, 10**9)
                        right_seen = np.minimum.accumulate(right_seen[::-1])[::-1]
                        context_radius = self.f0_fallback_context_radius
                        left_near = (idx - left_seen) <= context_radius
                        right_near = (right_seen - idx) <= context_radius
                        voiced_context = left_near & right_near
                    else:
                        voiced_context = np.zeros_like(f0, dtype=bool)

                    need_fill = (f0 <= 0) & energy_mask & voiced_context
                    if np.any(need_fill):
                        if log:
                            log.detail(
                                f"RMVPE掉线帧(主唱上下文): {int(need_fill.sum())}/{len(f0)}，启用保守兜底"
                            )

                        f0_min_fb = max(30.0, f0_min - 20.0)
                        f0_max_fb = min(1800.0, f0_max + 200.0)
                        f0_fb = _compute_harvest_f0(x, self.sr, f0_min_fb, f0_max_fb, 10.0)
                        if len(f0_fb) < len(f0):
                            f0_fb = np.pad(f0_fb, (0, len(f0) - len(f0_fb)), mode="edge")
                        else:
                            f0_fb = f0_fb[: len(f0)]

                        fill_mask = need_fill & (f0_fb > 0)
                        f0[fill_mask] = f0_fb[fill_mask]

                        need_fill2 = (f0 <= 0) & energy_mask & voiced_context
                        need_fill2_count = int(np.sum(need_fill2))
                        need_fill2_ratio = float(need_fill2_count) / max(len(f0), 1)
                        if np.any(need_fill2) and self.f0_fallback_use_crepe:
                            allow_crepe_fallback = (
                                need_fill2_count <= self.f0_fallback_crepe_max_frames
                                and need_fill2_ratio <= self.f0_fallback_crepe_max_ratio
                            )
                        else:
                            allow_crepe_fallback = False

                        if np.any(need_fill2) and allow_crepe_fallback:
                            if log:
                                log.detail(
                                    f"Harvest后仍掉线(主唱上下文): {int(need_fill2.sum())}/{len(f0)}，启用CREPE兜底"
                                )
                            f0_cr = _compute_crepe_f0(
                                x,
                                self.sr,
                                self.window,
                                f0_min_fb,
                                f0_max_fb,
                                self.device,
                                periodicity_threshold=self.crepe_pd_threshold,
                            )
                            if len(f0_cr) < len(f0):
                                f0_cr = np.pad(f0_cr, (0, len(f0) - len(f0_cr)), mode="edge")
                            else:
                                f0_cr = f0_cr[: len(f0)]

                            # Require cross-estimator agreement when both estimators are voiced.
                            both_voiced = (f0_cr > 0) & (f0_fb > 0)
                            agree_mask = np.zeros_like(f0, dtype=bool)
                            if np.any(both_voiced):
                                semitone_diff = np.abs(
                                    12.0 * np.log2((f0_cr + 1e-6) / (f0_fb + 1e-6))
                                )
                                agree_mask = both_voiced & (semitone_diff <= 2.0)

                            fill_mask2 = need_fill2 & (
                                ((f0_cr > 0) & (f0_fb <= 0)) | agree_mask
                            )
                            f0[fill_mask2] = f0_cr[fill_mask2]
                        elif np.any(need_fill2) and log:
                            log.detail(
                                f"Harvest后仍掉线(主唱上下文): {need_fill2_count}/{len(f0)}，"
                                "已跳过CREPE兜底（超出保守阈值）"
                            )

                        final_drop = (f0 <= 0) & energy_mask & voiced_context
                        if np.any(final_drop) and log:
                            log.detail(
                                f"保守兜底后保留无声帧: {int(final_drop.sum())}/{len(f0)}"
                            )

                        # Only smooth short, context-consistent gaps.
                        f0 = repair_f0(
                            f0,
                            max_gap=self.f0_fallback_post_gap,
                            mask=voiced_context,
                        )
            elif f0_method == "rmvpe" and log:
                log.detail("RMVPE严格模式: 不启用Harvest/CREPE兜底，仅使用RMVPE原始结果")

        if self.f0_stabilize:
            f0, octave_fixed, outlier_fixed = _stabilize_f0(
                f0,
                max_semitones=self.f0_stabilize_max_semitones,
                window=self.f0_stabilize_window,
                octave_fix=self.f0_stabilize_octave,
            )
            if log:
                log.detail(
                    f"F0稳定器完成: 八度修正={octave_fixed}, 跳变修正={outlier_fixed}"
                )
        if self.f0_rate_limit:
            f0, rate_fixed = _limit_f0_slope(
                f0,
                max_semitones=self.f0_rate_limit_semitones,
            )
            if log:
                log.detail(f"F0限速完成: 修正帧={rate_fixed}")

        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()

        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        if log:
            log.detail(f"F0处理完成: coarse shape={f0_coarse.shape}, bak shape={f0bak.shape}")

        # 恢复原始hybrid模式设置
        if restore_hybrid_mode:
            self.f0_hybrid_mode = original_hybrid_mode

        return f0_coarse, f0bak

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        energy_ref_db=None,
    ):  # ,file_index,file_big_npy
        if log:
            log.detail(f"VC推理: 音频长度={len(audio0)}, 版本={version}, 保护={protect}")

        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        if log:
            log.detail(f"HuBERT输出层: {inputs['output_layer']}")

        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        if log:
            log.detail(f"特征提取完成: shape={feats.shape}")

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            if log:
                log.detail(f"应用索引检索: index_rate={index_rate}")
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )
            if log:
                log.detail("索引混合完成")

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            if log:
                log.detail(f"应用保护: protect={protect}")
            pitchff = _build_protect_mix_curve(pitchf, protect).unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()

        # --- 能量感知软门控（所有特征操作完成后、推理前）---
        # 使用连续衰减曲线代替硬二值化，避免静音/有声边界的撕裂伪影。
        _p_len_val = p_len.item() if isinstance(p_len, torch.Tensor) else int(p_len)
        _audio_np = audio0.astype(np.float32)
        _frame_rms = librosa.feature.rms(
            y=_audio_np, frame_length=self.window * 2, hop_length=self.window, center=True
        )[0]
        if _frame_rms.ndim > 1:
            _frame_rms = _frame_rms[0]
        if len(_frame_rms) > _p_len_val:
            _frame_rms = _frame_rms[:_p_len_val]
        elif len(_frame_rms) < _p_len_val:
            _frame_rms = np.pad(_frame_rms, (0, _p_len_val - len(_frame_rms)), mode='edge')

        _energy_db = 20.0 * np.log10(_frame_rms + 1e-8)
        _ref = energy_ref_db if energy_ref_db is not None else float(np.percentile(_energy_db, 95))
        _unvoiced_mask = None
        if pitchf is not None:
            _pitchf_np = pitchf[0].detach().float().cpu().numpy()
            if len(_pitchf_np) > _p_len_val:
                _pitchf_np = _pitchf_np[:_p_len_val]
            elif len(_pitchf_np) < _p_len_val:
                _pitchf_np = np.pad(_pitchf_np, (0, _p_len_val - len(_pitchf_np)), mode="edge")
            _unvoiced_mask = _pitchf_np <= 1e-4

        _feat_gate_curve, _f0_gate_curve = compute_breath_preserving_energy_gates(
            energy_db=_energy_db,
            ref_db=_ref,
            unvoiced_mask=_unvoiced_mask,
            quiet_floor=0.05,
            breath_floor=self.unvoiced_feature_gate_floor,
            breath_active_margin_db=self.breath_active_margin_db,
            transition_width_db=6.0,
        )
        # Smooth temporally
        _sm = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        _sm /= _sm.sum()
        _feat_gate_curve = np.convolve(_feat_gate_curve, _sm, mode='same')[:_p_len_val]
        _f0_gate_curve = np.convolve(_f0_gate_curve, _sm, mode='same')[:_p_len_val]
        _feat_gate_curve = np.clip(_feat_gate_curve, 0.05, 1.0)
        _f0_gate_curve = np.clip(_f0_gate_curve, 0.05, 1.0)
        if log and _unvoiced_mask is not None:
            _breath_frames = int(np.sum(_feat_gate_curve > (_f0_gate_curve + 1e-4)))
            if _breath_frames > 0:
                log.detail(
                    f"气声保护门控: { _breath_frames }/{ _p_len_val } 帧保留更高特征底噪"
                )

        # Apply soft gate to features
        _feat_len = feats.shape[1]
        if len(_feat_gate_curve) > _feat_len:
            _feat_gate = _feat_gate_curve[:_feat_len]
        elif len(_feat_gate_curve) < _feat_len:
            _feat_gate = np.pad(_feat_gate_curve, (0, _feat_len - len(_feat_gate_curve)), mode='constant', constant_values=1.0)
        else:
            _feat_gate = _feat_gate_curve
        _gate_t = torch.from_numpy(_feat_gate.astype(np.float32)).to(feats.device).unsqueeze(0).unsqueeze(-1)
        feats = feats * _gate_t

        # F0 soft gating: consistently soft-attenuate both pitch confidence and pitch value
        if pitch is not None and pitchf is not None:
            _pitch_len = pitch.shape[1]
            if len(_f0_gate_curve) > _pitch_len:
                _f0_gate = _f0_gate_curve[:_pitch_len]
            elif len(_f0_gate_curve) < _pitch_len:
                _f0_gate = np.pad(_f0_gate_curve, (0, _pitch_len - len(_f0_gate_curve)), mode='constant', constant_values=1.0)
            else:
                _f0_gate = _f0_gate_curve
            _f0_gate_t = torch.from_numpy(_f0_gate.astype(np.float32)).to(pitch.device).unsqueeze(0)
            pitchf = pitchf * _f0_gate_t
            # Soft-blend pitch toward silence bin (1) instead of hard switch
            _silence_pitch = torch.ones_like(pitch)
            _blend = _f0_gate_t.unsqueeze(-1) if _f0_gate_t.dim() < pitch.dim() else _f0_gate_t
            pitch = (pitch.float() * _blend + _silence_pitch.float() * (1.0 - _blend)).long()

        if log:
            log.detail("执行神经网络推理...")

        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1

        if log:
            log.detail(f"VC推理完成: 输出长度={len(audio1)}, 耗时={t2-t0:.3f}s")

        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=None,
    ):
        if log:
            log.progress("开始推理管道...")
            log.detail(f"输入音频: {input_audio_path}")
            log.detail(f"音频长度: {len(audio)} 样本 ({len(audio)/16000:.2f}秒)")
            log.config(f"F0方法: {f0_method}, 音调偏移: {f0_up_key}")
            log.config(f"索引率: {index_rate}, 滤波半径: {filter_radius}")
            log.config(f"目标采样率: {tgt_sr}Hz, 重采样: {resample_sr}Hz")
            log.config(f"RMS混合率: {rms_mix_rate}, 保护: {protect}")
            log.config(f"版本: {version}, F0启用: {if_f0}")

        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                if log:
                    log.model(f"加载索引文件: {file_index}")
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
                if log:
                    log.detail(f"索引加载完成: {index.ntotal} 个向量")
            except:
                traceback.print_exc()
                if log:
                    log.warning("索引加载失败，将不使用索引")
                index = big_npy = None
        else:
            index = big_npy = None
            if log:
                log.detail("未使用索引文件")

        if log:
            log.detail("应用高通滤波...")
        audio = signal.filtfilt(bh, ah, audio)

        # 全局能量参考（用于分段 vc() 的能量遮蔽阈值一致性）
        _global_rms = librosa.feature.rms(
            y=audio, frame_length=self.window * 2, hop_length=self.window, center=True
        )[0]
        if _global_rms.ndim > 1:
            _global_rms = _global_rms[0]
        if _global_rms.size > 0:
            _global_energy_db = 20.0 * np.log10(_global_rms + 1e-8)
            _global_ref_db = float(np.percentile(_global_energy_db, 95))
        else:
            _global_ref_db = -20.0

        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if not self.disable_chunking and audio_pad.shape[0] > self.t_max:
            if log:
                log.detail(f"音频较长，进行分段处理: {audio_pad.shape[0]} > {self.t_max}")
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
            if log:
                log.detail(f"分段数量: {len(opt_ts) + 1}")
        else:
            if log:
                if self.disable_chunking:
                    log.detail("已禁用分段，单次处理")
                else:
                    log.detail("音频较短，单次处理")

        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        if log:
            log.detail(f"填充后音频长度: {audio_pad.shape[0]}, p_len: {p_len}")

        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                if log:
                    log.detail(f"加载自定义F0文件: {f0_file.name}")
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
                if log:
                    log.detail(f"自定义F0加载完成: {inp_f0.shape}")
            except:
                traceback.print_exc()
                if log:
                    log.warning("自定义F0加载失败")

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            if log:
                log.progress("提取基频(F0)...")
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
            if log:
                log.success("F0提取完成")
        t2 = ttime()
        times[1] += t2 - t1
        if log:
            log.detail(f"F0提取耗时: {t2-t1:.3f}s")

        # 分段推理（带交叉淡入淡出消除边界撕裂）
        segment_count = len(opt_ts) + 1
        current_segment = 0

        # Crossfade length at target rate (~12ms). Each boundary segment
        # keeps this many extra samples from the normally-trimmed padding
        # region. The overlap between adjacent segments is 2 * _xfade_tgt.
        _xfade_tgt = min(int(0.012 * tgt_sr), self.t_pad_tgt // 4) if len(opt_ts) > 0 else 0

        def _trim_segment(raw, is_first, is_last):
            """Trim padding from vc() output, keeping crossfade overlap."""
            left = self.t_pad_tgt if is_first else (self.t_pad_tgt - _xfade_tgt)
            right = self.t_pad_tgt if is_last else (self.t_pad_tgt - _xfade_tgt)
            return raw[left : -right] if right > 0 else raw[left:]

        for idx, t in enumerate(opt_ts):
            current_segment += 1
            if log:
                log.progress(f"处理分段 {current_segment}/{segment_count}...")
            t = t // self.window * self.window
            if if_f0 == 1:
                raw = self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[s : t + self.t_pad2 + self.window],
                    pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                    pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    energy_ref_db=_global_ref_db,
                )
            else:
                raw = self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[s : t + self.t_pad2 + self.window],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    energy_ref_db=_global_ref_db,
                )
            audio_opt.append(_trim_segment(raw, is_first=(idx == 0), is_last=False))
            s = t

        # 最后一段
        if log:
            log.progress(f"处理分段 {segment_count}/{segment_count}...")
        if if_f0 == 1:
            raw = self.vc(
                model,
                net_g,
                sid,
                audio_pad[t:],
                pitch[:, t // self.window :] if t is not None else pitch,
                pitchf[:, t // self.window :] if t is not None else pitchf,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
                energy_ref_db=_global_ref_db,
            )
        else:
            raw = self.vc(
                model,
                net_g,
                sid,
                audio_pad[t:],
                None,
                None,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
                energy_ref_db=_global_ref_db,
            )
        audio_opt.append(_trim_segment(raw, is_first=(len(opt_ts) == 0), is_last=True))

        if log:
            log.detail("合并音频分段...")

        # Overlap-add crossfade: adjacent segments share 2*_xfade_tgt
        # samples of overlapping content (same original audio region
        # processed as part of different chunks). Linear crossfade
        # ensures amplitude-preserving smooth transition.
        if len(audio_opt) > 1 and _xfade_tgt > 0:
            overlap = 2 * _xfade_tgt
            result = audio_opt[0]
            for seg in audio_opt[1:]:
                xf = min(overlap, len(result), len(seg))
                if xf > 1:
                    fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
                    fade_in = 1.0 - fade_out
                    blended = result[-xf:] * fade_out + seg[:xf] * fade_in
                    result = np.concatenate([result[:-xf], blended, seg[xf:]])
                else:
                    result = np.concatenate([result, seg])
            audio_opt = result
        else:
            audio_opt = np.concatenate(audio_opt) if audio_opt else np.array([], dtype=np.float32)

        if rms_mix_rate != 1:
            if log:
                log.detail(f"应用RMS混合: rate={rms_mix_rate}")
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)

        if tgt_sr != resample_sr >= 16000:
            if log:
                log.detail(f"重采样: {tgt_sr}Hz -> {resample_sr}Hz")
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )

        peak_before_clip = float(np.max(np.abs(audio_opt)))
        audio_opt = soft_clip(audio_opt, threshold=0.9, ceiling=0.99)
        if log and peak_before_clip > 0.9:
            peak_after_clip = float(np.max(np.abs(audio_opt)))
            log.detail(
                f"音频软削波: 峰值 {peak_before_clip:.4f} -> {peak_after_clip:.4f}"
            )
        audio_opt = np.clip(audio_opt, -0.99, 0.99)
        audio_opt = (audio_opt * 32767.0).astype(np.int16)

        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if log:
                log.detail("已清理CUDA缓存")

        if log:
            log.success(f"推理管道完成: 输出长度={len(audio_opt)} 样本")

        return audio_opt

