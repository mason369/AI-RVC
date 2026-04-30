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


def _frame_curve_to_sample_curve(curve: np.ndarray, target_length: int, hop_length: int) -> np.ndarray:
    """Expand a frame-level curve to sample length using linear interpolation."""
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if curve.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    frame_positions = np.arange(curve.size, dtype=np.float32)
    sample_positions = np.arange(target_length, dtype=np.float32) / max(float(hop_length), 1.0)
    expanded = np.interp(sample_positions, frame_positions, curve, left=curve[0], right=curve[-1])
    return np.asarray(expanded, dtype=np.float32)


def _apply_adaptive_vocal_ducking(
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Duck accompaniment slightly under active vocals so default mixes keep the lead forward."""
    vocals = np.asarray(vocals, dtype=np.float32)
    accompaniment = np.asarray(accompaniment, dtype=np.float32)
    if vocals.ndim == 1:
        vocals = vocals.reshape(1, -1)
    if accompaniment.ndim == 1:
        accompaniment = accompaniment.reshape(1, -1)

    aligned_len = min(vocals.shape[-1], accompaniment.shape[-1])
    if aligned_len <= 2048:
        return accompaniment

    vocal_mono = vocals[:, :aligned_len].mean(axis=0)
    accompaniment_mono = accompaniment[:, :aligned_len].mean(axis=0)
    eps = 1e-8
    frame_length = 2048
    hop_length = 512

    vocal_rms = librosa.feature.rms(
        y=vocal_mono,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float32)
    accompaniment_rms = librosa.feature.rms(
        y=accompaniment_mono,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float32)

    frame_count = min(vocal_rms.size, accompaniment_rms.size)
    if frame_count <= 4:
        return accompaniment

    vocal_rms = vocal_rms[:frame_count]
    accompaniment_rms = accompaniment_rms[:frame_count]

    vocal_db = 20.0 * np.log10(vocal_rms + eps)
    vocal_peak_db = float(np.percentile(vocal_db, 95))
    activity = np.square(np.clip((vocal_db - (vocal_peak_db - 24.0)) / 11.0, 0.0, 1.0))
    accompaniment_db = 20.0 * np.log10(accompaniment_rms + eps)
    dense_backing = np.clip(
        (accompaniment_db - float(np.percentile(accompaniment_db, 65)) + 4.0) / 10.0,
        0.0,
        1.0,
    )

    vocal_global_rms = float(np.sqrt(np.mean(np.square(vocal_mono)) + 1e-12))
    accompaniment_global_rms = float(np.sqrt(np.mean(np.square(accompaniment_mono)) + 1e-12))
    balance_ratio = float(vocal_global_rms / (accompaniment_global_rms + 1e-12))
    duck_need = float(np.clip((0.92 - balance_ratio) / 0.34, 0.0, 1.0))
    if duck_need <= 0.02:
        return accompaniment

    smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    smooth_kernel /= np.sum(smooth_kernel)
    duck_curve = activity * (0.50 + 0.50 * dense_backing) * (0.45 + 0.55 * duck_need)
    duck_curve = np.convolve(duck_curve, smooth_kernel, mode="same")

    # Avoid abrupt accompaniment drops exactly when vocals enter.
    activity_delta = np.diff(activity, prepend=activity[:1])
    onset_curve = np.clip(activity_delta / 0.12, 0.0, 1.0).astype(np.float32)
    onset_curve = np.convolve(onset_curve, smooth_kernel, mode="same")
    duck_curve *= (1.0 - 0.38 * onset_curve)

    attack_frames = max(2, int(0.18 * sr / hop_length))
    release_frames = max(4, int(0.30 * sr / hop_length))
    attack_step = 1.0 / float(attack_frames)
    release_step = 1.0 / float(release_frames)
    smoothed_curve = np.empty_like(duck_curve, dtype=np.float32)
    smoothed_curve[0] = float(np.clip(duck_curve[0], 0.0, 1.0))
    for i in range(1, duck_curve.size):
        target = float(np.clip(duck_curve[i], 0.0, 1.0))
        previous = float(smoothed_curve[i - 1])
        if target > previous:
            smoothed_curve[i] = min(target, previous + attack_step)
        else:
            smoothed_curve[i] = max(target, previous - release_step)

    duck_curve = np.clip(smoothed_curve, 0.0, 1.0).astype(np.float32)
    sample_curve = _frame_curve_to_sample_curve(duck_curve, accompaniment.shape[-1], hop_length)
    max_duck_db = 1.5 + 1.4 * duck_need
    gain_curve = np.power(10.0, -(max_duck_db * sample_curve) / 20.0).astype(np.float32)

    ducked = accompaniment.copy().astype(np.float32)
    ducked *= gain_curve[np.newaxis, :]

    if log:
        log.detail(
            "Adaptive mix ducking: "
            f"balance={balance_ratio:.3f}, "
            f"avg_duck={max_duck_db * float(np.mean(sample_curve)):.2f}dB, "
            f"max_duck={max_duck_db * float(np.max(sample_curve)):.2f}dB, "
            f"onset_suppression={float(np.max(onset_curve)):.2f}"
        )

    return ducked


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
        log.detail("默认混音保持伴奏电平连续，跳过自动人声侧链压低")

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
