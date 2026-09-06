# -*- coding: utf-8 -*-
"""
RVC 推理管道 - 端到端 AI 翻唱
"""
import os
import gc
import torch
import numpy as np
import faiss
from pathlib import Path
from typing import Optional, Tuple, Union
from scipy import signal as sp_signal

from lib.audio import load_audio, save_audio, normalize_audio, soft_clip
from lib.device import get_device, empty_device_cache, supports_fp16
from lib.logger import log
from infer.f0_extractor import get_f0_extractor, shift_f0, F0Method
from infer.rvc_version import inspect_rvc_model_version
from infer.contracts import inspect_checkpoint, validate_state_dict, read_index, retrieve_features, number, integer

# 48Hz 高通 Butterworth 滤波器（与官方管道一致，去除低频隆隆声）
_bh, _ah = sp_signal.butter(N=5, Wn=48, btype="high", fs=16000)


class VoiceConversionPipeline:
    """RVC 推理管道"""

    def __init__(self, device: str = "cuda"):
        """
        初始化管道

        Args:
            device: 计算设备 ("cuda" 或 "cpu")
        """
        self.device = get_device(device)
        self.hubert_model = None
        self.hubert_model_type = None
        self.voice_model = None
        self.index = None
        self.index_vectors = None
        self.uses_f0 = True
        self.f0_extractor = None
        self.spk_count = 1
        self.model_feature_dim = None
        self.model_version = "v2"  # 默认 v2（768 维）

        # 默认参数
        self.sample_rate = 16000  # HuBERT 输入采样率
        self.output_sr = 48000    # 输出采样率

    def unload_hubert(self):
        """卸载 HuBERT 模型释放显存"""
        if self.hubert_model is not None:
            self.hubert_model.cpu()
            del self.hubert_model
            self.hubert_model = None
            self.hubert_model_type = None
        gc.collect()
        empty_device_cache(self.device)

    def unload_f0_extractor(self):
        """卸载 F0 提取器释放显存"""
        if self.f0_extractor is not None:
            # RMVPEExtractor.model 是 RMVPE 包装类，内部有 model 和 mel_extractor
            if hasattr(self.f0_extractor, 'model') and self.f0_extractor.model is not None:
                rmvpe = self.f0_extractor.model
                # 卸载内部的 E2E 模型
                if hasattr(rmvpe, 'model') and rmvpe.model is not None:
                    rmvpe.model.cpu()
                    del rmvpe.model
                    rmvpe.model = None
                # 卸载 mel_extractor
                if hasattr(rmvpe, 'mel_extractor') and rmvpe.mel_extractor is not None:
                    rmvpe.mel_extractor.cpu()
                    del rmvpe.mel_extractor
                    rmvpe.mel_extractor = None
                del self.f0_extractor.model
                self.f0_extractor.model = None
            del self.f0_extractor
            self.f0_extractor = None
        gc.collect()
        empty_device_cache(self.device)

    def unload_voice_model(self):
        """卸载语音模型释放显存"""
        if self.voice_model is not None:
            self.voice_model.cpu()
            del self.voice_model
            self.voice_model = None
        gc.collect()
        empty_device_cache(self.device)

    def unload_all(self):
        """卸载所有模型"""
        self.unload_hubert()
        self.unload_f0_extractor()
        self.unload_voice_model()
        self.index = None
        self.index_vectors = None

    def load_hubert(self, model_path: str):
        """只加载指定的本地 HuBERT；损坏或缺失时不下载其他编码器替代。"""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"HuBERT 模型文件不存在：{path}")
        from fairseq import checkpoint_utils
        from fairseq.data.dictionary import Dictionary
        # The pinned HuBERT checkpoint contains this Fairseq vocabulary class.
        # Keep PyTorch's weights-only loader; allow only the required class.
        with torch.serialization.safe_globals([Dictionary]):
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(path)], suffix="")
        if len(models) != 1 or not hasattr(models[0], "final_proj"):
            raise ValueError("HuBERT 必须包含 RVC v1 所需的原生 final_proj 权重")
        self.hubert_model = models[0].to(self.device).float().eval()
        self.hubert_model_type = "fairseq"
        log.info(f"HuBERT 模型已加载: {path} ({self.device}, FP32)")

    def load_voice_model(self, model_path: str) -> dict:
        """按实际权重结构加载 v1/v2、F0/非 F0 模型，不猜测缺失的架构。"""
        from infer.lib.infer_pack.models import (
            SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid_nono,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        contract = inspect_checkpoint(checkpoint, str(model_path))
        classes = {
            ("v1", True): SynthesizerTrnMs256NSFsid,
            ("v2", True): SynthesizerTrnMs768NSFsid,
            ("v1", False): SynthesizerTrnMs256NSFsid_nono,
            ("v2", False): SynthesizerTrnMs768NSFsid_nono,
        }
        model = classes[(contract.version, contract.uses_f0)](*contract.config, is_half=False)
        del model.enc_q
        validate_state_dict(model, checkpoint["weight"])
        model.load_state_dict({k: v for k, v in checkpoint["weight"].items()
                               if not k.startswith("enc_q.")}, strict=True)
        self.voice_model = model.to(self.device).float().eval()
        self.model_version = contract.version
        self.model_feature_dim = contract.feature_dim
        self.output_sr = contract.sample_rate
        self.spk_count = contract.speaker_count
        self.uses_f0 = contract.uses_f0
        self.index = None
        self.index_vectors = None
        info = {"name": Path(model_path).stem, "sample_rate": self.output_sr,
                "version": self.model_version, "feature_dim": self.model_feature_dim,
                "speaker_count": self.spk_count, "uses_f0": self.uses_f0}
        log.info(f"语音模型已加载: {info}")
        return info

    def load_index(self, index_path: str):
        """校验成功后才提交索引状态；失败不会留下不匹配的索引。"""
        if self.model_feature_dim is None:
            raise RuntimeError("请先加载语音模型，再加载索引")
        index, vectors = read_index(index_path, self.model_feature_dim)
        self.index, self.index_vectors = index, vectors
        log.info(f"索引已加载: {index_path} ({index.d} 维, {index.ntotal} 个向量)")

    def load_f0_extractor(self, method: F0Method = "rmvpe",
                          rmvpe_path: str = None):
        """
        加载 F0 提取器

        Args:
            method: F0 提取方法
            rmvpe_path: RMVPE 模型路径
        """
        self.f0_extractor = get_f0_extractor(
            method,
            device=str(self.device),
            rmvpe_path=rmvpe_path
        )
        log.info(f"F0 提取器已加载: {method}")

    @torch.no_grad()
    def extract_features(self, audio: np.ndarray, use_final_proj: bool = False) -> torch.Tensor:
        """
        使用 HuBERT 提取特征

        Args:
            audio: 16kHz 音频数据
            use_final_proj: 是否使用 final_proj 将 768 维降到 256 维（v1 模型需要）

        Returns:
            torch.Tensor: HuBERT 特征
        """
        if self.hubert_model is None:
            raise RuntimeError("请先加载 HuBERT 模型")

        # 转换为张量
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if self.hubert_model_type == "fairseq":
            # v1 模型使用第 9 层，v2 模型使用第 12 层
            output_layer = 9 if use_final_proj else 12
            feats = self.hubert_model.extract_features(
                audio_tensor,
                padding_mask=None,
                output_layer=output_layer
            )[0]
            # v1 模型需要 256 维特征，使用 final_proj 投影
            # v2 模型需要 768 维特征，不使用 final_proj
            if use_final_proj:
                if not hasattr(self.hubert_model, 'final_proj'):
                    raise ValueError("v1 需要 HuBERT 原生 final_proj，不能截断或补零代替")
                feats = self.hubert_model.final_proj(feats)
            expected_dim = 256 if use_final_proj else 768
            if feats.shape[-1] != expected_dim:
                raise ValueError(f"HuBERT 特征维度错误：预期 {expected_dim}，实际 {feats.shape[-1]}")
            return feats

        raise ValueError(f"不支持的本地 HuBERT 后端：{self.hubert_model_type!r}")

    def search_index(self, features: np.ndarray, k: int = 8) -> np.ndarray:
        """执行有限数值、维度一致的 FAISS 检索；失败明确停止。"""
        if self.index is None or self.index_vectors is None:
            raise RuntimeError("未加载有效索引，不能执行检索")
        return retrieve_features(self.index, self.index_vectors, features, k)
    @staticmethod
    def _f0_to_coarse(
        f0: np.ndarray,
        f0_min: float = 50.0,
        f0_max: float = 1100.0
    ) -> np.ndarray:
        """Convert F0 (Hz) to official RVC coarse bins (1-255)."""
        f0 = np.asarray(f0, dtype=np.float32)
        f0_max = max(float(f0_max), float(f0_min) + 1.0)
        f0_mel_min = 1127 * np.log(1 + float(f0_min) / 700.0)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700.0)
        f0_mel = 1127 * np.log1p(np.maximum(f0, 0.0) / 700.0)
        voiced = f0_mel > 0
        f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int64)
    def _apply_rms_mix(
        self,
        audio_out: np.ndarray,
        audio_in: np.ndarray,
        sr_out: int,
        sr_in: int,
        hop_length: int,
        rms_mix_rate: float
    ) -> np.ndarray:
        """Match output RMS envelope to input RMS (0=off, 1=full match)."""
        if rms_mix_rate <= 0:
            return audio_out

        import librosa

        frame_length_in = 1024
        rms_in = librosa.feature.rms(
            y=audio_in,
            frame_length=frame_length_in,
            hop_length=hop_length,
            center=True
        )[0]

        hop_out = int(round(hop_length * sr_out / sr_in))
        frame_length_out = int(round(frame_length_in * sr_out / sr_in))
        rms_out = librosa.feature.rms(
            y=audio_out,
            frame_length=frame_length_out,
            hop_length=hop_out,
            center=True
        )[0]

        min_len = min(len(rms_in), len(rms_out))
        if min_len == 0:
            return audio_out

        rms_in = rms_in[:min_len]
        rms_out = rms_out[:min_len]

        gain = rms_in / (rms_out + 1e-6)
        gain = np.clip(gain, 0.2, 4.0)
        gain = gain ** rms_mix_rate

        gain_samples = np.repeat(gain, hop_out)
        if len(gain_samples) < len(audio_out):
            gain_samples = np.pad(
                gain_samples,
                (0, len(audio_out) - len(gain_samples)),
                mode="edge"
            )
        else:
            gain_samples = gain_samples[:len(audio_out)]

        return audio_out * gain_samples

    def _apply_silence_gate(
        self,
        audio_out: np.ndarray,
        audio_in: np.ndarray,
        f0: np.ndarray,
        sr_out: int,
        sr_in: int,
        hop_length: int,
        threshold_db: float,
        smoothing_ms: float,
        min_silence_ms: float,
        protect: float
    ) -> np.ndarray:
        """Silence gate based on input RMS and F0."""
        import librosa

        frame_length = 1024
        rms = librosa.feature.rms(
            y=audio_in,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True
        )[0]

        if len(rms) == 0 or len(f0) == 0:
            return audio_out

        # Align RMS length to F0 length
        if len(rms) < len(f0):
            rms = np.pad(rms, (0, len(f0) - len(rms)), mode="edge")
        else:
            rms = rms[:len(f0)]

        rms_db = 20 * np.log10(rms + 1e-6)
        ref_db = np.percentile(rms_db, 95)
        gate_db = ref_db + threshold_db  # threshold_db should be negative

        silent = (rms_db < gate_db) & (f0 <= 0)

        if min_silence_ms > 0:
            min_frames = int(
                round((min_silence_ms / 1000) * (sr_in / hop_length))
            )
            if min_frames > 1:
                silent_int = silent.astype(int)
                changes = np.diff(
                    np.concatenate(([0], silent_int, [0]))
                )
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                keep_silent = np.zeros_like(silent, dtype=bool)
                for s, e in zip(starts, ends):
                    if e - s >= min_frames:
                        keep_silent[s:e] = True
                silent = keep_silent

        mask = 1.0 - silent.astype(float)

        if smoothing_ms > 0:
            smooth_frames = int(
                round((smoothing_ms / 1000) * (sr_in / hop_length))
            )
            if smooth_frames > 1:
                kernel = np.ones(smooth_frames) / smooth_frames
                mask = np.convolve(
                    mask,
                    kernel,
                    mode="same"
                )
                mask = np.clip(mask, 0.0, 1.0)
        protect = float(np.clip(protect, 0.0, 1.0))
        if protect > 0:
            mask = mask * (1.0 - protect) + protect

        samples_per_frame = int(round(sr_out * hop_length / sr_in))
        mask_samples = np.repeat(mask, samples_per_frame)

        if len(mask_samples) < len(audio_out):
            mask_samples = np.pad(
                mask_samples,
                (0, len(audio_out) - len(mask_samples)),
                mode="edge"
            )
        else:
            mask_samples = mask_samples[:len(audio_out)]

        return audio_out * mask_samples

    def _process_chunk(
        self,
        features: np.ndarray,
        f0: np.ndarray,
        use_fp16: bool = False,
        speaker_id: int = 0,
    ) -> np.ndarray:
        """
        处理单个音频块

        Args:
            features: HuBERT 特征 [T, C]
            f0: F0 数组
            use_fp16: 是否使用 FP16 推理

        Returns:
            np.ndarray: 合成的音频
        """
        import torch.nn.functional as F

        log.debug(f"[_process_chunk] 输入特征: shape={features.shape}, dtype={features.dtype}")
        log.debug(f"[_process_chunk] 输入特征统计: max={np.max(np.abs(features)):.4f}, mean={np.mean(np.abs(features)):.4f}, std={np.std(features):.4f}")
        log.debug(f"[_process_chunk] 输入 F0: len={len(f0)}, max={np.max(f0):.1f}, min={np.min(f0):.1f}, non-zero={np.sum(f0 > 0)}")

        # 转换为张量
        features_tensor = torch.from_numpy(features).float().to(self.device).unsqueeze(0)
        # HuBERT 输出帧率是 50fps (hop=320 @ 16kHz)，但 RVC 模型期望 100fps
        # 需要 2x 上采样特征
        # 注意：interpolate 需要 [B, C, T] 格式，但模型需要 [B, T, C] 格式
        features_tensor = F.interpolate(features_tensor.transpose(1, 2), scale_factor=2, mode='nearest').transpose(1, 2)
        log.debug(f"[_process_chunk] 2x上采样后特征: shape={features_tensor.shape}")

        # F0 对齐到上采样后的特征长度
        # features_tensor 形状是 [B, T, C]，所以时间维度是 shape[1]
        target_len = features_tensor.shape[1]
        original_f0_len = len(f0)
        if len(f0) > target_len:
            f0 = f0[:target_len]
        elif len(f0) < target_len:
            f0 = np.pad(f0, (0, target_len - len(f0)), mode='edge')
        log.debug(f"[_process_chunk] F0 对齐: {original_f0_len} -> {len(f0)} (目标: {target_len})")

        f0_tensor = torch.from_numpy(f0.copy()).float().to(self.device).unsqueeze(0)
        # 将 F0 (Hz) 转换为 pitch 索引 (0-255)
        # RVC mel 量化映射到 coarse pitch bins
        f0_coarse = torch.from_numpy(self._f0_to_coarse(f0)).to(self.device).unsqueeze(0)
        log.debug(f"[_process_chunk] F0 张量: shape={f0_tensor.shape}, max={f0_tensor.max().item():.1f}, min={f0_tensor.min().item():.1f}")
        log.debug(f"[_process_chunk] F0 coarse (pitch索引): shape={f0_coarse.shape}, max={f0_coarse.max().item()}, min={f0_coarse.min().item()}")

        safe_speaker_id = integer(speaker_id, "speaker_id", 0, self.spk_count - 1)
        sid = torch.tensor([safe_speaker_id], device=self.device)
        log.debug(f"[_process_chunk] 说话人 ID: {sid.item()}")

        # FP16 推理
        log.debug(f"[_process_chunk] 开始推理, use_fp16={use_fp16}, device={self.device.type}")
        args = [features_tensor, torch.tensor([features_tensor.shape[1]], device=self.device)]
        if self.uses_f0:
            args.extend([f0_coarse, f0_tensor])
        args.append(sid)
        if use_fp16 and supports_fp16(self.device):
            with torch.amp.autocast(str(self.device.type), dtype=torch.float16):
                audio_out, x_mask, _ = self.voice_model.infer(*args)
        else:
            audio_out, x_mask, _ = self.voice_model.infer(*args)

        log.debug(f"[_process_chunk] 推理完成, audio_out: shape={audio_out.shape}, dtype={audio_out.dtype}")
        log.debug(f"[_process_chunk] x_mask: shape={x_mask.shape}, sum={x_mask.sum().item()}")

        # 清理
        del features_tensor, f0_tensor, f0_coarse
        empty_device_cache(self.device)

        audio_out = audio_out.squeeze().cpu().detach().float().numpy()
        log.debug(f"Chunk audio: len={len(audio_out)}, max={np.max(np.abs(audio_out)):.4f}, min={np.min(audio_out):.4f}")

        # 注意：不再对 F0=0 区域应用硬静音 mask
        # 辅音（如 k, t, s, p）通常没有基频（F0=0），硬静音会导致只剩元音
        # 如果需要降噪，应该在后处理阶段使用更智能的方法

        return audio_out

    def convert(
        self,
        audio_path: str,
        output_path: str,
        pitch_shift: float = 0,
        index_ratio: float = 0.2,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        speaker_id: int = 0,
        silence_gate: bool = True,
        silence_threshold_db: float = -45.0,
        silence_smoothing_ms: float = 50.0,
        silence_min_duration_ms: float = 200.0
    ) -> str:
        """
        执行 RVC 推理

        Args:
            audio_path: 输入音频路径
            output_path: 输出音频路径
            pitch_shift: 音调偏移 (半音)
            index_ratio: 索引混合比率 (0-1)
            filter_radius: 中值滤波半径
            resample_sr: 重采样率 (0 表示不重采样)
            rms_mix_rate: RMS 混合比率
            protect: 保护清辅音
            speaker_id: 说话人 ID（多说话人模型可调）
            silence_gate: 启用静音门限（默认开启以消除静音段底噪）
            silence_threshold_db: 静音阈值 (dB, 相对峰值)
            silence_smoothing_ms: 门限平滑时长 (ms)
            silence_min_duration_ms: 最短静音时长 (ms)

        Returns:
            str: 输出文件路径
        """
        # 检查模型
        if self.voice_model is None:
            raise RuntimeError("请先加载语音模型")
        if self.hubert_model is None:
            raise RuntimeError("请先加载 HuBERT 模型")
        if self.uses_f0 and self.f0_extractor is None:
            raise RuntimeError("请先加载 F0 提取器")
        pitch_shift = integer(pitch_shift, "pitch_shift", -24, 24)
        index_ratio = number(index_ratio, "index_ratio", 0, 1)
        filter_radius = integer(filter_radius, "filter_radius", 0, 15)
        rms_mix_rate = number(rms_mix_rate, "rms_mix_rate", 0, 1)
        protect = number(protect, "protect", 0, 0.5)
        speaker_id = integer(speaker_id, "speaker_id", 0, self.spk_count - 1)
        resample_sr = integer(resample_sr, "resample_sr", 0, 192000)
        if resample_sr and resample_sr < 16000:
            raise ValueError("resample_sr 必须为 0 或 16000～192000 Hz")
        if type(silence_gate) is not bool:
            raise ValueError("silence_gate 必须是布尔值")
        number(silence_threshold_db, "silence_threshold_db", -120, 0)
        number(silence_smoothing_ms, "silence_smoothing_ms", 0, 10000)
        number(silence_min_duration_ms, "silence_min_duration_ms", 0, 60000)
        if not self.uses_f0 and pitch_shift != 0:
            raise ValueError("非 F0 模型不支持音高偏移，请使用 F0 模型或设为 0")
        if index_ratio > 0 and self.index is None:
            raise ValueError("index_ratio > 0 时必须提供有效索引；不使用索引请显式设为 0")

        # 加载音频
        audio = load_audio(audio_path, sr=self.sample_rate)
        audio = normalize_audio(audio)

        # 高通滤波去除低频隆隆声（与官方管道一致）
        audio = sp_signal.filtfilt(_bh, _ah, audio).astype(np.float32)

        # 步骤1: 提取 F0 (使用 RMVPE 或 Hybrid)
        f0 = self.f0_extractor.extract(audio) if self.uses_f0 else np.zeros(max(1, len(audio) // 160), dtype=np.float32)

        # 音调偏移
        if pitch_shift != 0:
            f0 = shift_f0(f0, pitch_shift)

        # 智能中值滤波 - 仅在F0跳变过大时应用，保留自然颤音
        if filter_radius > 0:
            from scipy.ndimage import median_filter

            # 计算F0跳变（半音）
            f0_semitone_diff = np.abs(12 * np.log2((f0 + 1e-6) / (np.roll(f0, 1) + 1e-6)))
            f0_semitone_diff[0] = 0

            # 只对跳变超过2个半音的区域应用滤波
            need_filter = f0_semitone_diff > 2.0

            # 扩展需要滤波的区域（前后各1帧）
            kernel = np.ones(3, dtype=bool)
            need_filter = np.convolve(need_filter, kernel, mode='same')

            # 应用滤波
            f0_filtered = median_filter(f0, size=filter_radius)

            # 高音区域 (>500Hz) 使用更温和的滤波，避免高音被过度平滑
            # 参考: RMVPE论文建议高频区域使用自适应平滑
            high_pitch_mask = f0 > 500

            # 对高音区域使用更小的滤波半径
            if np.any(high_pitch_mask):
                f0_filtered_high = median_filter(f0, size=max(1, filter_radius // 2))
                f0_filtered = np.where(high_pitch_mask, f0_filtered_high, f0_filtered)

            # 混合：只在需要的地方滤波，其他保留原始
            f0 = np.where(need_filter, f0_filtered, f0)

        # 释放 F0 提取器显存
        self.unload_f0_extractor()

        # 步骤2: 提取 HuBERT 特征
        # v1 模型需要 256 维特征（使用 final_proj），v2 模型需要 768 维
        use_final_proj = (self.model_version == "v1")
        features = self.extract_features(audio, use_final_proj=use_final_proj)
        features = features.squeeze(0).cpu().numpy()

        # 释放 HuBERT 显存
        self.unload_hubert()

        # 索引检索 (CPU 操作)
        if self.index is not None and index_ratio > 0:
            features_before_index = features.copy()
            retrieved = self.search_index(features)

            # 简单的自适应索引混合（不使用白化和残差去除）
            # 高音区域使用稍高的索引率
            adaptive_index_ratio = np.ones(len(features)) * index_ratio

            f0_per_feat = 2
            for fi in range(len(features)):
                f0_start = fi * f0_per_feat
                f0_end = min(f0_start + f0_per_feat, len(f0))
                if f0_end > f0_start:
                    f0_segment = f0[f0_start:f0_end]
                    avg_f0 = np.mean(f0_segment[f0_segment > 0]) if np.any(f0_segment > 0) else 0
                    # 高音区域提升索引率
                    if avg_f0 > 450:
                        adaptive_index_ratio[fi] = min(0.75, index_ratio * 1.3)

            adaptive_index_ratio = adaptive_index_ratio[:, np.newaxis]
            features = features * (1 - adaptive_index_ratio) + retrieved * adaptive_index_ratio

            # 动态辅音保护：基于F0置信度和能量调整protect强度
            # 避免索引检索破坏辅音清晰度，与官方管道行为一致
            if self.uses_f0 and protect < 0.5:
                # 构建逐帧保护掩码：F0>0 的帧用 1.0（完全使用索引混合后特征），
                # F0=0 的帧用 protect 值（大部分保留原始特征）
                # F0 帧率是特征帧率的 2 倍 (hop 160 vs 320)，需要下采样对齐
                f0_per_feat = 2  # 每个特征帧对应 2 个 F0 帧
                n_feat = features.shape[0]
                protect_mask = np.ones(n_feat, dtype=np.float32)

                # 计算每个特征帧的F0稳定性和能量
                for fi in range(n_feat):
                    f0_start = fi * f0_per_feat
                    f0_end = min(f0_start + f0_per_feat, len(f0))
                    if f0_end > f0_start:
                        f0_segment = f0[f0_start:f0_end]
                        # 无声段（F0=0）：强保护，保留更多原始特征
                        # 参考: "Voice Conversion for Articulation Disorders" 建议保护辅音
                        if np.all(f0_segment <= 0):
                            # 提高无声段保护强度，从 protect 提升到 protect * 1.5
                            protect_mask[fi] = min(0.8, protect * 1.5)
                        # F0不稳定段（方差大）：中等保护
                        elif len(f0_segment) > 1 and np.std(f0_segment) > 50:
                            protect_mask[fi] = protect + (1.0 - protect) * 0.3
                        # 低能量段（可能是呼吸音）：增强保护
                        # 使用特征的L2范数作为能量指标
                        feat_energy = np.linalg.norm(features_before_index[fi])
                        if feat_energy < 0.5:  # 低能量阈值
                            protect_mask[fi] = min(0.8, protect * 1.3)

                # 平滑保护掩码，避免突变
                smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
                smooth_kernel /= np.sum(smooth_kernel)
                protect_mask = np.convolve(protect_mask, smooth_kernel, mode="same")
                protect_mask = np.convolve(protect_mask, smooth_kernel, mode="same")
                protect_mask = np.clip(protect_mask, protect, 1.0)
                protect_mask = protect_mask[:, np.newaxis]  # [T, 1] 广播到 [T, C]
                features = features * protect_mask + features_before_index * (1 - protect_mask)

        # 静音处理仅由下方 silence_gate 控制；不叠加隐藏的特征/F0 门控。

        # 步骤3: 语音合成 (voice_model 推理) - 分块处理
        # 分块参数 - 增加重叠以减少边界伪影
        CHUNK_SECONDS = 30  # 每块 30 秒
        OVERLAP_SECONDS = 2.0  # 重叠 2.0 秒（从1.0增加到2.0，减少破音）
        HOP_LENGTH = 320  # HuBERT hop length

        # 计算分块大小（以特征帧为单位）
        chunk_frames = int(CHUNK_SECONDS * self.sample_rate / HOP_LENGTH)
        overlap_frames = int(OVERLAP_SECONDS * self.sample_rate / HOP_LENGTH)

        total_frames = features.shape[0]

        # 如果音频短于一块，直接处理
        if total_frames <= chunk_frames:
            audio_out = self._process_chunk(features, f0, speaker_id=speaker_id)
        else:
            # 分块处理
            log.info(f"音频较长 ({total_frames} 帧)，启用分块处理...")
            audio_chunks = []
            chunk_idx = 0

            for start in range(0, total_frames, chunk_frames - overlap_frames):
                end = min(start + chunk_frames, total_frames)
                chunk_features = features[start:end]

                # 计算对应的 F0 范围
                # F0 帧率是特征帧率的 2 倍 (hop 160 vs 320)
                f0_start = start * 2
                f0_end = min(end * 2, len(f0))
                chunk_f0 = f0[f0_start:f0_end]

                log.debug(f"处理块 {chunk_idx}: 帧 {start}-{end}")

                # 处理当前块
                chunk_audio = self._process_chunk(chunk_features, chunk_f0, speaker_id=speaker_id)
                audio_chunks.append(chunk_audio)
                chunk_idx += 1

                # 清理显存
                gc.collect()
                empty_device_cache(self.device)

            # 交叉淡入淡出拼接
            audio_out = self._crossfade_chunks(audio_chunks, overlap_frames)
            log.info(f"分块处理完成，共 {chunk_idx} 块")

        # 后处理
        if isinstance(audio_out, tuple):
            audio_out = audio_out[0]
        audio_out = np.asarray(audio_out).flatten()

        # 重采样
        if resample_sr > 0 and resample_sr != self.output_sr:
            import librosa
            audio_out = librosa.resample(
                audio_out,
                orig_sr=self.output_sr,
                target_sr=resample_sr
            )
            save_sr = resample_sr
        else:
            save_sr = self.output_sr

        # 可选 RMS 包络混合
        if rms_mix_rate > 0:
            audio_out = self._apply_rms_mix(
                audio_out=audio_out,
                audio_in=audio,
                sr_out=save_sr,
                sr_in=self.sample_rate,
                hop_length=160,
                rms_mix_rate=rms_mix_rate
            )

        # 可选静音门限 (减少无声段气声/噪声)
        if silence_gate:
            audio_out = self._apply_silence_gate(
                audio_out=audio_out,
                audio_in=audio,
                f0=f0,
                sr_out=save_sr,
                sr_in=self.sample_rate,
                hop_length=160,
                threshold_db=silence_threshold_db,
                smoothing_ms=silence_smoothing_ms,
                min_silence_ms=silence_min_duration_ms,
                protect=protect
            )

        # 本地路线保留明确的相位/底噪处理；失败向上抛出，不跳过。
        from lib.vocoder_fix import apply_vocoder_artifact_fix
        audio_out = apply_vocoder_artifact_fix(
            audio_out, sr=save_sr, f0=f0 if self.uses_f0 else None,
            fix_phase=True, fix_breath=True, fix_sustained=False,
        )
        log.detail("已应用vocoder伪影修复（相位+底噪清理）")

        # 峰值限幅（不改变整体响度，后续由 cover_pipeline 控制音量）
        audio_out = soft_clip(audio_out, threshold=0.9, ceiling=0.99)

        # 保存
        save_audio(output_path, audio_out, sr=save_sr)

        return output_path

    def _crossfade_chunks(self, chunks: list, overlap_frames: int) -> np.ndarray:
        """
        使用 SOLA (Synchronized Overlap-Add) 拼接音频块

        SOLA 通过在重叠区域搜索最佳相位对齐点来避免分块边界的撕裂伪影。
        参考: w-okada/voice-changer Issue #163, DDSP-SVC 实现

        Args:
            chunks: 音频块列表
            overlap_frames: 重叠帧数（特征帧）

        Returns:
            np.ndarray: 拼接后的音频
        """
        if len(chunks) == 1:
            return chunks[0]

        # 正确计算重叠的音频样本数
        # 1 特征帧 = HOP_LENGTH 输入样本 @ 16kHz
        # 输出样本数 = HOP_LENGTH * (output_sr / input_sr)
        HOP_LENGTH = 320
        INPUT_SR = 16000
        output_sr = getattr(self, 'output_sr', 40000)

        # 每个特征帧对应的输出样本数
        samples_per_frame = int(HOP_LENGTH * output_sr / INPUT_SR)
        overlap_samples = overlap_frames * samples_per_frame

        log.debug(f"SOLA Crossfade: overlap_frames={overlap_frames}, samples_per_frame={samples_per_frame}, overlap_samples={overlap_samples}")

        result = chunks[0]

        for i in range(1, len(chunks)):
            chunk = chunks[i]

            # 确保重叠区域不超过任一块的长度
            actual_overlap = min(overlap_samples, len(result), len(chunk))

            if actual_overlap > 0:
                # SOLA: 在重叠区域搜索最佳相位对齐点
                # 搜索范围：不超过一个基频周期（约 100-200 样本 @ 48kHz）
                search_range = min(int(output_sr * 0.005), actual_overlap // 4)  # 5ms 或 1/4 重叠

                # 提取前一块的尾部作为参考
                reference = result[-actual_overlap:]

                # 在新块的开头搜索最佳对齐位置
                best_offset = 0
                max_correlation = -1.0

                for offset in range(max(0, -search_range), min(search_range + 1, len(chunk) - actual_overlap + 1)):
                    # 提取候选区域
                    candidate_start = max(0, offset)
                    candidate_end = candidate_start + actual_overlap

                    if candidate_end > len(chunk):
                        continue

                    candidate = chunk[candidate_start:candidate_end]

                    # 计算归一化互相关
                    ref_norm = np.linalg.norm(reference)
                    cand_norm = np.linalg.norm(candidate)

                    if ref_norm > 1e-6 and cand_norm > 1e-6:
                        correlation = np.dot(reference, candidate) / (ref_norm * cand_norm)

                        if correlation > max_correlation:
                            max_correlation = correlation
                            best_offset = offset

                log.debug(f"SOLA chunk {i}: best_offset={best_offset}, correlation={max_correlation:.4f}")

                # 如果相关性太低（<0.3），说明信号不连续，使用简单crossfade避免伪影
                if max_correlation < 0.3:
                    log.debug(f"SOLA chunk {i}: low correlation, using simple crossfade")
                    fade_out = np.linspace(1, 0, actual_overlap)
                    fade_in = np.linspace(0, 1, actual_overlap)
                    result_end = result[-actual_overlap:] * fade_out
                    chunk_start = chunk[:actual_overlap] * fade_in
                    result = np.concatenate([
                        result[:-actual_overlap],
                        result_end + chunk_start,
                        chunk[actual_overlap:]
                    ])
                    continue

                # 在最佳对齐点应用交叉淡入淡出
                aligned_start = max(0, best_offset)
                aligned_end = aligned_start + actual_overlap

                if aligned_end <= len(chunk):
                    # 创建淡入淡出曲线（使用余弦窗以获得更平滑的过渡）
                    fade_out = np.cos(np.linspace(0, np.pi / 2, actual_overlap)) ** 2
                    fade_in = np.sin(np.linspace(0, np.pi / 2, actual_overlap)) ** 2

                    # 应用交叉淡入淡出
                    result_end = result[-actual_overlap:] * fade_out
                    chunk_aligned = chunk[aligned_start:aligned_end] * fade_in

                    # 拼接
                    result = np.concatenate([
                        result[:-actual_overlap],
                        result_end + chunk_aligned,
                        chunk[aligned_end:]
                    ])
                else:
                    # 对齐失败，回退到简单拼接
                    log.warning(f"SOLA alignment failed for chunk {i}, using simple crossfade")
                    fade_out = np.linspace(1, 0, actual_overlap)
                    fade_in = np.linspace(0, 1, actual_overlap)
                    result_end = result[-actual_overlap:] * fade_out
                    chunk_start = chunk[:actual_overlap] * fade_in
                    result = np.concatenate([
                        result[:-actual_overlap],
                        result_end + chunk_start,
                        chunk[actual_overlap:]
                    ])
            else:
                # 无重叠，直接拼接
                result = np.concatenate([result, chunk])

        return result


def list_voice_models(weights_dir: str = "assets/weights") -> list:
    """
    列出可用的语音模型

    Args:
        weights_dir: 模型目录

    Returns:
        list: 模型信息列表
    """
    models = []
    weights_path = Path(weights_dir)

    if not weights_path.exists():
        return models

    # 递归搜索所有子目录
    for pth_file in weights_path.glob("**/*.pth"):
        from infer.official_adapter import _resolve_index_path
        # A character directory is an explicit asset bundle; archives often use
        # different weight/index basenames. Generic weight dirs require a name match.
        relative = pth_file.relative_to(weights_path)
        index_error = None
        try:
            if len(relative.parts) >= 3 and relative.parts[0] == 'characters':
                from tools.character_assets import model_files
                _, index_file = model_files(weights_path / 'characters' / relative.parts[1])
            else:
                index_file = _resolve_index_path(pth_file, None)
        except (ValueError, OSError) as exc:
            # Keep the invalid entry visible; both the UI table and MCP expose
            # this error. Conversion with positive retrieval still fails closed.
            index_file, index_error = None, str(exc)

        models.append({
            "name": pth_file.stem,
            "model_path": str(pth_file),
            "index_path": str(index_file) if index_file else None,
            "index_error": index_error,
        })

    return models


