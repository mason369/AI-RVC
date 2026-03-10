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
        self.hubert_layer = 12
        self.voice_model = None
        self.index = None
        self.f0_extractor = None
        self.spk_count = 1
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

    def load_hubert(self, model_path: str):
        """
        加载 HuBERT 模型

        Args:
            model_path: HuBERT 模型路径（可以是本地 .pt 文件或 Hugging Face 模型名）
        """
        # 优先使用 fairseq 兼容实现（官方实现）
        if os.path.isfile(model_path):
            try:
                from fairseq import checkpoint_utils

                models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                    [model_path],
                    suffix=""
                )
                model = models[0]
                model = model.to(self.device).eval()
                self.hubert_model = model
                self.hubert_model_type = "fairseq"
                log.info(f"HuBERT 模型已加载: {model_path} ({self.device})")
                return
            except Exception as e:
                log.warning(f"fairseq 加载失败，尝试 torchaudio: {e}")

        try:
            import torchaudio

            bundle = torchaudio.pipelines.HUBERT_BASE
            model = bundle.get_model()
            model = model.to(self.device).eval()
            self.hubert_model = model
            self.hubert_model_type = "torchaudio"
            log.info(
                f"HuBERT 模型已加载: torchaudio HUBERT_BASE ({self.device})"
            )
            return
        except Exception as e:
            log.warning(f"torchaudio 加载失败，尝试 transformers: {e}")

        from transformers import HubertModel

        if os.path.isfile(model_path):
            log.info("检测到本地模型文件，将使用 Hugging Face 预训练模型替代")
            model_name = "facebook/hubert-base-ls960"
        else:
            model_name = model_path

        try:
            self.hubert_model = HubertModel.from_pretrained(model_name)
        except Exception as e:
            log.warning(f"从网络加载失败，尝试使用本地缓存: {e}")
            self.hubert_model = HubertModel.from_pretrained(
                model_name,
                local_files_only=True
            )
        self.hubert_model = self.hubert_model.to(self.device).eval()
        self.hubert_model_type = "transformers"
        log.info(f"HuBERT 模型已加载: {model_name} ({self.device})")

    def load_voice_model(self, model_path: str) -> dict:
        """
        加载语音模型

        Args:
            model_path: 模型文件路径 (.pth)

        Returns:
            dict: 模型信息
        """
        log.debug(f"正在加载语音模型: {model_path}")
        cpt = torch.load(model_path, map_location="cpu", weights_only=False)

        log.debug(f"模型文件 keys: {cpt.keys()}")

        # 提取模型配置
        config = cpt.get("config", [])
        self.output_sr = cpt.get("sr", 48000)

        log.debug(f"config 类型: {type(config)}, 内容: {config}")
        log.debug(f"采样率: {self.output_sr}")

        # 处理 list 格式的 config（RVC v2 标准格式）
        if isinstance(config, list) and len(config) >= 18:
            model_config = {
                "spec_channels": config[0],
                "segment_size": config[1],
                "inter_channels": config[2],
                "hidden_channels": config[3],
                "filter_channels": config[4],
                "n_heads": config[5],
                "n_layers": config[6],
                "kernel_size": config[7],
                "p_dropout": config[8],
                "resblock": config[9],
                "resblock_kernel_sizes": config[10],
                "resblock_dilation_sizes": config[11],
                "upsample_rates": config[12],
                "upsample_initial_channel": config[13],
                "upsample_kernel_sizes": config[14],
                "spk_embed_dim": config[15],
                "gin_channels": config[16],
            }
            # 使用 config 中的采样率（如果有）
            if len(config) > 17:
                self.output_sr = config[17]
        elif isinstance(config, dict):
            # 兼容 dict 格式
            model_config = config
        else:
            # 使用默认值
            log.warning("无法解析 config，使用默认值")
            model_config = {}

        log.debug(f"解析后的配置: {model_config}")

        # 根据hidden_channels选择正确的合成器
        hidden_channels = model_config.get("hidden_channels", 192)
        if hidden_channels == 256 or hidden_channels >= 512:
            # v2模型：768维
            from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
            synthesizer_class = SynthesizerTrnMs768NSFsid
            self.model_version = "v2"
            log.debug(f"使用v2合成器 (768维): hidden_channels={hidden_channels}")
        else:
            # v1模型：256维
            from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid
            synthesizer_class = SynthesizerTrnMs256NSFsid
            self.model_version = "v1"
            log.debug(f"使用v1合成器 (256维): hidden_channels={hidden_channels}")

        # 加载模型权重
        self.voice_model = synthesizer_class(
            spec_channels=model_config.get("spec_channels", 1025),
            segment_size=model_config.get("segment_size", 32),
            inter_channels=model_config.get("inter_channels", 192),
            hidden_channels=model_config.get("hidden_channels", 192),
            filter_channels=model_config.get("filter_channels", 768),
            n_heads=model_config.get("n_heads", 2),
            n_layers=model_config.get("n_layers", 6),
            kernel_size=model_config.get("kernel_size", 3),
            p_dropout=model_config.get("p_dropout", 0),
            resblock=model_config.get("resblock", "1"),
            resblock_kernel_sizes=model_config.get("resblock_kernel_sizes", [3, 7, 11]),
            resblock_dilation_sizes=model_config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            upsample_rates=model_config.get("upsample_rates", [10, 10, 2, 2]),
            upsample_initial_channel=model_config.get("upsample_initial_channel", 512),
            upsample_kernel_sizes=model_config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            spk_embed_dim=model_config.get("spk_embed_dim", 109),
            gin_channels=model_config.get("gin_channels", 256),
            sr=self.output_sr,
            is_half=supports_fp16(self.device)  # 根据设备能力决定是否使用半精度
        )
        self.spk_count = int(model_config.get("spk_embed_dim", 1) or 1)

        # 加载权重
        self.voice_model.load_state_dict(cpt["weight"], strict=False)
        self.voice_model = self.voice_model.to(self.device).eval()

        model_info = {
            "name": Path(model_path).stem,
            "sample_rate": self.output_sr,
            "version": cpt.get("version", "v2")
        }

        log.info(f"语音模型已加载: {model_info['name']} ({self.output_sr}Hz)")
        return model_info

    def load_index(self, index_path: str):
        """
        加载 FAISS 索引

        Args:
            index_path: 索引文件路径 (.index)
        """
        self.index = faiss.read_index(index_path)
        # 启用 direct_map 以支持 reconstruct()
        try:
            self.index.make_direct_map()
        except Exception:
            pass  # 某些索引类型不支持，忽略
        log.info(f"索引已加载: {index_path}")

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
            if use_final_proj and hasattr(self.hubert_model, 'final_proj'):
                feats = self.hubert_model.final_proj(feats)
            return feats

        if self.hubert_model_type == "torchaudio":
            feats_list, _ = self.hubert_model.extract_features(audio_tensor)
            layer_idx = min(self.hubert_layer - 1, len(feats_list) - 1)
            return feats_list[layer_idx]

        # transformers fallback
        outputs = self.hubert_model(audio_tensor, output_hidden_states=True)
        layer_idx = min(self.hubert_layer, len(outputs.hidden_states) - 1)
        return outputs.hidden_states[layer_idx]

    def search_index(self, features: np.ndarray, k: int = 8) -> np.ndarray:
        """
        在索引中搜索相似特征

        Args:
            features: 输入特征
            k: 返回的近邻数量

        Returns:
            np.ndarray: 检索到的特征
        """
        if self.index is None:
            return features

        # 检查特征维度是否与索引匹配
        if features.shape[-1] != self.index.d:
            log.warning(f"特征维度 ({features.shape[-1]}) 与索引维度 ({self.index.d}) 不匹配，跳过索引搜索")
            return features

        # 搜索（使用距离倒数平方加权，与官方管道一致）
        scores, indices = self.index.search(features, k)

        # 尝试重建特征，如果索引不支持则跳过
        try:
            big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        except RuntimeError as e:
            if "direct map" in str(e):
                log.warning("索引不支持向量重建，跳过索引混合")
                return features
            raise

        # 距离倒数平方加权
        weight = np.square(1.0 / (scores + 1e-6))
        weight /= weight.sum(axis=1, keepdims=True)
        retrieved = np.sum(
            big_npy[indices] * np.expand_dims(weight, axis=2), axis=1
        )
        return retrieved
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

        safe_speaker_id = int(max(0, min(max(1, int(self.spk_count)) - 1, int(speaker_id))))
        sid = torch.tensor([safe_speaker_id], device=self.device)
        log.debug(f"[_process_chunk] 说话人 ID: {sid.item()}")

        # FP16 推理
        log.debug(f"[_process_chunk] 开始推理, use_fp16={use_fp16}, device={self.device.type}")
        if use_fp16 and supports_fp16(self.device):
            with torch.amp.autocast(str(self.device.type)):
                audio_out, x_mask, _ = self.voice_model.infer(
                    features_tensor,
                    torch.tensor([features_tensor.shape[1]], device=self.device),
                    f0_coarse,
                    f0_tensor,
                    sid
                )
        else:
            audio_out, x_mask, _ = self.voice_model.infer(
                features_tensor,
                torch.tensor([features_tensor.shape[1]], device=self.device),
                f0_coarse,
                f0_tensor,
                sid
            )

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
        silence_gate: bool = False,
        silence_threshold_db: float = -40.0,
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
            silence_gate: 启用静音门限
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
        if self.f0_extractor is None:
            raise RuntimeError("请先加载 F0 提取器")

        # 加载音频
        audio = load_audio(audio_path, sr=self.sample_rate)
        audio = normalize_audio(audio)
        rms_mix_rate = float(np.clip(rms_mix_rate, 0.0, 1.0))
        speaker_id = int(max(0, min(max(1, int(self.spk_count)) - 1, int(speaker_id))))

        # 高通滤波去除低频隆隆声（与官方管道一致）
        audio = sp_signal.filtfilt(_bh, _ah, audio).astype(np.float32)

        # 步骤1: 提取 F0 (使用 RMVPE 或 Hybrid)
        f0 = self.f0_extractor.extract(audio)

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
            if protect < 0.5:
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

        # --- 能量感知硬门控（索引+protect 之后、分块推理之前）---
        import librosa as _librosa_local
        _hop_feat = 320  # HuBERT hop
        _n_feat = features.shape[0]
        _frame_rms = _librosa_local.feature.rms(
            y=audio, frame_length=_hop_feat * 2, hop_length=_hop_feat, center=True
        )[0]
        if _frame_rms.ndim > 1:
            _frame_rms = _frame_rms[0]
        if len(_frame_rms) > _n_feat:
            _frame_rms = _frame_rms[:_n_feat]
        elif len(_frame_rms) < _n_feat:
            _frame_rms = np.pad(_frame_rms, (0, _n_feat - len(_frame_rms)), mode='edge')
        _energy_db = 20.0 * np.log10(_frame_rms + 1e-8)
        _ref_db = float(np.percentile(_energy_db, 95)) if _frame_rms.size > 0 else -20.0
        # 硬门控：只有低于 ref-45dB 的真正静默帧被清零
        _silence_threshold = _ref_db - 45.0
        _energy_gate = (_energy_db > _silence_threshold).astype(np.float32)
        _sm = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        _sm /= _sm.sum()
        _energy_gate = np.convolve(_energy_gate, _sm, mode='same')[:_n_feat]
        _energy_gate = (_energy_gate > 0.5).astype(np.float32)  # 重新二值化

        # 特征硬门控（50fps）
        features = features * _energy_gate[:, np.newaxis]

        # F0 硬清零（100fps = 特征帧率 × 2）
        _f0_gate = np.repeat(_energy_gate, 2)
        if len(_f0_gate) > len(f0):
            _f0_gate = _f0_gate[:len(f0)]
        elif len(_f0_gate) < len(f0):
            _f0_gate = np.pad(_f0_gate, (0, len(f0) - len(_f0_gate)), mode='constant', constant_values=1.0)
        f0 = f0 * _f0_gate

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

        # 应用人声清理后处理（减少齿音和呼吸音）
        # 参考: "Managing Sibilance" 和 "How to REALLY Clean Vocals"
        try:
            from lib.vocal_cleanup import apply_vocal_cleanup
            audio_out = apply_vocal_cleanup(
                audio_out,
                sr=save_sr,
                reduce_sibilance_enabled=True,
                reduce_breath_enabled=True,
                sibilance_reduction_db=3.0,  # 降低衰减量，避免过度处理
                breath_reduction_db=6.0      # 降低衰减量
            )
            log.detail("已应用人声清理后处理")
        except Exception as e:
            log.warning(f"人声清理后处理失败: {e}")

        # 峰值限幅（不改变整体响度，后续由 cover_pipeline 控制音量）
        audio_out = soft_clip(audio_out, threshold=0.9, ceiling=0.99)

        # 保存
        save_audio(output_path, audio_out, sr=save_sr)

        return output_path

    def _crossfade_chunks(self, chunks: list, overlap_frames: int) -> np.ndarray:
        """
        使用交叉淡入淡出拼接音频块

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
        samples_per_frame = int(HOP_LENGTH * output_sr / INPUT_SR)  # 16kHz->40kHz: 800, 16kHz->48kHz: 960
        overlap_samples = overlap_frames * samples_per_frame

        log.debug(f"Crossfade: overlap_frames={overlap_frames}, samples_per_frame={samples_per_frame}, overlap_samples={overlap_samples}")

        result = chunks[0]

        for i in range(1, len(chunks)):
            chunk = chunks[i]

            # 确保重叠区域不超过任一块的长度
            actual_overlap = min(overlap_samples, len(result), len(chunk))

            if actual_overlap > 0:
                # 创建淡入淡出曲线
                fade_out = np.linspace(1, 0, actual_overlap)
                fade_in = np.linspace(0, 1, actual_overlap)

                # 应用交叉淡入淡出
                result_end = result[-actual_overlap:] * fade_out
                chunk_start = chunk[:actual_overlap] * fade_in

                # 拼接
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
        # 查找对应的索引文件（同目录下）
        index_file = pth_file.with_suffix(".index")
        if not index_file.exists():
            # 尝试其他命名方式
            index_file = pth_file.parent / f"{pth_file.stem}_v2.index"
        if not index_file.exists():
            # 尝试不区分大小写匹配
            for f in pth_file.parent.glob("*.index"):
                if f.stem.lower() == pth_file.stem.lower():
                    index_file = f
                    break

        models.append({
            "name": pth_file.stem,
            "model_path": str(pth_file),
            "index_path": str(index_file) if index_file.exists() else None
        })

    return models


