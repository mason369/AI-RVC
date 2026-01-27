# -*- coding: utf-8 -*-
"""
RVC 推理管道 - 端到端语音转换
"""
import os
import torch
import numpy as np
import faiss
from pathlib import Path
from typing import Optional, Tuple, Union

from lib.audio import load_audio, save_audio, normalize_audio
from lib.device import get_device
from infer.f0_extractor import get_f0_extractor, shift_f0, F0Method


class VoiceConversionPipeline:
    """RVC 语音转换管道"""

    def __init__(self, device: str = "cuda"):
        """
        初始化管道

        Args:
            device: 计算设备 ("cuda" 或 "cpu")
        """
        self.device = get_device(device)
        self.hubert_model = None
        self.voice_model = None
        self.index = None
        self.f0_extractor = None

        # 默认参数
        self.sample_rate = 16000  # HuBERT 输入采样率
        self.output_sr = 48000    # 输出采样率

    def load_hubert(self, model_path: str):
        """
        加载 HuBERT 模型

        Args:
            model_path: HuBERT 模型路径
        """
        from fairseq import checkpoint_utils

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            suffix=""
        )
        self.hubert_model = models[0].to(self.device).eval()
        print(f"HuBERT 模型已加载: {self.device}")

    def load_voice_model(self, model_path: str) -> dict:
        """
        加载语音模型

        Args:
            model_path: 模型文件路径 (.pth)

        Returns:
            dict: 模型信息
        """
        cpt = torch.load(model_path, map_location="cpu")

        # 提取模型配置
        config = cpt.get("config", {})
        self.output_sr = cpt.get("sr", 48000)

        # 加载模型权重
        from models.synthesizer import SynthesizerTrnMs768NSFsid

        self.voice_model = SynthesizerTrnMs768NSFsid(
            spec_channels=config.get("spec_channels", 1025),
            segment_size=config.get("segment_size", 32),
            inter_channels=config.get("inter_channels", 192),
            hidden_channels=config.get("hidden_channels", 192),
            filter_channels=config.get("filter_channels", 768),
            n_heads=config.get("n_heads", 2),
            n_layers=config.get("n_layers", 6),
            kernel_size=config.get("kernel_size", 3),
            p_dropout=config.get("p_dropout", 0),
            resblock_kernel_sizes=config.get("resblock_kernel_sizes", [3, 7, 11]),
            resblock_dilation_sizes=config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            upsample_rates=config.get("upsample_rates", [10, 10, 2, 2]),
            upsample_initial_channel=config.get("upsample_initial_channel", 512),
            upsample_kernel_sizes=config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            spk_embed_dim=config.get("spk_embed_dim", 109),
            gin_channels=config.get("gin_channels", 256),
            sr=self.output_sr
        )

        # 加载权重
        self.voice_model.load_state_dict(cpt["weight"], strict=False)
        self.voice_model = self.voice_model.to(self.device).eval()

        model_info = {
            "name": Path(model_path).stem,
            "sample_rate": self.output_sr,
            "version": cpt.get("version", "v2")
        }

        print(f"语音模型已加载: {model_info['name']} ({self.output_sr}Hz)")
        return model_info

    def load_index(self, index_path: str):
        """
        加载 FAISS 索引

        Args:
            index_path: 索引文件路径 (.index)
        """
        self.index = faiss.read_index(index_path)
        print(f"索引已加载: {index_path}")

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
        print(f"F0 提取器已加载: {method}")

    @torch.no_grad()
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """
        使用 HuBERT 提取特征

        Args:
            audio: 16kHz 音频数据

        Returns:
            torch.Tensor: HuBERT 特征
        """
        if self.hubert_model is None:
            raise RuntimeError("请先加载 HuBERT 模型")

        # 转换为张量
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 填充
        padding_mask = torch.zeros_like(audio_tensor, dtype=torch.bool)

        # 提取特征
        inputs = {
            "source": audio_tensor,
            "padding_mask": padding_mask,
            "output_layer": 12
        }

        logits = self.hubert_model.extract_features(**inputs)
        feats = logits[0]

        return feats

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

        # 搜索
        _, indices = self.index.search(features, k)

        # 获取特征
        retrieved = np.zeros_like(features)
        for i, idx in enumerate(indices):
            retrieved[i] = np.mean(
                [self.index.reconstruct(int(j)) for j in idx],
                axis=0
            )

        return retrieved

    def convert(
        self,
        audio_path: str,
        output_path: str,
        pitch_shift: float = 0,
        index_ratio: float = 0.5,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33
    ) -> str:
        """
        执行语音转换

        Args:
            audio_path: 输入音频路径
            output_path: 输出音频路径
            pitch_shift: 音调偏移 (半音)
            index_ratio: 索引混合比率 (0-1)
            filter_radius: 中值滤波半径
            resample_sr: 重采样率 (0 表示不重采样)
            rms_mix_rate: RMS 混合比率
            protect: 保护清辅音

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

        # 提取 F0
        f0 = self.f0_extractor.extract(audio)

        # 音调偏移
        if pitch_shift != 0:
            f0 = shift_f0(f0, pitch_shift)

        # 中值滤波
        if filter_radius > 0:
            from scipy.ndimage import median_filter
            f0 = median_filter(f0, size=filter_radius)

        # 提取 HuBERT 特征
        features = self.extract_features(audio)
        features = features.squeeze(0).cpu().numpy()

        # 索引检索
        if self.index is not None and index_ratio > 0:
            retrieved = self.search_index(features)
            features = features * (1 - index_ratio) + retrieved * index_ratio

        # 转换为张量
        features = torch.from_numpy(features).float().to(self.device).unsqueeze(0)
        f0_tensor = torch.from_numpy(f0).float().to(self.device).unsqueeze(0)
        f0_int = torch.round(f0_tensor).long()

        # 推理
        sid = torch.tensor([0], device=self.device)
        audio_out = self.voice_model.infer(
            features,
            torch.tensor([features.shape[1]], device=self.device),
            f0_int,
            f0_tensor,
            sid
        )

        # 后处理
        audio_out = audio_out.squeeze().cpu().numpy()

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

        # 保存
        save_audio(output_path, audio_out, sr=save_sr)

        return output_path


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

    for pth_file in weights_path.glob("*.pth"):
        # 查找对应的索引文件
        index_file = pth_file.with_suffix(".index")
        if not index_file.exists():
            # 尝试其他命名方式
            index_file = weights_path / f"{pth_file.stem}_v2.index"

        models.append({
            "name": pth_file.stem,
            "model_path": str(pth_file),
            "index_path": str(index_file) if index_file.exists() else None
        })

    return models
