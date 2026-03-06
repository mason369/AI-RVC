# -*- coding: utf-8 -*-
"""
人声分离模块 - 支持 Demucs 和 Mel-Band Roformer (audio-separator)
"""
import os
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable

from lib.logger import log
from lib.device import get_device, empty_device_cache

# Demucs 导入
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torchaudio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# audio-separator 导入 (Mel-Band Roformer 等)
try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_AVAILABLE = True
    # 抑制 audio-separator 的英文日志，我们有自己的中文日志
    import logging as _logging
    _logging.getLogger("audio_separator").setLevel(_logging.WARNING)
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False


# Mel-Band Roformer 默认模型
ROFORMER_DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"


class RoformerSeparator:
    """人声分离器 - 基于 Mel-Band Roformer (通过 audio-separator)"""

    def __init__(
        self,
        model_filename: str = ROFORMER_DEFAULT_MODEL,
        device: str = "cuda",
    ):
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError(
                "请安装 audio-separator: pip install audio-separator[gpu]"
            )
        self.model_filename = model_filename
        self.device = str(get_device(device))
        self.separator = None

    def load_model(self, output_dir: str = ""):
        """加载 Roformer 模型"""
        if self.separator is not None:
            return

        log.info(f"正在加载 Mel-Band Roformer 模型: {self.model_filename}")

        model_dir = str(
            Path(__file__).parent.parent / "assets" / "separator_models"
        )
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        self.separator = Separator(
            output_dir=output_dir or str(
                Path(__file__).parent.parent / "temp" / "separator"
            ),
            model_file_dir=model_dir,
        )
        self.separator.load_model(self.model_filename)
        log.info(f"Mel-Band Roformer 模型已加载")

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[str, str]:
        """
        分离人声和伴奏

        Returns:
            Tuple[vocals_path, accompaniment_path]
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback("正在加载 Roformer 模型...", 0.1)

        self.load_model(output_dir=str(output_path))

        # audio-separator 需要 output_dir 在实例上设置
        self.separator.output_dir = str(output_path)

        if progress_callback:
            progress_callback("正在使用 Mel-Band Roformer 分离人声...", 0.3)

        output_files = self.separator.separate(audio_path)

        # audio-separator 返回的可能是纯文件名，需要拼上 output_dir
        resolved_files = []
        for f in output_files:
            p = Path(f)
            if not p.is_absolute():
                p = output_path / p
            resolved_files.append(str(p))

        # audio-separator 返回文件列表，通常 [primary, secondary]
        # primary = Vocals, secondary = Instrumental (或反过来，取决于模型)
        vocals_path = None
        accompaniment_path = None

        for f in resolved_files:
            f_lower = Path(f).name.lower()
            # audio-separator uses parenthesized stem markers like (vocals), (other)
            # Check these first to avoid false matches from model names (e.g. vocals_mel_band_roformer)
            if "(other)" in f_lower or "(instrumental)" in f_lower or "(no_vocal" in f_lower:
                accompaniment_path = f
            elif "(vocal" in f_lower or "(primary)" in f_lower:
                vocals_path = f
            elif "instrument" in f_lower or "no_vocal" in f_lower or "secondary" in f_lower:
                accompaniment_path = f
            elif "vocal" in f_lower or "primary" in f_lower:
                vocals_path = f

        # 如果无法通过文件名判断，按顺序分配
        if vocals_path is None and accompaniment_path is None and len(resolved_files) >= 2:
            vocals_path = resolved_files[0]
            accompaniment_path = resolved_files[1]
        elif vocals_path is None and len(resolved_files) >= 1:
            vocals_path = resolved_files[0]
        elif accompaniment_path is None and len(resolved_files) >= 2:
            accompaniment_path = resolved_files[1]

        # 重命名为标准名称
        final_vocals = str(output_path / "vocals.wav")
        final_accompaniment = str(output_path / "accompaniment.wav")

        if vocals_path and vocals_path != final_vocals:
            import shutil
            shutil.move(vocals_path, final_vocals)
        if accompaniment_path and accompaniment_path != final_accompaniment:
            import shutil
            shutil.move(accompaniment_path, final_accompaniment)

        if progress_callback:
            progress_callback("Mel-Band Roformer 人声分离完成", 1.0)

        return final_vocals, final_accompaniment

    def unload_model(self):
        """卸载模型释放显存"""
        if self.separator is not None:
            del self.separator
            self.separator = None
        gc.collect()
        empty_device_cache()


class VocalSeparator:
    """人声分离器 - 基于 Demucs"""

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cuda",
        shifts: int = 2,
        overlap: float = 0.25,
        split: bool = True
    ):
        """
        初始化分离器

        Args:
            model_name: Demucs 模型名称 (htdemucs, htdemucs_ft, mdx_extra)
            device: 计算设备
        """
        if not DEMUCS_AVAILABLE:
            raise ImportError("请安装 demucs: pip install demucs")

        self.model_name = model_name
        self.device = str(get_device(device))
        self.model = None
        self.shifts = shifts
        self.overlap = overlap
        self.split = split

    def load_model(self):
        """加载 Demucs 模型"""
        if self.model is not None:
            return

        log.info(f"正在加载 Demucs 模型: {self.model_name}")
        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        log.info(f"Demucs 模型已加载 ({self.device})")

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[str, str]:
        """
        分离人声和伴奏

        Args:
            audio_path: 输入音频路径
            output_dir: 输出目录
            progress_callback: 进度回调 (message, progress)

        Returns:
            Tuple[vocals_path, accompaniment_path]
        """
        self.load_model()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback("正在加载音频...", 0.1)

        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)

        # 重采样到模型采样率
        if sample_rate != self.model.samplerate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.model.samplerate)
            waveform = resampler(waveform)

        # 确保是立体声
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # 添加 batch 维度
        waveform = waveform.unsqueeze(0).to(self.device)

        if progress_callback:
            progress_callback("正在分离人声...", 0.3)

        # 执行分离
        with torch.no_grad():
            try:
                sources = apply_model(
                    self.model,
                    waveform,
                    device=self.device,
                    shifts=self.shifts,
                    overlap=self.overlap,
                    split=self.split
                )
            except TypeError:
                sources = apply_model(self.model, waveform, device=self.device)

        # sources 形状: (batch, sources, channels, samples)
        # 获取各音轨索引
        source_names = self.model.sources
        vocals_idx = source_names.index("vocals")
        drums_idx = source_names.index("drums")
        bass_idx = source_names.index("bass")
        other_idx = source_names.index("other")

        # 提取人声
        vocals = sources[0, vocals_idx]  # (channels, samples)

        # 合并非人声音轨作为伴奏
        accompaniment = sources[0, drums_idx] + sources[0, bass_idx] + sources[0, other_idx]

        if progress_callback:
            progress_callback("正在保存分离结果...", 0.8)

        # 保存结果
        vocals_path = output_path / "vocals.wav"
        accompaniment_path = output_path / "accompaniment.wav"

        # 保存为 WAV
        torchaudio.save(
            str(vocals_path),
            vocals.cpu(),
            self.model.samplerate
        )
        torchaudio.save(
            str(accompaniment_path),
            accompaniment.cpu(),
            self.model.samplerate
        )

        if progress_callback:
            progress_callback("人声分离完成", 1.0)

        # 释放显存
        empty_device_cache()

        return str(vocals_path), str(accompaniment_path)

    def unload_model(self):
        """卸载模型释放显存"""
        if self.model is not None:
            self.model.cpu()  # 先移到 CPU
            del self.model
            self.model = None
        gc.collect()
        empty_device_cache()


def check_demucs_available() -> bool:
    """检查 Demucs 是否可用"""
    return DEMUCS_AVAILABLE


def check_roformer_available() -> bool:
    """检查 audio-separator (Roformer) 是否可用"""
    return AUDIO_SEPARATOR_AVAILABLE


def get_available_models() -> list:
    """获取可用的分离模型列表"""
    models = []
    if AUDIO_SEPARATOR_AVAILABLE:
        models.append({
            "name": "roformer",
            "description": "Mel-Band Roformer (Kimberley Jensen) - 高质量人声分离"
        })
    if DEMUCS_AVAILABLE:
        models.extend([
            {"name": "htdemucs", "description": "Demucs 默认模型，平衡质量和速度 (SDR ~9dB)"},
            {"name": "htdemucs_ft", "description": "Demucs 微调版本，质量更高但更慢"},
            {"name": "mdx_extra", "description": "MDX 模型，适合某些音乐类型"},
        ])
    return models
