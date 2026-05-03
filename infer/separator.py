# -*- coding: utf-8 -*-
"""
人声分离模块 - 支持 Demucs 和 Mel-Band Roformer (audio-separator)
"""
import os
import gc
import shutil
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Callable, Union

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


ModelSpec = Union[str, list[str], tuple[str, ...]]


# Public scored SOTA defaults from audio-separator 0.44.1's model table.
# Keep the cover pipeline unchanged; only the separator model choices change.
ENSEMBLE_PRESET_PREFIX = "ensemble:"

ROFORMER_LEGACY_SINGLE_MODEL = "vocals_mel_band_roformer.ckpt"
ROFORMER_SOTA_PRESET = "vocal_rvc"
ROFORMER_DEFAULT_MODEL = f"{ENSEMBLE_PRESET_PREFIX}{ROFORMER_SOTA_PRESET}"
ROFORMER_SOTA_MODEL = ROFORMER_DEFAULT_MODEL
ROFORMER_SOTA_MODELS = [
    "melband_roformer_big_beta6x.ckpt",
    "mel_band_roformer_vocals_fv4_gabox.ckpt",
]

KARAOKE_LEGACY_SINGLE_MODEL = "mel_band_roformer_karaoke_gabox.ckpt"
KARAOKE_SOTA_PRESET = "karaoke"
KARAOKE_DEFAULT_MODEL = f"{ENSEMBLE_PRESET_PREFIX}{KARAOKE_SOTA_PRESET}"
KARAOKE_SOTA_MODEL = KARAOKE_DEFAULT_MODEL
KARAOKE_SOTA_MODELS = [
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "mel_band_roformer_karaoke_becruily.ckpt",
]
KARAOKE_EXPERIMENTAL_MODELS = [
    "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "mel_band_roformer_karaoke_becruily.ckpt",
]

ROFORMER_DEREVERB_DEFAULT_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"


def _model_spec_key(model_spec: ModelSpec) -> tuple[str, ...]:
    if isinstance(model_spec, (list, tuple)):
        return tuple(str(item) for item in model_spec)
    return (str(model_spec),)


def _model_spec_label(model_spec: ModelSpec) -> str:
    if isinstance(model_spec, (list, tuple)):
        return "ensemble[" + ", ".join(str(item) for item in model_spec) + "]"
    return str(model_spec)


def _parse_ensemble_preset(model_spec: ModelSpec) -> Optional[str]:
    if not isinstance(model_spec, str):
        return None
    spec = model_spec.strip()
    if not spec.lower().startswith(ENSEMBLE_PRESET_PREFIX):
        return None
    preset = spec[len(ENSEMBLE_PRESET_PREFIX):].strip()
    return preset or None


def _load_audio_separator_model(
    *,
    model_spec: ModelSpec,
    output_dir: str,
    model_dir: str,
) -> Separator:
    preset_name = _parse_ensemble_preset(model_spec)
    separator_kwargs = {
        "log_level": _logging.WARNING,
        "output_dir": output_dir,
        "model_file_dir": model_dir,
    }
    if preset_name:
        separator_kwargs["ensemble_preset"] = preset_name

    separator = Separator(**separator_kwargs)
    if preset_name:
        separator.load_model()
    else:
        separator.load_model(list(model_spec) if isinstance(model_spec, tuple) else model_spec)
    return separator


def _resolve_output_files(output_files, output_dir: Path) -> list[str]:
    """Resolve relative output filenames returned by audio-separator."""
    resolved_files = []
    for file_name in output_files:
        file_path = Path(file_name)
        if not file_path.is_absolute():
            file_path = output_dir / file_path
        resolved_files.append(str(file_path))
    return resolved_files


def _safe_move(src_path: str, dst_path: str) -> None:
    """Move file with overwrite."""
    if src_path == dst_path:
        return
    dst = Path(dst_path)
    if dst.exists():
        dst.unlink()
    shutil.move(src_path, dst_path)


def _get_audio_activity_stats(audio_path: str) -> tuple[float, float, int]:
    """Return simple activity stats for validating separator outputs."""
    audio, _ = sf.read(audio_path, dtype="float32", always_2d=True)
    if audio.size == 0:
        return 0.0, 0.0, 0

    mono = np.mean(audio, axis=1, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64) + 1e-12))
    peak = float(np.max(np.abs(mono)))
    nonzero = int(np.count_nonzero(np.abs(mono) > 1e-6))
    return rms, peak, nonzero


class RoformerSeparator:
    """人声分离器 - 基于 Mel-Band Roformer (通过 audio-separator)"""

    def __init__(
        self,
        model_filename: ModelSpec = ROFORMER_DEFAULT_MODEL,
        device: str = "cuda",
    ):
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError(
                "请安装 audio-separator: pip install audio-separator[gpu]"
        )
        self.model_filename = model_filename
        self.model_candidates = [model_filename]
        self.device = str(get_device(device))
        self.separator = None
        self.active_model = None

    def load_model(self, output_dir: str = ""):
        """加载指定 RoFormer 模型；严格 SOTA 模式下不自动降级。"""
        model_dir = str(
            Path(__file__).parent.parent / "assets" / "separator_models"
        )
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        target_dir = output_dir or str(
            Path(__file__).parent.parent / "temp" / "separator"
        )

        # Recreate the Separator when output_dir changes, because
        # some audio-separator versions cache internal paths at init.
        if self.separator is not None:
            if getattr(self, '_init_output_dir', None) == target_dir:
                return
            # output_dir changed — rebuild Separator
            del self.separator
            self.separator = None
            gc.collect()

        model_name = self.model_filename
        log.info(
            "正在加载公开 SOTA RoFormer 分离模型: "
            f"{_model_spec_label(model_name)}"
        )
        separator = _load_audio_separator_model(
            model_spec=model_name,
            output_dir=target_dir,
            model_dir=model_dir,
        )
        self.separator = separator
        self._init_output_dir = target_dir
        self.active_model = model_name
        log.info(
            "RoFormer 分离模型已加载: "
            f"{_model_spec_label(model_name)}"
        )

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

        if progress_callback:
            progress_callback("正在使用 RoFormer 分离人声...", 0.3)

        self.load_model(output_dir=str(output_path))
        # audio-separator 需要 output_dir 在实例上设置
        self.separator.output_dir = str(output_path)
        output_files = self.separator.separate(audio_path)

        # audio-separator 返回的可能是纯文件名，需要拼上 output_dir
        resolved_files = []
        for f in output_files:
            p = Path(f)
            if not p.is_absolute():
                p = output_path / p
            resolved_files.append(str(p))

        # Recovery: if resolved files don't exist, search the output dir
        # for freshly created files. This handles cases where audio-separator
        # writes to a slightly different path (e.g. after output_dir update
        # on a reused Separator instance).
        if resolved_files and not any(Path(f).exists() for f in resolved_files):
            import glob as _glob
            all_wavs = sorted(
                _glob.glob(str(output_path / "*.wav")),
                key=lambda x: os.path.getmtime(x),
                reverse=True,
            )
            # Take the most recent files (should be our separation output)
            if len(all_wavs) >= 2:
                resolved_files = all_wavs[:2]
            elif len(all_wavs) == 1:
                resolved_files = all_wavs[:1]

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
            if not Path(vocals_path).exists():
                raise FileNotFoundError(
                    f"分离器输出人声文件不存在: {vocals_path}\n"
                    f"输出目录内容: {list(output_path.glob('*'))}"
                )
            shutil.move(vocals_path, final_vocals)
        if accompaniment_path and accompaniment_path != final_accompaniment:
            if not Path(accompaniment_path).exists():
                raise FileNotFoundError(
                    f"分离器输出伴奏文件不存在: {accompaniment_path}\n"
                    f"输出目录内容: {list(output_path.glob('*'))}"
                )
            shutil.move(accompaniment_path, final_accompaniment)

        if progress_callback:
            progress_callback("Mel-Band Roformer 人声分离完成", 1.0)

        return final_vocals, final_accompaniment

    def unload_model(self):
        """卸载模型释放显存"""
        if self.separator is not None:
            del self.separator
            self.separator = None
        self.active_model = None
        gc.collect()
        empty_device_cache()


class KaraokeSeparator:
    """主唱/和声分离器 - 基于 Mel-Band Roformer Karaoke 模型"""

    def __init__(
        self,
        model_filename: ModelSpec = KARAOKE_DEFAULT_MODEL,
        device: str = "cuda",
    ):
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError(
                "请安装 audio-separator: pip install audio-separator[gpu]"
            )
        self.device = str(get_device(device))
        self.separator = None
        self.active_model = None
        self.model_filename = model_filename
        self.model_candidates = [model_filename]

    def load_model(self, output_dir: str = ""):
        """加载指定 Karaoke 模型；严格 SOTA 模式下不自动降级。"""
        model_dir = str(Path(__file__).parent.parent / "assets" / "separator_models")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        target_dir = output_dir or str(
            Path(__file__).parent.parent / "temp" / "separator"
        )

        # Recreate the Separator when output_dir changes
        if self.separator is not None:
            if getattr(self, '_init_output_dir', None) == target_dir:
                return
            del self.separator
            self.separator = None
            self.active_model = None
            gc.collect()

        model_name = self.model_filename
        log.info(
            "正在加载公开 SOTA Karaoke 模型: "
            f"{_model_spec_label(model_name)}"
        )
        separator = _load_audio_separator_model(
            model_spec=model_name,
            output_dir=target_dir,
            model_dir=model_dir,
        )
        self.separator = separator
        self._init_output_dir = target_dir
        self.active_model = model_name
        log.info(
            "Karaoke 模型已加载: "
            f"{_model_spec_label(model_name)}"
        )

    @staticmethod
    def _classify_stem(file_name: str) -> Optional[str]:
        lower_name = file_name.lower()

        lead_markers = [
            "(vocals)",
            "(lead)",
            "(karaoke)",
            "(main_vocal)",
            "(main vocals)",
            "_(vocals)_",
        ]
        backing_markers = [
            "(instrumental)",
            "(other)",
            "(backing)",
            "(no_vocal",
            "_(instrumental)_",
            "_(other)_",
        ]

        for marker in lead_markers:
            if marker in lower_name:
                return "lead"
        for marker in backing_markers:
            if marker in lower_name:
                return "backing"

        if "vocals" in lower_name:
            return "lead"
        if "instrumental" in lower_name or "other" in lower_name:
            return "backing"
        return None

    def separate(self, audio_path: str, output_dir: str) -> Tuple[str, str]:
        """
        分离主唱和和声

        Returns:
            Tuple[lead_vocals_path, backing_vocals_path]
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.load_model(output_dir=str(output_path))
        self.separator.output_dir = str(output_path)
        output_files = self.separator.separate(audio_path)

        resolved_files = _resolve_output_files(output_files, output_path)
        log.detail(
            f"Karaoke分离器输出文件: {[Path(file_path).name for file_path in resolved_files]}"
        )

        lead_vocals_path = None
        backing_vocals_path = None
        for file_path in resolved_files:
            stem_role = self._classify_stem(Path(file_path).name)
            log.detail(
                f"  {Path(file_path).name} -> 分类为: {stem_role or 'unknown'}"
            )
            if stem_role == "lead" and lead_vocals_path is None:
                lead_vocals_path = file_path
            elif stem_role == "backing" and backing_vocals_path is None:
                backing_vocals_path = file_path

        if lead_vocals_path is None and resolved_files:
            lead_vocals_path = resolved_files[0]
        if backing_vocals_path is None:
            for file_path in resolved_files:
                if file_path != lead_vocals_path:
                    backing_vocals_path = file_path
                    break

        if not lead_vocals_path or not Path(lead_vocals_path).exists():
            raise FileNotFoundError(
                f"Karaoke主唱轨未找到，输出文件: {[Path(p).name for p in resolved_files]}"
            )
        if not backing_vocals_path or not Path(backing_vocals_path).exists():
            raise FileNotFoundError(
                f"Karaoke和声轨未找到，输出文件: {[Path(p).name for p in resolved_files]}"
            )

        lead_rms, lead_peak, lead_nonzero = _get_audio_activity_stats(lead_vocals_path)
        backing_rms, backing_peak, backing_nonzero = _get_audio_activity_stats(backing_vocals_path)
        log.detail(
            "Karaoke输出能量检测: "
            f"lead_rms={lead_rms:.6f}, lead_peak={lead_peak:.6f}, lead_nonzero={lead_nonzero}; "
            f"backing_rms={backing_rms:.6f}, backing_peak={backing_peak:.6f}, backing_nonzero={backing_nonzero}"
        )

        lead_is_nearly_silent = lead_nonzero == 0 or (lead_rms < 1e-5 and lead_peak < 1e-4)
        backing_has_content = backing_nonzero > 0 and (backing_rms >= 5e-5 or backing_peak >= 5e-4)
        if lead_is_nearly_silent and backing_has_content:
            log.warning("Karaoke主唱轨几乎静音，检测到输出疑似反转，已自动交换主唱/和声")
            lead_vocals_path, backing_vocals_path = backing_vocals_path, lead_vocals_path

        final_lead = str(output_path / "lead_vocals.wav")
        final_backing = str(output_path / "backing_vocals.wav")
        _safe_move(lead_vocals_path, final_lead)
        _safe_move(backing_vocals_path, final_backing)

        return final_lead, final_backing

    def unload_model(self):
        """卸载模型释放显存"""
        if self.separator is not None:
            del self.separator
            self.separator = None
        self.active_model = None
        gc.collect()
        empty_device_cache()


class RoformerDereverbSeparator:
    """学习型 RoFormer 去混响/去回声，输出更干的人声供 VC 使用。"""

    def __init__(
        self,
        model_filename: str = ROFORMER_DEREVERB_DEFAULT_MODEL,
        device: str = "cuda",
    ):
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError(
                "请安装 audio-separator: pip install audio-separator[gpu]"
            )
        self.device = str(get_device(device))
        self.separator = None
        self.active_model = None
        self.model_filename = model_filename
        self.model_candidates = [model_filename]

    def load_model(self, output_dir: str = ""):
        model_dir = str(Path(__file__).parent.parent / "assets" / "separator_models")
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        target_dir = output_dir or str(
            Path(__file__).parent.parent / "temp" / "separator"
        )

        if self.separator is not None:
            if getattr(self, "_init_output_dir", None) == target_dir:
                return
            del self.separator
            self.separator = None
            self.active_model = None
            gc.collect()

        model_name = self.model_filename
        log.info(f"正在加载 RoFormer De-Reverb 模型: {model_name}")
        separator = _load_audio_separator_model(
            model_spec=model_name,
            output_dir=target_dir,
            model_dir=model_dir,
        )
        self.separator = separator
        self._init_output_dir = target_dir
        self.active_model = model_name
        log.info(f"RoFormer De-Reverb 模型已加载: {model_name}")

    @staticmethod
    def _classify_stem(file_name: str) -> Optional[str]:
        lower_name = file_name.lower()

        dry_markers = [
            "(dry)",
            "(noreverb)",
            "(no_reverb)",
            "(no reverb)",
            "(dereverb)",
            "(de-reverb)",
            "(vocals)",
            "(primary)",
        ]
        wet_markers = [
            "(no dry)",
            "(no_dry)",
            "(reverb)",
            "(echo)",
            "(wet)",
            "(secondary)",
            "(instrumental)",
            "(other)",
        ]

        for marker in wet_markers:
            if marker in lower_name:
                return "wet"
        for marker in dry_markers:
            if marker in lower_name:
                return "dry"

        if "dry" in lower_name or "noreverb" in lower_name or "vocal" in lower_name:
            return "dry"
        if "no dry" in lower_name or "reverb" in lower_name or "echo" in lower_name:
            return "wet"
        return None

    def separate_dry(self, audio_path: str, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.load_model(output_dir=str(output_path))
        self.separator.output_dir = str(output_path)
        output_files = self.separator.separate(audio_path)

        resolved_files = _resolve_output_files(output_files, output_path)
        log.detail(
            "RoFormer De-Reverb 输出文件: "
            f"{[Path(file_path).name for file_path in resolved_files]}"
        )

        dry_path = None
        for file_path in resolved_files:
            stem_role = self._classify_stem(Path(file_path).name)
            log.detail(
                f"  {Path(file_path).name} -> 分类为: {stem_role or 'unknown'}"
            )
            if stem_role == "dry":
                dry_path = file_path
                break

        if dry_path is None and resolved_files:
            dry_path = resolved_files[0]

        if not dry_path or not Path(dry_path).exists():
            raise FileNotFoundError(
                f"RoFormer De-Reverb dry轨未找到，输出文件: {[Path(p).name for p in resolved_files]}"
            )

        final_dry = str(output_path / "roformer_deecho_vocals.wav")
        _safe_move(dry_path, final_dry)
        return final_dry

    def unload_model(self):
        if self.separator is not None:
            del self.separator
            self.separator = None
        self.active_model = None
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
            "description": "audio-separator public scored SOTA - 最高质量人声/伴奏分离"
        })
    if DEMUCS_AVAILABLE:
        models.extend([
            {"name": "htdemucs", "description": "Demucs 默认模型，平衡质量和速度 (SDR ~9dB)"},
            {"name": "htdemucs_ft", "description": "Demucs 微调版本，质量更高但更慢"},
            {"name": "mdx_extra", "description": "MDX 模型，适合某些音乐类型"},
        ])
    return models
