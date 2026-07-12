# -*- coding: utf-8 -*-
"""
人声分离模块 - 支持 BS PolarFormer、RoFormer/BS-RoFormer 和 Demucs
"""
from __future__ import annotations

import os
import gc
import shutil
import math
import subprocess
import torch
import numpy as np
import soundfile as sf
import logging as _logging
from pathlib import Path
from typing import Tuple, Optional, Callable, Union, Any

from lib.logger import log
from lib.device import get_device, empty_device_cache
from lib.ffmpeg_runtime import get_ffmpeg_bin_dir

# Demucs 导入
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torchaudio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# audio-separator 导入 (RoFormer/BS-RoFormer、De-Reverb 等)
try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_AVAILABLE = True
    AUDIO_SEPARATOR_IMPORT_ERROR = None
    # 抑制 audio-separator 的英文日志，我们有自己的中文日志
    _logging.getLogger("audio_separator").setLevel(_logging.WARNING)
except ImportError as exc:
    Separator = None
    AUDIO_SEPARATOR_AVAILABLE = False
    AUDIO_SEPARATOR_IMPORT_ERROR = exc


ModelSpec = Union[str, list[str], tuple[str, ...]]


def get_audio_separator_unavailable_reason() -> str:
    """Return the original audio-separator import failure, if any."""
    if AUDIO_SEPARATOR_AVAILABLE:
        return ""
    if AUDIO_SEPARATOR_IMPORT_ERROR is None:
        return "audio-separator 未安装或不可导入"
    return str(AUDIO_SEPARATOR_IMPORT_ERROR)


def _audio_separator_install_message() -> str:
    message = "请安装 audio-separator[cpu] 或 audio-separator[gpu]"
    reason = get_audio_separator_unavailable_reason()
    if reason:
        message += f"；原始错误: {reason}"
    return message


# Quality-oriented separator defaults for RVC cover preprocessing.
# Constant names keep the existing public API stable; docs avoid absolute license claims.
ENSEMBLE_PRESET_PREFIX = "ensemble:"

ROFORMER_LEGACY_SINGLE_MODEL = "vocals_mel_band_roformer.ckpt"
LEAP_XE_VOCALS_MODEL = "bs_roformer_leap_xe_voc.ckpt"
LEAP_XE_VOCALS_DISPLAY_NAME = "BS-RoFormer Leap XE 90 bands (pcunwa)"
LEAP_XE_VOCALS_UPSTREAM_MODEL_FILENAME = "Xe/bs_leap_xe_voc.ckpt"
LEAP_XE_VOCALS_CONFIG_FILENAME = "Xe/leap_xe_config_voc.yaml"
LEAP_XE_VOCALS_RUNTIME_CONFIG_FILENAME = "Xe/bs_roformer_leap_xe_config_voc.yaml"
BS_POLARFORMER_MODEL = "bs_polarformer_public_onnx_62bands"
BS_POLARFORMER_HF_REPO = "bgkb/bs_polarformer"
BS_POLARFORMER_ONNX_FILENAME = "bs_polarformer.onnx"
BS_POLARFORMER_CONFIG_FILENAME = "model_bs_polarformer_float16.yaml"
BS_POLARFORMER_DISPLAY_NAME = (
    "BS PolarFormer public ONNX 62 bands (bgkb/ZFTurbo)"
)
DEFAULT_POLARFORMER_MAX_CHUNK_SIZE = 441000
_SATURATION_FRAME_SECONDS = 0.25
_SATURATION_HOP_SECONDS = 0.125
_SATURATION_FADE_SECONDS = 0.05
HYBRID_LEAP_XE_POLARFORMER_PRESET = "leap_xe90_vocals+polarformer62_instrumental"
HYBRID_SEPARATOR_PREFIX = "hybrid:"
ROFORMER_SOTA_PRESET = HYBRID_LEAP_XE_POLARFORMER_PRESET
ROFORMER_DEFAULT_MODEL = f"{HYBRID_SEPARATOR_PREFIX}{ROFORMER_SOTA_PRESET}"
ROFORMER_SOTA_MODEL = ROFORMER_DEFAULT_MODEL
ROFORMER_SOTA_MODELS = [
    LEAP_XE_VOCALS_MODEL,
    LEAP_XE_VOCALS_CONFIG_FILENAME,
    LEAP_XE_VOCALS_RUNTIME_CONFIG_FILENAME,
    BS_POLARFORMER_ONNX_FILENAME,
    BS_POLARFORMER_CONFIG_FILENAME,
]

KARAOKE_LEGACY_SINGLE_MODEL = "mel_band_roformer_karaoke_gabox.ckpt"
KARAOKE_LEGACY_AUDIO_SEPARATOR_PRESET = "karaoke"
KARAOKE_MVSEP_9205_PRESET = "mvsep_9205_avg"
KARAOKE_SOTA_PRESET = KARAOKE_MVSEP_9205_PRESET
KARAOKE_DEFAULT_MODEL = f"{ENSEMBLE_PRESET_PREFIX}{KARAOKE_SOTA_PRESET}"
KARAOKE_SOTA_MODEL = KARAOKE_DEFAULT_MODEL
KARAOKE_SOTA_DISPLAY_NAME = (
    "BS-Kar-Gabox_IS + BS-Kar-Frazer&Becruily + BS-Kar-Anvuew (AVG)"
)
KARAOKE_SOTA_MODELS = [
    "bs_karaoke_gabox_IS.ckpt",
    "bs_roformer_karaoke_frazer_becruily.ckpt",
    "karaoke_bs_roformer_anvuew.ckpt",
]
KARAOKE_EXPERIMENTAL_MODELS = [
    "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "mel_band_roformer_karaoke_becruily.ckpt",
]

ROFORMER_DEREVERB_DEFAULT_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"

_CUSTOM_AUDIO_SEPARATOR_MODELS: dict[str, dict[str, str]] = {
    LEAP_XE_VOCALS_MODEL: {
        "repo_id": "pcunwa/BS-Roformer-Leap",
        "model_filename": LEAP_XE_VOCALS_UPSTREAM_MODEL_FILENAME,
        "config_filename": LEAP_XE_VOCALS_CONFIG_FILENAME,
        "runtime_config_filename": LEAP_XE_VOCALS_RUNTIME_CONFIG_FILENAME,
        "friendly_name": "Roformer Model: BS-RoFormer Leap XE Vocals by pcunwa",
    },
    "bs_karaoke_gabox_IS.ckpt": {
        "repo_id": "GaboxR67/MelBandRoformers",
        "model_filename": "bsroformers/bs_karaoke_gabox_IS.ckpt",
        "config_filename": "bsroformers/karaoke_bs_roformer.yaml",
        "friendly_name": "Roformer Model: BS-Karaoke Gabox IS",
    },
    "bs_roformer_karaoke_frazer_becruily.ckpt": {
        "repo_id": "becruily/bs-roformer-karaoke",
        "model_filename": "bs_roformer_karaoke_frazer_becruily.ckpt",
        "config_filename": "config_karaoke_frazer_becruily.yaml",
        "friendly_name": "Roformer Model: BS-Karaoke Frazer & Becruily",
    },
    "karaoke_bs_roformer_anvuew.ckpt": {
        "repo_id": "anvuew/karaoke_bs_roformer",
        "model_filename": "karaoke_bs_roformer_anvuew.ckpt",
        "config_filename": "karaoke_bs_roformer_anvuew.yaml",
        "friendly_name": "Roformer Model: BS-Karaoke Anvuew",
    },
}

_CUSTOM_ENSEMBLE_PRESETS: dict[str, dict[str, Any]] = {
    KARAOKE_MVSEP_9205_PRESET: {
        "name": "MVSep 9205 AVG",
        "models": KARAOKE_SOTA_MODELS,
        "algorithm": "avg_wave",
        "weights": None,
    },
}


def _model_spec_key(model_spec: ModelSpec) -> tuple[str, ...]:
    if isinstance(model_spec, (list, tuple)):
        return tuple(str(item) for item in model_spec)
    return (str(model_spec),)


def _model_spec_label(model_spec: ModelSpec) -> str:
    if isinstance(model_spec, (list, tuple)):
        return "ensemble[" + ", ".join(str(item) for item in model_spec) + "]"
    if str(model_spec).strip().lower() == ROFORMER_DEFAULT_MODEL.lower():
        return f"{LEAP_XE_VOCALS_DISPLAY_NAME} + {BS_POLARFORMER_DISPLAY_NAME}"
    if str(model_spec).strip().lower() == KARAOKE_DEFAULT_MODEL.lower():
        return KARAOKE_SOTA_DISPLAY_NAME
    return str(model_spec)


def _parse_ensemble_preset(model_spec: ModelSpec) -> Optional[str]:
    if not isinstance(model_spec, str):
        return None
    spec = model_spec.strip()
    if not spec.lower().startswith(ENSEMBLE_PRESET_PREFIX):
        return None
    preset = spec[len(ENSEMBLE_PRESET_PREFIX):].strip()
    return preset or None


def _parse_hybrid_preset(model_spec: ModelSpec) -> Optional[str]:
    if not isinstance(model_spec, str):
        return None
    spec = model_spec.strip()
    if not spec.lower().startswith(HYBRID_SEPARATOR_PREFIX):
        return None
    preset = spec[len(HYBRID_SEPARATOR_PREFIX):].strip()
    return preset or None


def _is_hybrid_leap_xe_polarformer_model_spec(model_spec: ModelSpec) -> bool:
    preset = _parse_hybrid_preset(model_spec)
    return (preset or "").lower() == HYBRID_LEAP_XE_POLARFORMER_PRESET.lower()


def _is_bs_polarformer_model_spec(model_spec: ModelSpec) -> bool:
    if not isinstance(model_spec, str):
        return False
    spec = model_spec.strip().lower()
    return spec in {
        BS_POLARFORMER_MODEL.lower(),
        BS_POLARFORMER_ONNX_FILENAME.lower(),
    }


def get_separator_chain_labels(
    separator_name: str,
    roformer_model: ModelSpec,
    karaoke_enabled: bool,
    karaoke_model: ModelSpec,
) -> list[str]:
    """Return the model names that the current separation task will execute."""
    labels = []
    normalized_separator = str(separator_name).strip().lower()

    if normalized_separator == "roformer":
        if _is_hybrid_leap_xe_polarformer_model_spec(roformer_model):
            labels.extend(
                [
                    "输入: Leap XE 与 PolarFormer 共用 44.1kHz 双声道 PCM（非WAV预解码）",
                    f"人声: {LEAP_XE_VOCALS_DISPLAY_NAME}",
                    f"纯伴奏: {BS_POLARFORMER_DISPLAY_NAME}（含孤立声道饱和保护）",
                ]
            )
        else:
            labels.append(f"人声/伴奏: {_model_spec_label(roformer_model)}")

    if karaoke_enabled:
        labels.append(f"主唱/带和声伴奏: {_model_spec_label(karaoke_model)}")
        labels.append("纯和声: Leap 人声 - MVSep 主唱")

    return labels


def _download_hf_file(repo_id: str, filename: str, model_dir: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("请安装 huggingface_hub 以下载外部分离模型") from exc

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"下载模型文件失败: repo={repo_id}, file={filename}"
        ) from exc


def _resolve_polarformer_chunk_size(configured_chunk_size: int) -> int:
    """Apply TelkNet's bounded PolarFormer inference window."""
    if configured_chunk_size <= 0:
        raise RuntimeError(
            f"BS PolarFormer chunk_size 无效: {configured_chunk_size}"
        )

    raw_override = os.environ.get("POLARFORMER_MAX_CHUNK_SIZE", "").strip()
    if raw_override:
        try:
            max_chunk_size = int(raw_override)
        except ValueError as exc:
            raise RuntimeError(
                f"POLARFORMER_MAX_CHUNK_SIZE 无效: {raw_override!r}"
            ) from exc
    else:
        max_chunk_size = DEFAULT_POLARFORMER_MAX_CHUNK_SIZE

    if max_chunk_size <= 0:
        return configured_chunk_size
    return max(1, min(configured_chunk_size, max_chunk_size))


def _ensure_separator_pcm_wav(audio_path: str, work_dir: Path) -> str:
    """Match TelkNet by decoding non-WAV input once before both separators."""
    source_path = Path(audio_path)
    if source_path.suffix.lower() == ".wav":
        return str(source_path)

    ffmpeg_bin_dir = get_ffmpeg_bin_dir()
    if ffmpeg_bin_dir is None:
        raise RuntimeError(
            "非 WAV 音频进入分离链路前需要 FFmpeg，但当前未找到 ffmpeg 可执行文件"
        )
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_path = ffmpeg_bin_dir / ffmpeg_name
    if not ffmpeg_path.is_file():
        raise RuntimeError(f"FFmpeg 可执行文件不存在: {ffmpeg_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / f"{source_path.stem}_separator_input.wav"
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source_path),
        "-ar",
        "44100",
        "-ac",
        "2",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("分离前解码需要 FFmpeg，但执行文件无法启动") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = f"\nFFmpeg stderr:\n{stderr}" if stderr else ""
        raise RuntimeError(f"FFmpeg 无法解码 {source_path.name}.{detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg 解码 {source_path.name} 超时") from exc
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"FFmpeg 解码 {source_path.name} 失败: {exc}") from exc

    if not wav_path.is_file() or wav_path.stat().st_size <= 0:
        raise RuntimeError(f"FFmpeg 未生成有效的分离输入: {wav_path}")
    return str(wav_path)


def _suppress_isolated_channel_saturation(
    estimate: np.ndarray,
    mixture: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """抑制 PolarFormer 偶发的单声道持续满幅异常。"""
    if (
        estimate.ndim != 2
        or mixture.ndim != 2
        or estimate.shape[0] < 2
        or mixture.shape[0] < 2
    ):
        return estimate

    total = min(estimate.shape[1], mixture.shape[1])
    if total <= 0 or sample_rate <= 0:
        return estimate

    cleaned = np.array(estimate[:, :total], dtype=np.float32, copy=True)
    source_mix = np.asarray(mixture[:, :total], dtype=np.float32)
    frame = max(1, int(round(_SATURATION_FRAME_SECONDS * sample_rate)))
    hop = max(1, int(round(_SATURATION_HOP_SECONDS * sample_rate)))
    fade = max(1, int(round(_SATURATION_FADE_SECONDS * sample_rate)))
    affected: list[tuple[int, float, float]] = []

    for channel in range(2):
        other = 1 - channel
        mask = np.zeros(total, dtype=bool)
        for start in range(0, max(1, total - frame + 1), hop):
            end = min(total, start + frame)
            estimated = cleaned[channel, start:end]
            estimated_other = cleaned[other, start:end]
            source = source_mix[channel, start:end]
            if estimated.size == 0:
                continue
            estimated_rms = float(
                np.sqrt(np.mean(np.square(estimated), dtype=np.float64) + 1e-12)
            )
            other_rms = float(
                np.sqrt(np.mean(np.square(estimated_other), dtype=np.float64) + 1e-12)
            )
            source_rms = float(
                np.sqrt(np.mean(np.square(source), dtype=np.float64) + 1e-12)
            )
            estimated_peak = float(np.max(np.abs(estimated)))
            if (
                estimated_peak >= 0.98
                and estimated_rms >= 0.80
                and other_rms <= 0.02
                and estimated_rms >= max(0.35, source_rms * 2.5)
            ):
                mask[start:end] = True

        if not np.any(mask):
            continue

        padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
        edges = np.diff(padded)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for start, end in zip(starts, ends):
            if end - start < max(frame, int(round(0.5 * sample_rate))):
                mask[start:end] = False

        padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
        edges = np.diff(padded)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for start, end in zip(starts, ends):
            affected.append(
                (channel, float(start) / sample_rate, float(end) / sample_rate)
            )
            left = max(0, start - fade)
            right = min(total, end + fade)
            gain = np.ones(right - left, dtype=np.float32)
            inner_start = start - left
            inner_end = end - left
            gain[inner_start:inner_end] = 0.0
            if inner_start > 0:
                gain[:inner_start] = np.linspace(
                    1.0,
                    0.0,
                    inner_start,
                    endpoint=False,
                    dtype=np.float32,
                )
            if inner_end < gain.size:
                gain[inner_end:] = np.linspace(
                    0.0,
                    1.0,
                    gain.size - inner_end,
                    endpoint=True,
                    dtype=np.float32,
                )
            cleaned[channel, left:right] *= gain

    if affected:
        log.warning(
            "已抑制 PolarFormer 单声道持续饱和异常: "
            + str(
                [
                    {
                        "channel": channel,
                        "start": round(start, 3),
                        "end": round(end, 3),
                    }
                    for channel, start, end in affected
                ]
            )
        )

    result = np.array(estimate, dtype=np.float32, copy=True)
    result[:, :total] = cleaned
    return result


def _install_custom_audio_separator_models(separator: Separator) -> None:
    if getattr(separator, "_ai_rvc_custom_models_installed", False):
        return

    original_download_model_files = separator.download_model_files

    def download_model_files(model_filename):
        model_spec = _CUSTOM_AUDIO_SEPARATOR_MODELS.get(str(model_filename))
        if model_spec is None:
            return original_download_model_files(model_filename)

        model_path = _download_hf_file(
            model_spec["repo_id"],
            model_spec["model_filename"],
            separator.model_file_dir,
        )
        config_path = _download_hf_file(
            model_spec["repo_id"],
            model_spec["config_filename"],
            separator.model_file_dir,
        )
        runtime_config_filename = model_spec.get("runtime_config_filename")
        if runtime_config_filename:
            import yaml

            runtime_config_path = Path(separator.model_file_dir) / runtime_config_filename
            runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "r", encoding="utf-8") as config_handle:
                runtime_config = yaml.load(config_handle, Loader=yaml.FullLoader)
            if not isinstance(runtime_config, dict):
                raise RuntimeError(f"Leap XE 配置格式无效: {config_path}")
            runtime_config["model_type"] = "bs_roformer"
            with open(runtime_config_path, "w", encoding="utf-8") as config_handle:
                yaml.dump(
                    runtime_config,
                    config_handle,
                    Dumper=yaml.Dumper,
                    allow_unicode=True,
                    sort_keys=False,
                )
            config_path = str(runtime_config_path)
        return (
            str(model_filename),
            "MDXC",
            model_spec["friendly_name"],
            model_path,
            config_path,
        )

    separator.download_model_files = download_model_files
    separator._ai_rvc_custom_models_installed = True


def _load_audio_separator_model(
    *,
    model_spec: ModelSpec,
    output_dir: str,
    model_dir: str,
) -> Separator:
    preset_name = _parse_ensemble_preset(model_spec)
    custom_preset = _CUSTOM_ENSEMBLE_PRESETS.get(preset_name or "")
    separator_kwargs = {
        "log_level": _logging.WARNING,
        "output_dir": output_dir,
        "model_file_dir": model_dir,
    }
    if preset_name and custom_preset is None:
        separator_kwargs["ensemble_preset"] = preset_name

    separator = Separator(**separator_kwargs)
    _install_custom_audio_separator_models(separator)
    if custom_preset is not None:
        separator.ensemble_preset = preset_name
        separator.ensemble_algorithm = custom_preset["algorithm"]
        separator.ensemble_weights = custom_preset.get("weights")
        separator.load_model(list(custom_preset["models"]))
    elif preset_name:
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
        if file_path.exists():
            resolved_files.append(str(file_path))
            continue

        role = _classify_common_stem_role(file_path.name)
        if role:
            candidates = [
                candidate
                for candidate in output_dir.glob("*.wav")
                if _classify_common_stem_role(candidate.name) == role
            ]
            if len(candidates) == 1:
                resolved_files.append(str(candidates[0]))
                continue

        resolved_files.append(str(file_path))
    return resolved_files


def _classify_common_stem_role(file_name: str) -> Optional[str]:
    lower_name = file_name.lower()
    if any(marker in lower_name for marker in ("(noreverb)", "(no_reverb)", "(no reverb)", "(dry)")):
        return "dry"
    if any(marker in lower_name for marker in ("(reverb)", "(echo)", "(wet)")):
        return "wet"
    if any(marker in lower_name for marker in ("(instrumental)", "(other)", "(backing)")):
        return "backing"
    if any(marker in lower_name for marker in ("(vocals)", "(lead)", "(main_vocal)", "(main vocals)")):
        return "lead"
    return None


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


class _BSPolarFormerRuntime:
    def __init__(self, model_dir: str, output_dir: str, device: str):
        self.model_dir = Path(model_dir) / "bs_polarformer"
        self.output_dir = output_dir
        self.device = str(device).lower()
        self._session = None
        self._config = None
        self._onnx_path = None

    def _ensure_assets(self) -> tuple[Path, Path]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = Path(
            _download_hf_file(
                BS_POLARFORMER_HF_REPO,
                BS_POLARFORMER_ONNX_FILENAME,
                str(self.model_dir),
            )
        )
        config_path = Path(
            _download_hf_file(
                BS_POLARFORMER_HF_REPO,
                BS_POLARFORMER_CONFIG_FILENAME,
                str(self.model_dir),
            )
        )
        return onnx_path, config_path

    @staticmethod
    def _select_onnx_providers(ort_module, device: str) -> list:
        available = ort_module.get_available_providers()
        if device.startswith("cuda"):
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "已选择 CUDA，但 onnxruntime 未提供 CUDAExecutionProvider；"
                    "请安装支持 CUDA 的 ONNX Runtime，或显式把设备改为 CPU"
                )
            device_id = 0
            if device.startswith("cuda:"):
                raw_device_id = device.split(":", 1)[1]
                if raw_device_id.isdigit():
                    device_id = int(raw_device_id)
            return [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]

        if device.startswith("dml") or device.startswith("privateuseone"):
            if "DmlExecutionProvider" not in available:
                raise RuntimeError(
                    "已选择 DirectML，但 onnxruntime 未提供 DmlExecutionProvider；"
                    "请安装支持 DirectML 的 ONNX Runtime，或显式把设备改为 CPU"
                )
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

        if device != "cpu":
            raise RuntimeError(
                "BS PolarFormer ONNX 当前只支持 CPU、CUDA 或 DirectML provider；"
                f"当前设备为 {device}"
            )
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("onnxruntime 未提供 CPUExecutionProvider")
        return ["CPUExecutionProvider"]

    def load_model(self, output_dir: str = ""):
        import onnxruntime as ort
        import yaml

        if output_dir:
            self.output_dir = output_dir

        onnx_path, config_path = self._ensure_assets()
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.full_load(handle)

        providers = self._select_onnx_providers(ort, self.device)
        log.info(
            f"正在加载 {BS_POLARFORMER_DISPLAY_NAME}: "
            f"{onnx_path.name}, providers={providers}"
        )
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        active_providers = self._session.get_providers()
        if self.device.startswith("cuda") and (
            not active_providers or active_providers[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError(
                "BS PolarFormer 请求 CUDA，但 ONNX Runtime 未启用 CUDAExecutionProvider；"
                f"active_providers={active_providers}"
            )
        log.info(f"BS PolarFormer 实际 ONNX providers: {active_providers}")
        self._config = config
        self._onnx_path = onnx_path

    @staticmethod
    def _prepare_stft(audio, stft_kwargs, stft_win_length):
        import torch
        from einops import rearrange, pack, unpack

        audio_t = torch.from_numpy(audio).float().unsqueeze(0)
        raw_audio, packed_shape = pack([audio_t], "* t")
        stft_window = torch.hann_window(stft_win_length)
        stft_repr = torch.stft(
            raw_audio,
            **stft_kwargs,
            window=stft_window,
            return_complex=True,
        )
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack(stft_repr, packed_shape, "* f t c")[0]
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")
        features = rearrange(stft_repr, "b f t c -> b t (f c)")
        return features, stft_repr, stft_window, raw_audio

    @staticmethod
    def _reconstruct_audio(
        stft_repr,
        mask,
        stft_kwargs,
        stft_window,
        audio_channels,
        raw_audio_len,
    ):
        import torch
        from einops import rearrange

        mask = torch.from_numpy(mask)
        stft_repr = stft_repr.unsqueeze(1)
        stft_complex = torch.view_as_complex(stft_repr.contiguous())
        mask_complex = torch.view_as_complex(mask.contiguous())
        masked = stft_complex * mask_complex
        masked = rearrange(masked, "b n (f s) t -> (b n s) f t", s=audio_channels)
        masked = masked.index_fill(1, torch.tensor(0), 0.0)
        reconstructed = torch.istft(
            masked,
            **stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=raw_audio_len,
        )
        return rearrange(reconstructed, "(b n s) t -> b n s t", s=audio_channels, n=1)

    def separate(self, audio_path: str) -> list[str]:
        import librosa

        if self._session is None or self._config is None:
            self.load_model(self.output_dir)

        config = self._config
        sample_rate = int(config["audio"]["sample_rate"])
        stft_kwargs = {
            "n_fft": int(config["model"]["stft_n_fft"]),
            "hop_length": int(config["model"]["stft_hop_length"]),
            "win_length": int(config["model"]["stft_win_length"]),
            "normalized": bool(config["model"].get("stft_normalized", False)),
        }
        audio_channels = 2 if config["model"].get("stereo", False) else 1
        chunk_size = _resolve_polarformer_chunk_size(
            int(config["inference"].get("chunk_size", 882000))
        )
        num_overlap = int(config["inference"].get("num_overlap", 2))
        batch_size = int(config["inference"].get("batch_size", 4))
        if chunk_size <= 0 or num_overlap <= 0 or batch_size <= 0:
            raise ValueError("BS PolarFormer 配置中的 chunk_size/num_overlap/batch_size 必须为正数")

        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
        input_was_mono = audio.ndim == 1
        if input_was_mono:
            audio = np.stack([audio, audio])
        if audio.shape[0] > 2:
            audio = audio[:2]

        total_samples = int(audio.shape[1])
        if total_samples <= 0:
            raise ValueError(f"BS PolarFormer 收到空音频: {audio_path}")

        step = max(1, chunk_size // num_overlap)
        result = np.zeros((2, total_samples), dtype=np.float32)
        count = np.zeros(total_samples, dtype=np.float32)

        chunks = []
        positions = []
        for start in range(0, total_samples, step):
            end = min(start + chunk_size, total_samples)
            chunk = audio[:, start:end]
            if chunk.shape[1] < chunk_size:
                pad = np.zeros((2, chunk_size - chunk.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=1)
            chunks.append(chunk.astype(np.float32, copy=False))
            positions.append((start, end))

        input_name = self._session.get_inputs()[0].name
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            batch_positions = positions[batch_start:batch_start + batch_size]
            for batch_offset, (chunk, (start, end)) in enumerate(
                zip(batch_chunks, batch_positions)
            ):
                chunk_index = batch_start + batch_offset + 1
                log.info(
                    f"BS PolarFormer 正在处理分块 {chunk_index}/{len(positions)}"
                )
                features, stft_repr, stft_window, raw_audio = self._prepare_stft(
                    chunk,
                    stft_kwargs,
                    stft_kwargs["win_length"],
                )
                mask = self._session.run(None, {input_name: features.numpy()})[0]
                reconstructed = self._reconstruct_audio(
                    stft_repr,
                    mask,
                    stft_kwargs,
                    stft_window,
                    audio_channels,
                    raw_audio.shape[-1],
                )
                reconstructed = reconstructed[0, 0].numpy()
                actual_len = end - start
                result[:, start:end] += reconstructed[:, :actual_len]
                count[start:end] += 1.0

        result = result / np.maximum(count, 1.0)[np.newaxis, :]
        result = _suppress_isolated_channel_saturation(
            result,
            audio[:, :total_samples],
            sample_rate,
        )
        instrumental = audio[:, :total_samples] - result

        if input_was_mono:
            result = result[:1]
            instrumental = instrumental[:1]

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(audio_path).stem
        vocals_path = output_path / f"{base_name}_(Vocals)_bs_polarformer.wav"
        instrumental_path = output_path / f"{base_name}_(Instrumental)_bs_polarformer.wav"
        sf.write(vocals_path, result.T, sample_rate)
        sf.write(instrumental_path, instrumental.T, sample_rate)
        return [str(vocals_path), str(instrumental_path)]


def _select_separator_output_by_role(
    output_files,
    output_dir: Path,
    role: str,
    label: str,
) -> str:
    resolved_files = _resolve_output_files(output_files, output_dir)
    matches = [
        file_path
        for file_path in resolved_files
        if Path(file_path).exists()
        and _classify_common_stem_role(Path(file_path).name) == role
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"混合 SOTA 分离未找到唯一的 {label} 输出；"
            f"role={role}, files={resolved_files}, output_dir={output_dir}"
        )
    return matches[0]


def _get_leap_xe_min_duration_seconds(model_dir: str) -> float:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 Leap XE 配置需要安装 PyYAML") from exc

    config_path = Path(model_dir) / LEAP_XE_VOCALS_RUNTIME_CONFIG_FILENAME
    if not config_path.exists():
        config_path = Path(model_dir) / LEAP_XE_VOCALS_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Leap XE 配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_cfg = config.get("model", {})
    audio_cfg = config.get("audio", {})
    inference_cfg = config.get("inference", {})
    stft_hop_length = int(model_cfg["stft_hop_length"])
    dim_t = int(inference_cfg["dim_t"])
    sample_rate = int(audio_cfg.get("sample_rate", 44100))
    chunk_size = stft_hop_length * (dim_t - 1)
    return chunk_size / float(sample_rate)


def _get_karaoke_min_duration_seconds(
    model_dir: str,
    model_spec: ModelSpec,
) -> float:
    """Return the longest fixed input window in a custom Karaoke ensemble."""
    preset_name = _parse_ensemble_preset(model_spec)
    preset = _CUSTOM_ENSEMBLE_PRESETS.get(preset_name or "")
    if preset is None:
        return 0.0

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 Karaoke 配置需要安装 PyYAML") from exc

    durations = []
    for model_name in preset["models"]:
        custom_model = _CUSTOM_AUDIO_SEPARATOR_MODELS.get(str(model_name))
        if custom_model is None:
            raise KeyError(f"Karaoke ensemble 模型缺少下载配置: {model_name}")

        config_path = Path(model_dir) / custom_model["config_filename"]
        if not config_path.exists():
            config_path = Path(
                _download_hf_file(
                    custom_model["repo_id"],
                    custom_model["config_filename"],
                    model_dir,
                )
            )
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)

        audio_config = config.get("audio", {})
        model_config = config.get("model", {})
        inference_config = config.get("inference", {})
        stft_hop_length = int(model_config["stft_hop_length"])
        dim_t = int(inference_config["dim_t"])
        sample_rate = int(audio_config["sample_rate"])
        if stft_hop_length <= 0 or dim_t <= 1 or sample_rate <= 0:
            raise ValueError(
                "Karaoke 配置中的 model.stft_hop_length、inference.dim_t "
                "和 audio.sample_rate 无效: "
                f"{config_path}"
            )
        chunk_size = stft_hop_length * (dim_t - 1)
        durations.append(chunk_size / float(sample_rate))

    if not durations:
        raise ValueError(f"Karaoke ensemble 没有模型: {preset_name}")
    return max(durations)


def _pad_audio_to_min_duration(
    audio_path: str,
    output_dir: Path,
    min_duration_seconds: float,
) -> tuple[str, Optional[float]]:
    try:
        info = sf.info(audio_path)
    except Exception as exc:
        raise RuntimeError(f"无法读取待分离音频信息: {audio_path}") from exc

    if info.samplerate <= 0:
        raise RuntimeError(f"音频采样率无效: {audio_path}")

    original_duration = info.frames / float(info.samplerate)
    if original_duration >= min_duration_seconds:
        return audio_path, original_duration

    data, sample_rate = sf.read(audio_path, always_2d=True)
    target_frames = int(math.ceil(min_duration_seconds * sample_rate)) + 1
    pad_frames = target_frames - data.shape[0]
    if pad_frames <= 0:
        return audio_path, original_duration

    padded = np.pad(data, ((0, pad_frames), (0, 0)), mode="constant")
    pad_dir = output_dir / "_hybrid_padded_inputs"
    pad_dir.mkdir(parents=True, exist_ok=True)
    padded_path = pad_dir / f"{Path(audio_path).stem}_leap_xe_padded.wav"
    sf.write(padded_path, padded, sample_rate)
    return str(padded_path), original_duration


def _trim_audio_to_duration(audio_path: str, duration_seconds: Optional[float]) -> None:
    if duration_seconds is None:
        return

    info = sf.info(audio_path)
    target_frames = int(round(duration_seconds * info.samplerate))
    if target_frames < 0:
        raise RuntimeError(f"目标音频时长无效: {duration_seconds}")
    if info.frames < target_frames:
        raise RuntimeError(
            f"分离输出短于原始音频，无法裁剪: output={audio_path}, "
            f"output_frames={info.frames}, target_frames={target_frames}"
        )
    if info.frames == target_frames:
        return

    data, sample_rate = sf.read(audio_path, always_2d=True)
    sf.write(audio_path, data[:target_frames], sample_rate)


class _HybridLeapXePolarFormerRuntime:
    """Use Leap XE for vocals and BS PolarFormer public ONNX for pure accompaniment."""

    def __init__(self, model_dir: str, output_dir: str, device: str):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = device
        self.vocals_separator = None
        self.instrumental_separator = None
        self._init_output_dir = None

    def load_model(self, output_dir: str = ""):
        if output_dir:
            self.output_dir = output_dir
        parent_dir = Path(self.output_dir)
        leap_dir = parent_dir / "_hybrid_leap_xe_vocals"
        polarformer_dir = parent_dir / "_hybrid_polarformer_instrumental"
        leap_dir.mkdir(parents=True, exist_ok=True)
        polarformer_dir.mkdir(parents=True, exist_ok=True)

        if self._init_output_dir == str(parent_dir):
            return

        log.info(
            "已准备人声/纯伴奏分离链路: "
            f"{LEAP_XE_VOCALS_DISPLAY_NAME} + {BS_POLARFORMER_DISPLAY_NAME}"
        )
        self._init_output_dir = str(parent_dir)

    def separate(self, audio_path: str) -> list[str]:
        if self._init_output_dir != str(Path(self.output_dir)):
            self.load_model(self.output_dir)

        parent_dir = Path(self.output_dir)
        leap_dir = parent_dir / "_hybrid_leap_xe_vocals"
        polarformer_dir = parent_dir / "_hybrid_polarformer_instrumental"
        leap_dir.mkdir(parents=True, exist_ok=True)
        polarformer_dir.mkdir(parents=True, exist_ok=True)
        separator_audio_path = _ensure_separator_pcm_wav(
            audio_path,
            parent_dir / "_input",
        )

        self.vocals_separator = _load_audio_separator_model(
            model_spec=LEAP_XE_VOCALS_MODEL,
            output_dir=str(leap_dir),
            model_dir=self.model_dir,
        )
        try:
            leap_input_path, original_duration = _pad_audio_to_min_duration(
                separator_audio_path,
                parent_dir,
                _get_leap_xe_min_duration_seconds(self.model_dir),
            )
            leap_outputs = self.vocals_separator.separate(leap_input_path)
            leap_vocals = _select_separator_output_by_role(
                leap_outputs,
                leap_dir,
                "lead",
                "Leap XE vocals",
            )
        finally:
            self.vocals_separator = None
            gc.collect()
            empty_device_cache()

        self.instrumental_separator = _BSPolarFormerRuntime(
            model_dir=self.model_dir,
            output_dir=str(polarformer_dir),
            device=self.device,
        )
        try:
            self.instrumental_separator.load_model(output_dir=str(polarformer_dir))
            polarformer_outputs = self.instrumental_separator.separate(
                separator_audio_path
            )
            polarformer_instrumental = _select_separator_output_by_role(
                polarformer_outputs,
                polarformer_dir,
                "backing",
                "BS PolarFormer instrumental",
            )
        finally:
            self.instrumental_separator = None
            gc.collect()
            empty_device_cache()

        parent_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(audio_path).stem
        vocals_path = str(parent_dir / f"{base_name}_(Vocals)_leap_xe90.wav")
        instrumental_path = str(
            parent_dir / f"{base_name}_(Instrumental)_polarformer62.wav"
        )
        _safe_move(leap_vocals, vocals_path)
        _trim_audio_to_duration(vocals_path, original_duration)
        _safe_move(polarformer_instrumental, instrumental_path)
        return [vocals_path, instrumental_path]


class RoformerSeparator:
    """人声/伴奏分离器；默认输出 Leap XE 人声与 PolarFormer 纯伴奏。"""

    def __init__(
        self,
        model_filename: ModelSpec = ROFORMER_DEFAULT_MODEL,
        device: str = "cuda",
    ):
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError(_audio_separator_install_message())
        self.model_filename = model_filename
        self.model_candidates = [model_filename]
        self.device = str(get_device(device))
        self.separator = None
        self.active_model = None

    def load_model(self, output_dir: str = ""):
        """加载指定人声/伴奏分离模型；严格高质量模式下不自动降级。"""
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
        stem_pair_label = (
            "人声/纯伴奏"
            if (
                _is_hybrid_leap_xe_polarformer_model_spec(model_name)
                or _is_bs_polarformer_model_spec(model_name)
            )
            else "人声/伴奏"
        )
        log.info(
            f"正在加载高质量{stem_pair_label}分离模型: "
            f"{_model_spec_label(model_name)}"
        )
        if _is_hybrid_leap_xe_polarformer_model_spec(model_name):
            separator = _HybridLeapXePolarFormerRuntime(
                model_dir=model_dir,
                output_dir=target_dir,
                device=self.device,
            )
            separator.load_model(output_dir=target_dir)
        elif _is_bs_polarformer_model_spec(model_name):
            separator = _BSPolarFormerRuntime(
                model_dir=model_dir,
                output_dir=target_dir,
                device=self.device,
            )
            separator.load_model(output_dir=target_dir)
        else:
            separator = _load_audio_separator_model(
                model_spec=model_name,
                output_dir=target_dir,
                model_dir=model_dir,
            )
        self.separator = separator
        self._init_output_dir = target_dir
        self.active_model = model_name
        log.info(
            f"{stem_pair_label}分离模型已加载: "
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

        stem_pair_label = (
            "人声/纯伴奏"
            if (
                _is_hybrid_leap_xe_polarformer_model_spec(self.model_filename)
                or _is_bs_polarformer_model_spec(self.model_filename)
            )
            else "人声/伴奏"
        )

        if progress_callback:
            progress_callback(f"正在加载{stem_pair_label}分离模型...", 0.1)

        if progress_callback:
            progress_callback(f"正在分离{stem_pair_label.replace('/', '和')}...", 0.3)

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
            progress_callback(f"{stem_pair_label}分离完成", 1.0)

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
    """卡拉OK分离器；默认 MVSep 9205 从原曲输出主唱与带和声伴奏。"""

    def __init__(
        self,
        model_filename: ModelSpec = KARAOKE_DEFAULT_MODEL,
        device: str = "cuda",
    ):
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError(_audio_separator_install_message())
        self.device = str(get_device(device))
        self.separator = None
        self.active_model = None
        self.model_filename = model_filename
        self.model_candidates = [model_filename]

    def load_model(self, output_dir: str = ""):
        """加载指定 Karaoke 模型；严格高质量模式下不自动降级。"""
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
            "正在加载高质量 Karaoke 模型: "
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
        分离主唱与卡拉OK第二路。

        默认 MVSep 9205 的第二路是带和声伴奏；旧模型的第二路可以是纯和声。

        Returns:
            Tuple[lead_vocals_path, second_stem_path]
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        is_mvsep_9205 = (
            _parse_ensemble_preset(self.model_filename)
            == KARAOKE_MVSEP_9205_PRESET
        )
        second_stem_label = "带和声伴奏" if is_mvsep_9205 else "第二路"

        self.load_model(output_dir=str(output_path))
        self.separator.output_dir = str(output_path)
        model_dir = str(Path(__file__).parent.parent / "assets" / "separator_models")
        min_duration_seconds = _get_karaoke_min_duration_seconds(
            model_dir,
            self.model_filename,
        )
        separation_input_path = audio_path
        original_duration = None
        if min_duration_seconds > 0:
            separation_input_path, original_duration = _pad_audio_to_min_duration(
                audio_path,
                output_path,
                min_duration_seconds,
            )

        try:
            output_files = self.separator.separate(separation_input_path)
        except FileNotFoundError as exc:
            if is_mvsep_9205:
                raise RuntimeError(
                    "MVSep 9205 AVG 未产生完整的主唱/带和声伴奏 stem；"
                    f"输入可能没有可用主唱，或某个模型输出缺失: {exc}"
                ) from exc
            raise

        resolved_files = _resolve_output_files(output_files, output_path)
        log.detail(
            f"Karaoke分离器输出文件: {[Path(file_path).name for file_path in resolved_files]}"
        )

        classified_paths: dict[str, list[str]] = {"lead": [], "backing": []}
        unknown_paths = []
        for file_path in resolved_files:
            stem_role = self._classify_stem(Path(file_path).name)
            log.detail(
                f"  {Path(file_path).name} -> 分类为: {stem_role or 'unknown'}"
            )
            if stem_role in classified_paths:
                classified_paths[stem_role].append(file_path)
            else:
                unknown_paths.append(file_path)

        if (
            unknown_paths
            or len(classified_paths["lead"]) != 1
            or len(classified_paths["backing"]) != 1
        ):
            raise RuntimeError(
                "Karaoke 输出无法唯一确定主唱和第二路，停止处理；"
                f"lead={[Path(p).name for p in classified_paths['lead']]}, "
                f"backing={[Path(p).name for p in classified_paths['backing']]}, "
                f"unknown={[Path(p).name for p in unknown_paths]}"
            )

        lead_vocals_path = classified_paths["lead"][0]
        second_stem_path = classified_paths["backing"][0]
        if not Path(lead_vocals_path).exists() or not Path(second_stem_path).exists():
            raise FileNotFoundError(
                "Karaoke 已分类输出文件不存在: "
                f"lead={lead_vocals_path}, second={second_stem_path}"
            )

        lead_rms, lead_peak, lead_nonzero = _get_audio_activity_stats(lead_vocals_path)
        second_rms, second_peak, second_nonzero = _get_audio_activity_stats(second_stem_path)
        log.detail(
            "Karaoke输出能量检测: "
            f"lead_rms={lead_rms:.6f}, lead_peak={lead_peak:.6f}, lead_nonzero={lead_nonzero}; "
            f"second_stem={second_stem_label}, second_rms={second_rms:.6f}, "
            f"second_peak={second_peak:.6f}, second_nonzero={second_nonzero}"
        )

        lead_is_nearly_silent = lead_nonzero == 0 or (lead_rms < 1e-5 and lead_peak < 1e-4)
        second_has_content = second_nonzero > 0 and (second_rms >= 5e-5 or second_peak >= 5e-4)
        if lead_is_nearly_silent and second_has_content:
            raise RuntimeError(
                "Karaoke 主唱轨几乎静音且第二路有内容，输出疑似反转；"
                "为避免错误转换伴奏，已停止处理"
            )

        final_lead = str(output_path / "lead_vocals.wav")
        final_second_stem = str(
            output_path
            / ("accompaniment.wav" if is_mvsep_9205 else "backing_vocals.wav")
        )
        _safe_move(lead_vocals_path, final_lead)
        _safe_move(second_stem_path, final_second_stem)
        _trim_audio_to_duration(final_lead, original_duration)
        _trim_audio_to_duration(final_second_stem, original_duration)
        log.detail(
            "Karaoke输出语义: "
            f"主唱={Path(final_lead).name}; "
            f"{second_stem_label}={Path(final_second_stem).name}"
        )

        return final_lead, final_second_stem

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
            raise ImportError(_audio_separator_install_message())
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
    """检查 audio-separator 是否可用，供 BS-RoFormer/Karaoke/De-Reverb 路线使用"""
    return AUDIO_SEPARATOR_AVAILABLE


def get_available_models() -> list:
    """获取可用的分离模型列表"""
    models = []
    if AUDIO_SEPARATOR_AVAILABLE:
        models.append({
            "name": "roformer",
            "description": "默认混合链路: Leap XE 90 vocals + BS PolarFormer public ONNX 62 pure accompaniment；RoFormer/BS-RoFormer 模型也用于 Karaoke、De-Reverb 和手动预设"
        })
    if DEMUCS_AVAILABLE:
        models.extend([
            {"name": "htdemucs", "description": "Demucs 默认模型，平衡质量和速度 (SDR ~9dB)"},
            {"name": "htdemucs_ft", "description": "Demucs 微调版本，质量更高但更慢"},
            {"name": "mdx_extra", "description": "MDX 模型，适合某些音乐类型"},
        ])
    return models
