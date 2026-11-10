"""Lossless, fail-closed compatibility handling for RVC inference checkpoints.

The same code is used by upload inspection and the real inference loader so a
model cannot pass a heuristic scanner and then be partially loaded by
``strict=False``.  Only tensor-only checkpoints accepted by PyTorch's
``weights_only`` loader are handled here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
from typing import Any

import torch

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.rvc_version import inspect_rvc_model_version


SUPPORTED_WEIGHT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
MAX_TENSOR_COUNT = 10_000
MAX_WEIGHT_NAME_LENGTH = 512
SUPPORTED_VOCODERS = {
    "hifi-gan": "HiFi-GAN",
    "hifigan": "HiFi-GAN",
    "mrf hifi-gan": "MRF HiFi-GAN",
    "mrf hifigan": "MRF HiFi-GAN",
    "refinegan": "RefineGAN",
}
_OLD_WEIGHT_G = ".weight_g"
_OLD_WEIGHT_V = ".weight_v"
_PARAM_WEIGHT_G = ".parametrizations.weight.original0"
_PARAM_WEIGHT_V = ".parametrizations.weight.original1"


class RvcCheckpointCompatibilityError(ValueError):
    """The file is safe to inspect but not exactly loadable by this runtime."""


@dataclass(frozen=True)
class RvcCompatibilityReport:
    rvc_version: str
    version_detection: str
    feature_dimension: int
    sample_rate_hz: int
    speaker_count: int
    configured_speaker_count: int | None
    speaker_metadata_normalized: bool
    f0_enabled: bool
    configured_f0_enabled: bool | None
    f0_metadata_normalized: bool
    vocoder: str
    configured_vocoder: str | None
    vocoder_metadata_normalized: bool
    runtime_architecture: str
    source_weight_key: str
    source_weight_norm_format: str
    weight_key_conversions: int
    removed_training_tensor_count: int
    parameter_count: int
    source_parameter_count: int
    tensor_count: int
    weight_dtypes: list[str]
    runtime_dtype: str
    runtime_precision_policy: str
    quantized: bool
    finite_weights_verified: bool
    compatibility_status: str
    inference_weight_coverage_percent: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RvcRuntimePrecisionPlan:
    """Lossless compute precision selected from checkpoint storage and hardware."""

    runtime_dtype: str
    torch_dtype: torch.dtype
    is_half: bool
    source_dtypes: tuple[str, ...]
    reason: str


def checkpoint_weight_dtypes(checkpoint: Mapping[str, Any]) -> list[str]:
    """Return inference-weight storage types without trusting metadata labels."""
    _source_key, raw_weights, _config = _root_parts(checkpoint)
    dtypes = {
        str(value.dtype).removeprefix("torch.")
        for name, value in raw_weights.items()
        if isinstance(value, torch.Tensor)
        and not str(name).removeprefix("module.").startswith("enc_q.")
    }
    return sorted(dtypes)


def checkpoint_storage_dtypes(path: Path) -> list[str]:
    """Read floating storage types from a PyTorch checkpoint without execution.

    Modern checkpoints declare tensor storage classes in the ZIP-contained
    ``data.pkl`` stream. Reading those declarations does not unpickle the file
    or materialize any tensor. Legacy checkpoints are handled by PyTorch's
    restricted weights-only loader and a bounded primitive-container walk.
    """
    marker_names = {
        b"HalfStorage": "float16",
        b"BFloat16Storage": "bfloat16",
        b"FloatStorage": "float32",
        b"DoubleStorage": "float64",
    }
    if checkpoint_container_format(path) == "pytorch_zip":
        import zipfile

        try:
            with zipfile.ZipFile(path) as archive:
                pickle_names = [
                    name
                    for name in archive.namelist()
                    if name == "data.pkl" or name.endswith("/data.pkl")
                ]
                if len(pickle_names) != 1:
                    raise RvcCheckpointCompatibilityError(
                        "PyTorch checkpoint must contain exactly one data.pkl"
                    )
                metadata = archive.read(pickle_names[0])
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise RvcCheckpointCompatibilityError(
                f"PyTorch checkpoint container is incomplete: {exc}"
            ) from exc
        dtypes = sorted({name for marker, name in marker_names.items() if marker in metadata})
        if not dtypes:
            raise RvcCheckpointCompatibilityError(
                "PyTorch checkpoint declares no floating tensor storage"
            )
        return dtypes

    try:
        root = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RvcCheckpointCompatibilityError(
            f"Legacy PyTorch checkpoint failed safe weights-only loading: {exc}"
        ) from exc
    dtypes: set[str] = set()
    pending = [root]
    visited: set[int] = set()
    visited_nodes = 0
    while pending:
        value = pending.pop()
        visited_nodes += 1
        if visited_nodes > 50_000:
            raise RvcCheckpointCompatibilityError(
                "Legacy PyTorch checkpoint contains too many metadata nodes"
            )
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                dtypes.add(str(value.dtype).removeprefix("torch."))
            continue
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(value)
    if not dtypes:
        raise RvcCheckpointCompatibilityError(
            "Legacy PyTorch checkpoint contains no floating tensors"
        )
    return sorted(dtypes)


def select_rvc_runtime_precision(
    source_dtypes: Sequence[str],
    device: str | torch.device,
    *,
    cuda_capability: tuple[int, int] | None = None,
    cuda_device_name: str | None = None,
) -> RvcRuntimePrecisionPlan:
    """Select the fastest runtime dtype that never narrows uploaded weights.

    Official exported RVC checkpoints are commonly stored as FP16. Those can
    remain FP16 on an eligible native Tensor Core CUDA device. GTX 16-series
    cards share SM 7.5 with RTX cards but require FP32 in the pinned runtime.
    FP32 stays FP32 and FP64 stays FP64. CPU/unsupported devices losslessly
    widen FP16/BF16 to FP32;
    mixed storage uses FP64 when any FP64 tensor is present and otherwise FP32.
    There is deliberately no narrowing cast in this function.
    """
    normalized = tuple(sorted(set(source_dtypes)))
    allowed = {"float16", "bfloat16", "float32", "float64"}
    if not normalized or not set(normalized).issubset(allowed):
        raise RvcCheckpointCompatibilityError(
            f"Cannot select RVC precision for storage dtypes: {list(normalized)}"
        )

    resolved_device = torch.device(device)
    half_unavailable_reason = "native FP16 execution is unavailable"
    if (normalized == ("float16",) and resolved_device.type == "cuda"
            and torch.version.hip is None):
        capability = cuda_capability
        name = cuda_device_name
        if (capability is None or name is None) and torch.cuda.is_available():
            index = resolved_device.index
            if capability is None:
                capability = torch.cuda.get_device_capability(index)
            if name is None:
                name = torch.cuda.get_device_name(index)
        # Match the pinned RVC GPU rule. SM alone cannot distinguish GTX 16xx
        # from RTX 20xx/T4; forcing half on GTX 16xx can produce NaN RMVPE F0.
        is_16_series = capability == (7, 5) and bool(re.search(r"16\d{2}", name or ""))
        if is_16_series:
            half_unavailable_reason = "the GTX 16-series RVC/RMVPE GPU rule requires FP32"
        if name and capability is not None and capability >= (7, 0) and not is_16_series:
            return RvcRuntimePrecisionPlan(
                runtime_dtype="float16",
                torch_dtype=torch.float16,
                is_half=True,
                source_dtypes=normalized,
                reason="FP16 checkpoint on a native FP16 Tensor Core CUDA device",
            )

    if "float64" in normalized:
        return RvcRuntimePrecisionPlan(
            runtime_dtype="float64",
            torch_dtype=torch.float64,
            is_half=False,
            source_dtypes=normalized,
            reason=(
                "FP64 checkpoint preserved without downcast"
                if normalized == ("float64",)
                else "mixed floating checkpoint widened to its exact common FP64 representation"
            ),
        )

    if normalized == ("float32",):
        reason = "FP32 checkpoint preserved without downcast"
    elif normalized == ("bfloat16",):
        reason = "BF16 checkpoint widened losslessly because the RVC pipeline has no BF16 execution contract"
    elif len(normalized) > 1:
        reason = "mixed floating checkpoint widened to its exact common FP32 representation"
    else:
        reason = f"FP16 checkpoint widened losslessly because {half_unavailable_reason}"
    return RvcRuntimePrecisionPlan(
        runtime_dtype="float32",
        torch_dtype=torch.float32,
        is_half=False,
        source_dtypes=normalized,
        reason=reason,
    )


def checkpoint_container_format(path: Path) -> str:
    with path.open("rb") as stream:
        return "pytorch_zip" if stream.read(4) == b"PK\x03\x04" else "pytorch_legacy"


def safe_load_rvc_checkpoint(path: Path) -> dict[str, Any]:
    """Load tensors and primitive metadata without importing checkpoint globals."""
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            # Modern PyTorch ZIP checkpoints can be validated and copied into
            # the selected runtime dtype without eagerly duplicating the whole
            # file in host RAM. Legacy pickle checkpoints do not support mmap.
            mmap=checkpoint_container_format(path) == "pytorch_zip",
        )
    except Exception as exc:
        raise RvcCheckpointCompatibilityError(
            f"RVC checkpoint failed safe weights-only loading: {exc}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise RvcCheckpointCompatibilityError("RVC checkpoint root must be a mapping")
    return checkpoint


def _configured_vocoder(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    return SUPPORTED_VOCODERS.get(" ".join(raw.strip().lower().split()))


def _detect_vocoder(keys: Sequence[str]) -> str:
    normalized = [key[7:] if key.startswith("module.") else key for key in keys]
    if any(
        key.startswith(("dec.pre_conv.", "dec.downsample_blocks.", "dec.upsample_conv_blocks."))
        for key in normalized
    ):
        return "RefineGAN"
    if any(key.startswith(("dec.upsamples.", "dec.mrfs.")) for key in normalized):
        return "MRF HiFi-GAN"
    return "HiFi-GAN"


def _configured_f0(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in {0, 1}:
        return bool(raw)
    return None


def _detect_f0(keys: Sequence[str]) -> bool:
    normalized = [key[7:] if key.startswith("module.") else key for key in keys]
    return any(key == "enc_p.emb_pitch.weight" for key in normalized)


def _weight_norm_format(keys: Sequence[str]) -> str:
    has_old = any(key.endswith((_OLD_WEIGHT_G, _OLD_WEIGHT_V)) for key in keys)
    has_param = any(key.endswith((_PARAM_WEIGHT_G, _PARAM_WEIGHT_V)) for key in keys)
    if has_old and has_param:
        return "mixed"
    if has_param:
        return "parametrized"
    if has_old:
        return "legacy"
    return "none"


def _convert_weight_norm_key(key: str, target: str) -> str:
    if target == "legacy":
        if key.endswith(_PARAM_WEIGHT_G):
            return key[: -len(_PARAM_WEIGHT_G)] + _OLD_WEIGHT_G
        if key.endswith(_PARAM_WEIGHT_V):
            return key[: -len(_PARAM_WEIGHT_V)] + _OLD_WEIGHT_V
    elif target == "parametrized":
        if key.endswith(_OLD_WEIGHT_G):
            return key[: -len(_OLD_WEIGHT_G)] + _PARAM_WEIGHT_G
        if key.endswith(_OLD_WEIGHT_V):
            return key[: -len(_OLD_WEIGHT_V)] + _PARAM_WEIGHT_V
    return key


def _canonicalize_weights(
    raw_weights: Mapping[Any, Any],
    *,
    target_weight_norm: str,
    verify_finite: bool,
) -> tuple[dict[str, torch.Tensor], str, int, int, int]:
    if not raw_weights:
        raise RvcCheckpointCompatibilityError("RVC checkpoint has no model weights")
    raw_keys = list(raw_weights.keys())
    if len(raw_keys) > MAX_TENSOR_COUNT:
        raise RvcCheckpointCompatibilityError("RVC checkpoint contains too many tensors")
    if not all(isinstance(key, str) and key for key in raw_keys):
        raise RvcCheckpointCompatibilityError("RVC weight names must be non-empty text")
    if any(len(key) > MAX_WEIGHT_NAME_LENGTH for key in raw_keys):
        raise RvcCheckpointCompatibilityError("RVC weight name is unreasonably long")
    strip_module = all(key.startswith("module.") for key in raw_keys)
    source_format = _weight_norm_format(raw_keys)
    normalized: dict[str, torch.Tensor] = {}
    conversions = 0
    removed_training = 0
    source_parameter_count = 0
    for source_name, value in raw_weights.items():
        if not isinstance(value, torch.Tensor):
            raise RvcCheckpointCompatibilityError(
                f"RVC checkpoint contains a non-tensor weight: {source_name}"
            )
        if value.is_quantized:
            raise RvcCheckpointCompatibilityError(
                f"Quantized RVC tensors are not accepted: {source_name}"
            )
        if value.device.type != "cpu":
            raise RvcCheckpointCompatibilityError(
                f"RVC tensor did not materialize on CPU: {source_name} ({value.device})"
            )
        if value.layout != torch.strided:
            raise RvcCheckpointCompatibilityError(
                f"RVC tensor layout is unsupported: {source_name} ({value.layout})"
            )
        if value.dtype not in SUPPORTED_WEIGHT_DTYPES:
            raise RvcCheckpointCompatibilityError(
                f"RVC tensor precision is unsupported: {source_name} ({value.dtype})"
            )
        if value.ndim > 8 or value.numel() <= 0:
            raise RvcCheckpointCompatibilityError(
                f"RVC tensor shape is invalid: {source_name} ({tuple(value.shape)})"
            )
        if verify_finite and not bool(torch.isfinite(value).all().item()):
            raise RvcCheckpointCompatibilityError(
                f"RVC tensor contains NaN or infinity: {source_name}"
            )
        source_parameter_count += int(value.numel())
        name = source_name[7:] if strip_module else source_name
        if name.startswith("enc_q."):
            removed_training += 1
            continue
        converted = _convert_weight_norm_key(name, target_weight_norm)
        conversions += int(converted != name)
        if converted in normalized:
            raise RvcCheckpointCompatibilityError(
                f"RVC weight names collide after lossless normalization: {converted}"
            )
        normalized[converted] = value
    return normalized, source_format, conversions, removed_training, source_parameter_count


def _official_synthesizer_class(version: str, f0_enabled: bool):
    classes = {
        ("v1", True): SynthesizerTrnMs256NSFsid,
        ("v1", False): SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): SynthesizerTrnMs768NSFsid,
        ("v2", False): SynthesizerTrnMs768NSFsid_nono,
    }
    return classes[(version, f0_enabled)]


def _instantiate_synthesizer(
    config: list[Any],
    *,
    version: str,
    f0_enabled: bool,
    vocoder: str,
    is_half: bool,
    meta: bool,
):
    context = torch.device("meta") if meta else nullcontext()
    try:
        with context:
            if vocoder == "HiFi-GAN":
                model = _official_synthesizer_class(version, f0_enabled)(
                    *config,
                    is_half=is_half,
                )
            else:
                from infer.lib.applio_algorithm.synthesizers import Synthesizer

                model = Synthesizer(
                    *config,
                    use_f0=f0_enabled,
                    text_enc_hidden_dim=768 if version == "v2" else 256,
                    vocoder=vocoder,
                    randomized=False,
                    checkpointing=False,
                )
    except Exception as exc:
        raise RvcCheckpointCompatibilityError(
            f"RVC network configuration cannot be constructed exactly: {exc}"
        ) from exc
    if getattr(model, "enc_q", None) is not None:
        del model.enc_q
    return model


def _root_parts(checkpoint: Mapping[str, Any]) -> tuple[str, Mapping[Any, Any], list[Any]]:
    source_weight_key = "weight"
    weights = checkpoint.get("weight")
    if not isinstance(weights, Mapping):
        source_weight_key = "model"
        weights = checkpoint.get("model")
    config = checkpoint.get("config")
    if not isinstance(weights, Mapping) or not isinstance(config, (list, tuple)):
        raise RvcCheckpointCompatibilityError(
            "Checkpoint is not an exported RVC inference model with weights and config"
        )
    if len(config) != 18:
        raise RvcCheckpointCompatibilityError(
            f"RVC inference config must contain exactly 18 fields, found {len(config)}"
        )
    return source_weight_key, weights, list(config)


def prepare_rvc_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model_label: str,
    validate_structure: bool,
    verify_finite: bool = True,
) -> tuple[dict[str, Any], RvcCompatibilityReport]:
    """Normalize metadata/key spelling without changing any tensor value or dtype."""
    source_weight_key, raw_weights, config = _root_parts(checkpoint)
    version_info = inspect_rvc_model_version(checkpoint, model_label)
    version = version_info.version
    feature_dimension = int(version_info.feature_dim)
    raw_weight_names = [str(key) for key in raw_weights]
    configured_f0_enabled = _configured_f0(checkpoint.get("f0"))
    f0_enabled = _detect_f0(raw_weight_names)
    f0_metadata_normalized = configured_f0_enabled != f0_enabled
    configured_vocoder = _configured_vocoder(checkpoint.get("vocoder"))
    vocoder = _detect_vocoder(raw_weight_names)
    vocoder_metadata_normalized = configured_vocoder != vocoder
    if not f0_enabled and vocoder != "HiFi-GAN":
        raise RvcCheckpointCompatibilityError(
            f"{vocoder} does not define an RVC non-F0 inference architecture"
        )
    target_weight_norm = "legacy" if vocoder == "HiFi-GAN" else "parametrized"
    (
        weights,
        source_weight_norm_format,
        weight_key_conversions,
        removed_training_tensor_count,
        source_parameter_count,
    ) = _canonicalize_weights(
        raw_weights,
        target_weight_norm=target_weight_norm,
        verify_finite=verify_finite,
    )

    speaker_weight = weights.get("emb_g.weight")
    if not isinstance(speaker_weight, torch.Tensor) or speaker_weight.ndim != 2:
        raise RvcCheckpointCompatibilityError("RVC speaker embedding is missing or invalid")
    speaker_count = int(speaker_weight.shape[0])
    if speaker_count < 1:
        raise RvcCheckpointCompatibilityError("RVC checkpoint has no speakers")
    configured_speakers = config[-3] if isinstance(config[-3], (int, float)) else None
    configured_speaker_count = int(configured_speakers) if configured_speakers is not None else None
    speaker_metadata_normalized = configured_speaker_count != speaker_count
    config[-3] = speaker_count

    sample_rate = config[-1]
    if isinstance(sample_rate, str):
        sample_rate = {"32k": 32000, "40k": 40000, "48k": 48000}.get(sample_rate)
    if not isinstance(sample_rate, (int, float)) or isinstance(sample_rate, bool):
        raise RvcCheckpointCompatibilityError("RVC sample rate is missing or invalid")
    if int(sample_rate) != sample_rate:
        raise RvcCheckpointCompatibilityError("RVC sample rate must be an integer")
    sample_rate_hz = int(sample_rate)
    if sample_rate_hz < 16000 or sample_rate_hz > 192000:
        raise RvcCheckpointCompatibilityError(
            f"RVC sample rate is outside the supported range: {sample_rate_hz}"
        )
    config[-1] = sample_rate_hz

    runtime_architecture = (
        f"official_rvc_{version}_{'f0' if f0_enabled else 'non_f0'}_hifigan"
        if vocoder == "HiFi-GAN"
        else f"applio_{version}_{'f0' if f0_enabled else 'non_f0'}_{vocoder.lower().replace(' ', '_').replace('-', '')}"
    )
    if validate_structure:
        model = _instantiate_synthesizer(
            config,
            version=version,
            f0_enabled=f0_enabled,
            vocoder=vocoder,
            is_half=False,
            meta=True,
        )
        expected = model.state_dict()
        expected_names = set(expected)
        actual_names = set(weights)
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        mismatched = sorted(
            name
            for name in expected_names & actual_names
            if tuple(expected[name].shape) != tuple(weights[name].shape)
        )
        del model
        if missing or unexpected or mismatched:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}")
            if unexpected:
                details.append(
                    f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
                )
            if mismatched:
                rendered = [
                    f"{name}:{tuple(weights[name].shape)}!={tuple(expected[name].shape)}"
                    for name in mismatched[:8]
                ]
                details.append(f"shape_mismatch={rendered}{'...' if len(mismatched) > 8 else ''}")
            raise RvcCheckpointCompatibilityError(
                "RVC checkpoint is not a complete exact match for its declared architecture: "
                + "; ".join(details)
            )

    dtypes = sorted({str(value.dtype).removeprefix("torch.") for value in weights.values()})
    parameter_count = sum(int(value.numel()) for value in weights.values())
    report = RvcCompatibilityReport(
        rvc_version=version,
        version_detection=(
            "metadata+weight_shape"
            if not version_info.metadata_mismatch and version_info.raw_version_present
            else "weight_shape"
        ),
        feature_dimension=feature_dimension,
        sample_rate_hz=sample_rate_hz,
        speaker_count=speaker_count,
        configured_speaker_count=configured_speaker_count,
        speaker_metadata_normalized=speaker_metadata_normalized,
        f0_enabled=f0_enabled,
        configured_f0_enabled=configured_f0_enabled,
        f0_metadata_normalized=f0_metadata_normalized,
        vocoder=vocoder,
        configured_vocoder=configured_vocoder,
        vocoder_metadata_normalized=vocoder_metadata_normalized,
        runtime_architecture=runtime_architecture,
        source_weight_key=source_weight_key,
        source_weight_norm_format=source_weight_norm_format,
        weight_key_conversions=weight_key_conversions,
        removed_training_tensor_count=removed_training_tensor_count,
        parameter_count=parameter_count,
        source_parameter_count=source_parameter_count,
        tensor_count=len(weights),
        weight_dtypes=dtypes,
        runtime_dtype="source_adaptive",
        runtime_precision_policy="preserve_source_or_losslessly_widen",
        quantized=False,
        finite_weights_verified=verify_finite,
        compatibility_status="verified_exact",
        inference_weight_coverage_percent=100,
    )
    prepared = dict(checkpoint)
    prepared.pop("model", None)
    prepared["weight"] = weights
    prepared["config"] = config
    prepared["version"] = version
    prepared["f0"] = int(f0_enabled)
    prepared["vocoder"] = vocoder
    return prepared, report


def build_rvc_synthesizer(
    checkpoint: Mapping[str, Any],
    *,
    model_label: str,
    is_half: bool,
    runtime_dtype: str | None = None,
):
    """Build the real network and prove every inference tensor was consumed."""
    prepared, report = prepare_rvc_checkpoint(
        checkpoint,
        model_label=model_label,
        validate_structure=False,
        verify_finite=True,
    )
    model = _instantiate_synthesizer(
        prepared["config"],
        version=report.rvc_version,
        f0_enabled=report.f0_enabled,
        vocoder=report.vocoder,
        is_half=is_half,
        meta=False,
    )
    dtype_name = runtime_dtype or ("float16" if is_half else "float32")
    try:
        runtime_torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[dtype_name]
    except KeyError as exc:
        raise RvcCheckpointCompatibilityError(
            f"Unsupported adaptive RVC runtime dtype: {dtype_name}"
        ) from exc
    # PyTorch copies checkpoint tensors into the destination parameter dtype.
    # Cast the empty network first: loading FP64 into an FP32 model and calling
    # model.double() afterwards irreversibly truncates the source checkpoint.
    model = model.to(dtype=runtime_torch_dtype)
    try:
        incompatible = model.load_state_dict(prepared["weight"], strict=True)
    except Exception as exc:
        raise RvcCheckpointCompatibilityError(
            f"RVC checkpoint did not fully populate the inference network: {exc}"
        ) from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RvcCheckpointCompatibilityError(
            "RVC checkpoint did not fully populate the inference network: "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    report = replace(
        report,
        runtime_dtype=dtype_name,
    )
    return model, prepared, report
