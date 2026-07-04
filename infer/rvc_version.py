# -*- coding: utf-8 -*-
"""RVC checkpoint version detection helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_MISSING = object()


@dataclass(frozen=True)
class RVCVersionInfo:
    version: str
    source: str
    raw_version: object
    raw_version_present: bool
    feature_dim: int | None
    metadata_mismatch: bool

    @property
    def raw_version_label(self) -> str:
        return repr(self.raw_version) if self.raw_version_present else "<missing>"


def _normalize_version(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"v1", "v2"}:
            return normalized
    return None


def _feature_dim_from_checkpoint(cpt: Mapping[str, Any]) -> int | None:
    weight = cpt.get("weight")
    if not isinstance(weight, Mapping):
        return None

    emb = weight.get("enc_p.emb_phone.weight")
    shape = getattr(emb, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def inspect_rvc_model_version(
    cpt: Mapping[str, Any],
    model_label: str = "RVC model",
) -> RVCVersionInfo:
    raw_version_present = "version" in cpt
    raw_version = cpt.get("version", _MISSING)
    metadata_version = _normalize_version(raw_version)

    feature_dim = _feature_dim_from_checkpoint(cpt)
    shape_version = None
    if feature_dim == 256:
        shape_version = "v1"
    elif feature_dim == 768:
        shape_version = "v2"
    elif feature_dim is not None:
        raise ValueError(
            f"无法判断 {model_label} 的 RVC 版本："
            f"enc_p.emb_phone.weight 第二维是 {feature_dim}，预期为 256(v1) 或 768(v2)。"
        )

    if shape_version is not None:
        metadata_mismatch = (
            metadata_version is not None and metadata_version != shape_version
        )
        source = "metadata+weight_shape" if metadata_version == shape_version else "weight_shape"
        return RVCVersionInfo(
            version=shape_version,
            source=source,
            raw_version=raw_version,
            raw_version_present=raw_version_present,
            feature_dim=feature_dim,
            metadata_mismatch=metadata_mismatch,
        )

    if metadata_version is not None:
        return RVCVersionInfo(
            version=metadata_version,
            source="metadata",
            raw_version=raw_version,
            raw_version_present=raw_version_present,
            feature_dim=None,
            metadata_mismatch=False,
        )

    raw_label = repr(raw_version) if raw_version_present else "<missing>"
    raise ValueError(
        f"无法判断 {model_label} 的 RVC 版本：version={raw_label}，"
        "且缺少可识别的 enc_p.emb_phone.weight 权重形状。"
    )


def infer_rvc_model_version(
    cpt: Mapping[str, Any],
    model_label: str = "RVC model",
) -> str:
    return inspect_rvc_model_version(cpt, model_label).version
