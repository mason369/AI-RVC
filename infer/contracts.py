"""运行时合同：拒绝无效参数、未知模型结构及失效索引。"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from infer.rvc_version import inspect_rvc_model_version


def number(value, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} 必须是数值")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum}～{maximum} 之间且为有限数值")
    return value


def integer(value, name: str, minimum: int, maximum: int) -> int:
    parsed = number(value, name, minimum, maximum)
    if not parsed.is_integer():
        raise ValueError(f"{name} 必须是整数，不会截断小数")
    return int(parsed)


@dataclass(frozen=True)
class ModelContract:
    version: str
    feature_dim: int
    sample_rate: int
    speaker_count: int
    uses_f0: bool
    config: list


def inspect_checkpoint(checkpoint: Mapping, label: str = "RVC 模型") -> ModelContract:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{label} 不是 RVC 推理权重字典")
    version = inspect_rvc_model_version(checkpoint, label)
    weights = checkpoint.get("weight")
    if not isinstance(weights, Mapping):
        raise ValueError(f"{label} 缺少 weight 推理权重")
    phone = weights.get("enc_p.emb_phone.weight")
    speaker = weights.get("emb_g.weight")
    if not isinstance(phone, torch.Tensor) or phone.ndim != 2:
        raise ValueError(f"{label} 缺少二维 enc_p.emb_phone.weight")
    if not isinstance(speaker, torch.Tensor) or speaker.ndim != 2 or speaker.shape[0] < 1:
        raise ValueError(f"{label} 缺少有效的二维 emb_g.weight")
    config = checkpoint.get("config")
    if not isinstance(config, (list, tuple)) or len(config) != 18:
        raise ValueError(f"{label} 必须包含标准 RVC 的 18 项模型配置，不会猜测架构")
    config = list(config)
    sample_rate = config[-1]
    if isinstance(sample_rate, str):
        sample_rate = {"32k": 32000, "40k": 40000, "48k": 48000}.get(sample_rate)
    if sample_rate not in (32000, 40000, 48000):
        raise ValueError(f"{label} 不支持采样率 {config[-1]!r}；支持原生 32/40/48 kHz")
    f0 = checkpoint.get("f0", 1)
    if type(f0) not in (bool, int) or f0 not in (0, 1):
        raise ValueError(f"{label} 的 f0 标志必须为 0 或 1")
    config[-3] = int(speaker.shape[0])
    config[-1] = int(sample_rate)
    return ModelContract(version.version, int(phone.shape[1]), int(sample_rate), config[-3], bool(f0), config)


def validate_state_dict(model, weights: Mapping) -> None:
    """允许缺省训练专用 enc_q；任何推理层缺失、冗余或形状错误均失败。"""
    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(key for key in set(weights) - set(expected) if not key.startswith("enc_q."))
    bad_shapes = sorted(key for key in set(expected) & set(weights)
                        if not isinstance(weights[key], torch.Tensor) or weights[key].shape != expected[key].shape)
    if missing or unexpected or bad_shapes:
        raise ValueError(f"RVC 推理权重不完整：缺失={missing}，未知={unexpected}，形状不符={bad_shapes}")
    for key in expected:
        weight = weights[key]
        if weight.dtype not in (torch.float32, torch.float16):
            raise ValueError(f"RVC 权重类型不支持：{key}={weight.dtype}；只接受 FP32/FP16 浮点权重，不进行量化")
        if not torch.isfinite(weight).all():
            raise ValueError(f"RVC 权重包含 NaN 或 Inf：{key}")


def read_index(path, feature_dim: int, *, min_vectors: int = 1):
    import faiss
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RVC 索引不存在：{path}")
    # Faiss's native Windows fopen cannot reliably open Unicode paths.
    # Python owns path decoding; Faiss still deserializes the original bytes.
    with path.open('rb') as stream:
        index = faiss.read_index(faiss.PyCallbackIOReader(stream.read))
    if int(index.d) != int(feature_dim):
        raise ValueError(f"RVC 索引维度与模型不匹配：model_feature_dim={feature_dim}, index_dim={index.d}")
    if not index.is_trained or index.ntotal < min_vectors:
        raise ValueError(f"RVC 索引未训练或向量不足：需要至少 {min_vectors} 个，实际 {index.ntotal}")
    if type(index) not in (faiss.IndexFlatL2, faiss.IndexIVFFlat) or index.metric_type != faiss.METRIC_L2:
        raise ValueError(f"不支持 RVC 索引类型 {type(index).__name__}；需要保留完整 FP32 向量的 FlatL2/IVFFlat，不接受量化或降维索引")
    vectors = index.reconstruct_n(0, index.ntotal)
    if vectors.shape != (index.ntotal, feature_dim) or not np.isfinite(vectors).all():
        raise ValueError("RVC 索引不能重建完整的有限数值特征，不会跳过检索")
    # IVF artifacts commonly store nprobe=1, which can return -1 even when
    # ntotal >= k. Search every original FP32 vector, never pad/skip neighbors.
    # IndexFlatL2 is Faiss's exhaustive implementation; no artifact is rewritten.
    if type(index) is faiss.IndexIVFFlat:
        exact = faiss.IndexFlatL2(feature_dim)
        exact.add(np.ascontiguousarray(vectors, dtype=np.float32))
        index = exact
    return index, vectors


def retrieve_features(index, vectors, features: np.ndarray, k: int = 8) -> np.ndarray:
    """完整向量检索及逆距离加权；处理精确重复向量，拒绝无效近邻。"""
    if features.ndim != 2 or features.shape[1] != index.d or not np.isfinite(features).all():
        raise ValueError("检索特征维度不匹配或包含非有限数值")
    k = min(integer(k, "k", 1, 1024), index.ntotal)
    if k < 1:
        raise ValueError("索引没有可检索向量")
    scores, indices = index.search(np.ascontiguousarray(features, dtype=np.float32), k)
    if np.any(indices < 0) or np.any(indices >= len(vectors)) or not np.isfinite(scores).all():
        raise ValueError("FAISS 未返回足够的有效近邻；不会跳过索引或混合无效向量")
    # A zero distance is an exact match, not a division-by-zero failure.
    exact = scores <= 0
    exact_rows = exact.any(axis=1)
    weights = np.zeros(scores.shape, dtype=np.float64)
    weights[exact_rows] = exact[exact_rows]
    weights[~exact_rows] = 1.0 / np.square(scores[~exact_rows].astype(np.float64))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(vectors[indices] * weights[:, :, None], axis=1).astype(np.float32)
