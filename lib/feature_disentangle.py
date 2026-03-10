# -*- coding: utf-8 -*-
"""
特征解耦模块 - 分离HuBERT特征中的音色和内容信息
基于最新研究文献的根本性解决方案
"""
import numpy as np
from scipy import signal
from typing import Tuple


def whiten_features(features: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    特征白化 - 去除音色相关的统计信息

    参考: "Real-Time Low-Latency Voice Conversion" (arXiv:2401.03078)
    白化可以去除说话人相关的统计特征，保留内容信息

    Args:
        features: 输入特征 [T, C]
        eps: 数值稳定性常数

    Returns:
        whitened_features: 白化后的特征
        mean: 均值（用于逆变换）
        std: 标准差（用于逆变换）
    """
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + eps

    whitened = (features - mean) / std

    return whitened, mean, std


def apply_target_statistics(
    features: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray
) -> np.ndarray:
    """
    应用目标说话人的统计特征

    Args:
        features: 白化后的特征
        target_mean: 目标均值
        target_std: 目标标准差

    Returns:
        转换后的特征
    """
    return features * target_std + target_mean


def extract_timbre_residual(
    features: np.ndarray,
    f0: np.ndarray,
    window_size: int = 50
) -> np.ndarray:
    """
    提取音色残差 - 识别并去除音色相关的特征分量

    参考: "Mitigating Timbre Leakage with Universal Semantic Mapping" (arXiv:2504.08524)
    音色信息通常表现为特征的低频变化

    Args:
        features: 输入特征 [T, C]
        f0: F0序列 [T*2]（用于对齐）
        window_size: 滑动窗口大小

    Returns:
        去除音色残差后的特征
    """
    T, C = features.shape

    # 计算局部均值（音色相关的低频成分）
    # 使用高斯窗口进行平滑
    window = signal.windows.gaussian(window_size, std=window_size/6)
    window = window / window.sum()

    # 对每个维度应用平滑
    timbre_component = np.zeros_like(features)
    for c in range(C):
        # 使用卷积计算移动平均
        padded = np.pad(features[:, c], (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded, window, mode='valid')
        timbre_component[:, c] = smoothed[:T]

    # 去除音色成分，保留内容成分（高频变化）
    content_features = features - timbre_component * 0.5  # 部分去除，避免过度

    return content_features


def adaptive_feature_replacement(
    source_features: np.ndarray,
    retrieved_features: np.ndarray,
    f0: np.ndarray,
    index_ratio: float = 0.5
) -> np.ndarray:
    """
    自适应特征替换 - 根据音高和能量动态调整替换强度

    参考: "One-Shot Singing Voice Conversion with Dual Attention" (arXiv:2508.05978)
    使用双注意力机制选择最相似的特征

    Args:
        source_features: 源特征 [T, C]
        retrieved_features: 检索到的特征 [T, C]
        f0: F0序列 [T*2]
        index_ratio: 基础索引率

    Returns:
        混合后的特征
    """
    T, C = source_features.shape

    # 计算每帧的替换权重
    weights = np.ones(T) * index_ratio

    # F0对齐到特征帧率
    f0_per_feat = 2
    for t in range(T):
        f0_start = t * f0_per_feat
        f0_end = min(f0_start + f0_per_feat, len(f0))

        if f0_end > f0_start:
            f0_segment = f0[f0_start:f0_end]
            avg_f0 = np.mean(f0_segment[f0_segment > 0]) if np.any(f0_segment > 0) else 0

            # 高音区域（>400Hz）：提高替换率
            if avg_f0 > 400:
                weights[t] = min(0.9, index_ratio * 1.5)
            # 中音区域（200-400Hz）：正常替换
            elif avg_f0 > 200:
                weights[t] = index_ratio
            # 低音区域（<200Hz）：降低替换率
            elif avg_f0 > 0:
                weights[t] = index_ratio * 0.8
            # 无声段（F0=0）：最小替换
            else:
                weights[t] = index_ratio * 0.3

    # 平滑权重曲线
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    weights = np.convolve(weights, kernel, mode='same')

    # 应用权重
    weights = weights[:, np.newaxis]  # [T, 1]
    mixed_features = source_features * (1 - weights) + retrieved_features * weights

    return mixed_features


def cosine_similarity_retrieval(
    query_features: np.ndarray,
    index_features: np.ndarray,
    k: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于余弦相似度的特征检索（而非欧氏距离）

    参考: "Enhancing zero-shot timbre conversion using semantic alignment" (arXiv:2507.09070)
    余弦相似度更关注特征方向（内容），而非幅度（音色）

    Args:
        query_features: 查询特征 [T, C]
        index_features: 索引特征 [N, C]
        k: 返回的近邻数量

    Returns:
        indices: 近邻索引 [T, k]
        similarities: 相似度分数 [T, k]
    """
    # 归一化特征
    query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    index_norm = index_features / (np.linalg.norm(index_features, axis=1, keepdims=True) + 1e-8)

    # 计算余弦相似度
    similarities = np.dot(query_norm, index_norm.T)  # [T, N]

    # 获取top-k
    indices = np.argsort(-similarities, axis=1)[:, :k]  # [T, k]

    # 获取对应的相似度分数
    similarities_topk = np.take_along_axis(similarities, indices, axis=1)

    return indices, similarities_topk


def weighted_feature_aggregation(
    index_features: np.ndarray,
    indices: np.ndarray,
    similarities: np.ndarray
) -> np.ndarray:
    """
    加权特征聚合 - 使用相似度加权而非距离倒数

    Args:
        index_features: 索引特征 [N, C]
        indices: 近邻索引 [T, k]
        similarities: 相似度分数 [T, k]

    Returns:
        聚合后的特征 [T, C]
    """
    T, k = indices.shape
    C = index_features.shape[1]

    # Softmax归一化权重
    exp_sim = np.exp(similarities * 10)  # 温度参数=0.1
    weights = exp_sim / (exp_sim.sum(axis=1, keepdims=True) + 1e-8)

    # 加权聚合
    retrieved = np.zeros((T, C))
    for t in range(T):
        for i, idx in enumerate(indices[t]):
            retrieved[t] += index_features[idx] * weights[t, i]

    return retrieved


def mitigate_timbre_leakage(
    features: np.ndarray,
    retrieved_features: np.ndarray,
    f0: np.ndarray,
    index_ratio: float = 0.5,
    use_whitening: bool = True,
    use_residual_removal: bool = True
) -> np.ndarray:
    """
    综合音色泄漏缓解策略

    结合多种技术：
    1. 特征白化
    2. 音色残差去除
    3. 自适应特征替换

    Args:
        features: 源特征
        retrieved_features: 检索到的特征
        f0: F0序列
        index_ratio: 索引混合率
        use_whitening: 是否使用白化
        use_residual_removal: 是否去除音色残差

    Returns:
        处理后的特征
    """
    # 1. 特征白化（可选）
    if use_whitening:
        features, src_mean, src_std = whiten_features(features)
        retrieved_features, tgt_mean, tgt_std = whiten_features(retrieved_features)

    # 2. 去除音色残差（可选）
    if use_residual_removal:
        features = extract_timbre_residual(features, f0, window_size=50)

    # 3. 自适应特征替换
    mixed_features = adaptive_feature_replacement(
        features,
        retrieved_features,
        f0,
        index_ratio
    )

    # 4. 恢复目标统计特征（如果使用了白化）
    if use_whitening:
        mixed_features = apply_target_statistics(mixed_features, tgt_mean, tgt_std)

    return mixed_features
