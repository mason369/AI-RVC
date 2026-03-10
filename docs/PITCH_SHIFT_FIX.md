# 问题修复说明 - 音高偏移和撕裂

## 发现的问题

用户反馈：
1. **人声偏高音** - 声音听起来比原来高
2. **人声不自然** - 音色失真
3. **长音撕裂** - 高音和长时间同一个音会在最后出现撕裂

## 根本原因

之前实施的"架构级修复"引入了严重的bug：

### 1. 谐波增强导致音高偏移
```python
# lib/spectral_postprocess.py (已删除)
magnitude[freq_idx] *= 1.1  # 增强谐波
```

**问题**: 增强谐波会改变频谱平衡，让声音听起来更尖锐、更高音。这不是修复，而是破坏。

### 2. 特征白化破坏音色
```python
# lib/feature_disentangle.py (已删除)
whitened = (features - mean) / std
# 然后应用目标统计
result = whitened * tgt_std + tgt_mean
```

**问题**:
- 白化去除了源音频的音色特征
- 应用目标统计时，目标统计来自检索特征，不是真实的目标音色
- 导致音色完全失真，不自然

### 3. 频谱平滑破坏相位连续性
```python
# 时间维度平滑
smoothed_mag = convolve(magnitude, kernel)
```

**问题**:
- 只平滑幅度，不处理相位
- 长音时相位不连续会导致撕裂
- 过度平滑会破坏瞬态，导致口齿不清

### 4. 音色残差去除过度
```python
content_features = features - timbre_component * 0.5
```

**问题**:
- 去除了太多信息
- "音色残差"的定义本身就有问题
- 导致特征失真

## 修复方案

### 1. 移除所有有问题的模块
- ❌ 删除 `lib/feature_disentangle.py`
- ❌ 删除 `lib/spectral_postprocess.py`

这些模块理论上很好，但实际实现有严重bug，导致音质恶化。

### 2. 恢复简单的索引混合
```python
# 简单的自适应索引混合
adaptive_index_ratio = np.ones(len(features)) * index_ratio

for fi in range(len(features)):
    avg_f0 = np.mean(f0_segment[f0_segment > 0])
    # 高音区域提升索引率
    if avg_f0 > 450:
        adaptive_index_ratio[fi] = min(0.75, index_ratio * 1.3)

features = features * (1 - adaptive_index_ratio) + retrieved * adaptive_index_ratio
```

**优势**:
- 简单、可靠
- 不破坏音色
- 不改变音高

### 3. 降低后处理强度
```python
# 降低齿音和呼吸音处理强度
sibilance_reduction_db=3.0  # 从 4.0 降低
breath_reduction_db=6.0      # 从 8.0 降低
```

**原因**: 过度处理会导致不自然

### 4. 调整配置到保守值
```json
{
  "index_rate": 0.35,  // 从 0.40 降低
  "protect": 0.33      // 从 0.35 降低
}
```

## 经验教训

### 不要过度工程化
- ❌ 特征白化：理论很好，实践很差
- ❌ 音色残差去除：概念模糊，效果负面
- ❌ 谐波增强：改变了音色，不是修复
- ❌ 频谱平滑：破坏相位连续性

### 简单就是美
- ✅ 简单的线性混合
- ✅ 基于F0的自适应权重
- ✅ 保守的参数设置
- ✅ 最小化后处理

### 测试很重要
- 之前的"架构级修复"没有充分测试
- 理论上的改进不等于实际的改进
- 必须用真实歌曲测试，听音质

## 当前状态

已恢复到稳定的实现：
- ✅ 简单的索引混合
- ✅ 自适应高音处理
- ✅ 保守的后处理
- ✅ 合理的默认参数

应该不会再有音高偏移和撕裂问题。

## 下一步

如果还有问题，应该：
1. 检查模型本身的质量
2. 检查F0提取的准确性
3. 检查分块处理的边界
4. 而不是添加更多"修复"
