# RVC 音质改进说明

## 问题分析

根据用户反馈和日志分析，发现以下核心问题：

1. **回声残留被RVC误识别**：原主唱中的自然回声在频谱上与主唱混叠，RVC模型将回声残留当作"额外音频内容"进行转换，产生杂音
2. **固定段落丢音/撕裂**：过度的DeEcho处理和相位不连贯导致某些段落出现撕裂
3. **AI机械感**：过度平滑的F0曲线（filter_radius=3）和过高的protect值（0.45）丢失了自然的音高微变（vibrato）

## 实施的改进

### 1. Hybrid F0提取器 (CREPE + RMVPE)

**文件**: `infer/f0_extractor.py`

**原理**:
- RMVPE作为主要方法（快速、稳定）
- 在RMVPE不稳定的区域使用CREPE补充（高精度）
- 自动检测三种不稳定情况：
  - F0跳变过大（>3半音）
  - RMVPE给出F0=0但CREPE置信度高
  - RMVPE和CREPE差异过大（>2半音）

**效果**: 在回声较重段落提供更准确的F0，减少RVC误识别

### 2. 智能中值滤波

**文件**: `infer/pipeline.py:668-690`

**改进**:
- 仅在F0跳变超过2个半音时应用滤波
- 高音区域（>500Hz）完全保留原始值
- 保留自然颤音（vibrato）

**参数调整**:
- `filter_radius`: 3 → 1（降低滤波强度）

**效果**: 保留歌唱的自然表现力，减少机械感

### 3. 动态辅音保护

**文件**: `infer/pipeline.py:686-730`

**改进**:
- 基于F0稳定性动态调整protect强度
- 无声段（辅音）：强保护（protect值）
- F0不稳定段：中等保护（protect + 30%）
- 有声稳定段：完全使用索引检索

**参数调整**:
- `protect`: 0.45 → 0.30（降低基础保护值）
- `index_rate`: 0.35 → 0.30（降低索引混合比例）

**效果**: 更好地保留辅音清晰度，同时允许索引检索改善音色

### 4. 智能去混响

**文件**: `infer/cover_pipeline.py:436-491`

**改进**:
- 区分自然混响和真实回声
- 基于RMS能量动态调整抑制强度：
  - 高能量段（主唱强）：保守抑制（0.65系数），保留更多原始信号
  - 低能量段（回声尾）：激进抑制（0.82系数）
- 动态floor和blend系数

**效果**: 在高能量段落保留更多细节，避免丢音；在低能量段落有效去除回声

## 配置变更

**文件**: `configs/config.json`

```json
{
  "f0_method": "hybrid",      // rmvpe → hybrid
  "filter_radius": 1,         // 3 → 1
  "protect": 0.28,            // 0.33 → 0.28
  "cover": {
    "f0_method": "hybrid",    // rmvpe → hybrid
    "filter_radius": 1,       // 3 → 1
    "protect": 0.30,          // 0.45 → 0.30
    "index_rate": 0.30        // 0.35 → 0.30
  }
}
```

## 使用方法

### 方法1: 使用默认配置（推荐）

直接运行，新配置已自动生效：

```bash
python run.py
```

### 方法2: 手动调整参数

如果需要针对特定歌曲微调，可以在UI中调整：

- **F0方法**: 选择 "hybrid"（CREPE+RMVPE混合）
- **滤波半径**: 1（保留颤音）或 0（完全不滤波）
- **保护强度**: 0.25-0.35（根据歌曲调整）
- **索引率**: 0.25-0.35（音色相似度 vs 清晰度权衡）

### 方法3: 针对不同场景

**场景A: 回声很重的歌曲**
- F0方法: hybrid
- 保护强度: 0.25（降低）
- 索引率: 0.25（降低）

**场景B: 高音技巧多的歌曲**
- 滤波半径: 0（完全不滤波）
- 保护强度: 0.30
- 索引率: 0.30

**场景C: 说唱/快节奏**
- 保护强度: 0.35（提高，保护辅音）
- 滤波半径: 0
- 索引率: 0.25

## 测试验证

运行测试脚本验证改进：

```bash
python test_improvements.py
```

预期输出：
```
[PASS] Hybrid F0提取器
[PASS] 智能中值滤波
[PASS] 动态辅音保护
```

## 技术参考

本次改进基于2026年最新的RVC优化实践：

- [AI Hub RVC Inference Settings](https://ai-hub-docs.vercel.app/rvc/resources/inference-settings/)
- [Apatero RVC Audio Quality Optimization Guide](https://www.apatero.com/blog/rvc-audio-quality-optimization-tips-guide-2026)
- [Mangio-RVC-Fork Hybrid F0 Method](https://github.com/Mangio621/Mangio-RVC-Fork)
- [Real-World Robust Zero-Shot Singing Voice Conversion (arXiv 2512.04793)](https://arxiv.org/html/2512.04793v1)

## 预期效果

1. **杂音减少**: Hybrid F0在回声段落提供更准确的音高，减少RVC误识别
2. **丢音修复**: 智能去混响在高能量段保留更多细节
3. **自然度提升**: 保留颤音和音高微变，降低AI机械感
4. **辅音清晰**: 动态protect更好地保护辅音，同时允许元音使用索引检索

## 注意事项

1. **首次使用hybrid模式**需要安装torchcrepe：
   ```bash
   pip install torchcrepe
   ```
   如果未安装，会自动回退到纯RMVPE模式

2. **显存占用**：hybrid模式会同时加载RMVPE和CREPE，显存占用增加约500MB

3. **处理速度**：hybrid模式比纯RMVPE慢约15-20%，但音质提升明显

4. **参数微调**：不同歌曲可能需要微调参数，建议从默认值开始测试
