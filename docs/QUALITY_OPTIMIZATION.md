# RVC 音质优化指南

## 问题诊断与解决方案

基于最新的语音转换研究文献和项目实际测试，本文档提供针对性的音质优化方案。

---

## 一、高音段原唱声音泄漏 (Timbre Leakage)

### 问题表现
- 高音后段部分出现原唱声音
- 听起来像和声混入
- 音色转换不彻底

### 根本原因
1. **HuBERT特征音色泄漏** - 自监督特征中残留源说话人音色信息
2. **高频F0跟踪误差** - 高音区基频估计不准确
3. **FAISS检索不足** - index_rate=0.3 时，70%特征来自源音频

### 解决方案

#### 方案A: 提高索引混合率 (推荐用于高音多的歌曲)
```json
{
  "cover": {
    "index_rate": 0.75,  // 从 0.30 提升到 0.75
    "protect": 0.25      // 降低 protect 以允许更多检索
  }
}
```

**原理**: 更多使用训练数据中的特征，减少源音频特征的直接使用

**适用场景**:
- 高音段落多的歌曲
- 目标角色音色特征明显
- 可接受轻微口齿模糊的代价

#### 方案B: 优化F0提取 (推荐用于复杂音高变化)
```json
{
  "cover": {
    "f0_method": "hybrid",
    "rmvpe_threshold": 0.005,     // 从 0.01 降低，提高灵敏度
    "f0_min": 80,                 // 从 50 提高，避免低频噪声
    "f0_max": 1600,               // 从 1100 提高，支持更高音
    "filter_radius": 3            // 从 1 提高，平滑F0曲线
  }
}
```

**原理**: 更准确的F0跟踪 → 更好的音高对齐 → 减少音色泄漏

#### 方案C: 源约束增强 (已启用，可调整强度)
```json
{
  "cover": {
    "source_constraint_mode": "on",
    "vc_preprocess_mode": "uvr_deecho"
  }
}
```

**当前状态**: 已启用，通过 UVR DeEcho 预处理和源能量约束来抑制泄漏

---

## 二、破音/失真 (Artifacts)

### 问题表现
- 固定段落出现破音
- 声音失真、不自然
- 相比之前已改善但仍可听出

### 根本原因
1. **F0对齐误差** - 导致频谱特征失真
2. **RMS混合不当** - 能量包络不匹配
3. **分块边界伪影** - 交叉淡化不完美

### 解决方案

#### 方案A: 优化RMS混合 (推荐)
```json
{
  "cover": {
    "rms_mix_rate": 0.75,  // 从 0.25 提升，更多保留源能量包络
    "protect": 0.35        // 适当提高，保护辅音段
  }
}
```

**原理**:
- `rms_mix_rate` 控制输出音频的能量包络
- 值越高，越接近源音频的动态范围
- 可减少能量突变导致的失真

#### 方案B: 增强F0平滑
```json
{
  "cover": {
    "filter_radius": 5,           // 从 1 提升到 5
    "f0_stabilize": true,         // 启用F0稳定
    "f0_stabilize_window": 3,     // 稳定窗口
    "f0_stabilize_max_semitones": 3.0  // 最大修正半音数
  }
}
```

**原理**: 中值滤波和稳定算法平滑F0曲线，减少突变

#### 方案C: 调整分块参数 (需修改代码)
当前分块大小: 1500帧 (约30秒)
重叠: 50帧 (约1秒)

可在 `infer/pipeline.py` 中调整:
```python
chunk_size = 2000  # 增大分块，减少边界
overlap = 100      # 增大重叠，更平滑过渡
```

---

## 三、呼吸音/齿音/电流音 (Breath/Sibilance Artifacts)

### 问题表现
- 换气呼吸部分有电流音
- 齿音(s, sh, ch)失真
- 高音部分口齿不清

### 根本原因
1. **Protect参数过低** - 无声辅音保护不足
2. **呼吸段F0=0** - 导致特征检索失败
3. **高频能量处理不当** - 齿音频段(4-8kHz)失真

### 解决方案

#### 方案A: 提高辅音保护 (强烈推荐)
```json
{
  "cover": {
    "protect": 0.50,      // 从 0.30 提升到 0.50
    "index_rate": 0.40    // 适当降低，平衡音色转换
  }
}
```

**原理**:
- `protect` 控制无声段(F0=0)保留原始特征的比例
- 值越高，辅音/呼吸音越接近源音频
- 可显著减少电流音和齿音失真

**权衡**: 可能轻微影响音色转换彻底性

#### 方案B: 预处理增强 (已启用)
```json
{
  "cover": {
    "vc_preprocess_mode": "uvr_deecho",  // 已启用
    "uvr5_model": "VR-DeEchoDeReverb",   // 去回声模型
    "uvr5_agg": 10                       // 激进度
  }
}
```

**当前状态**: 已使用 UVR DeEcho 预处理，可进一步调整 `uvr5_agg`

#### 方案C: 后处理降噪 (需添加)
可在混音阶段添加 De-esser 和 Breath Reduction:

```python
# 在 lib/mixer.py 中添加
from pedalboard import Compressor, HighpassFilter

# 齿音抑制 (6-10kHz)
board.append(Compressor(
    threshold_db=-20,
    ratio=4,
    attack_ms=1,
    release_ms=50
))

# 呼吸音高通滤波 (80Hz以下)
board.append(HighpassFilter(cutoff_frequency_hz=80))
```

---

## 四、综合优化配置

### 配置1: 平衡型 (推荐大多数情况)
```json
{
  "cover": {
    "index_rate": 0.50,
    "filter_radius": 3,
    "rms_mix_rate": 0.50,
    "protect": 0.40,
    "f0_method": "hybrid",
    "rmvpe_threshold": 0.005,
    "f0_stabilize": true,
    "f0_stabilize_window": 3,
    "vc_preprocess_mode": "uvr_deecho",
    "source_constraint_mode": "on"
  }
}
```

### 配置2: 音色优先 (彻底转换，可能有轻微伪影)
```json
{
  "cover": {
    "index_rate": 0.80,
    "filter_radius": 5,
    "rms_mix_rate": 0.30,
    "protect": 0.25,
    "f0_stabilize": true
  }
}
```

### 配置3: 清晰度优先 (保留更多源特征，减少伪影)
```json
{
  "cover": {
    "index_rate": 0.30,
    "filter_radius": 1,
    "rms_mix_rate": 0.75,
    "protect": 0.55,
    "f0_stabilize": false
  }
}
```

---

## 五、参数详解

### index_rate (索引混合率)
- **范围**: 0.0 - 1.0
- **默认**: 0.30
- **作用**: 控制使用训练数据特征的比例
- **调整建议**:
  - 提高 → 音色更接近目标，但可能口齿模糊
  - 降低 → 保留更多源特征，清晰但音色转换不彻底

### protect (辅音保护)
- **范围**: 0.0 - 1.0
- **默认**: 0.30
- **作用**: 保护无声辅音和呼吸音
- **调整建议**:
  - 提高 → 减少齿音/呼吸音伪影，但音色转换减弱
  - 降低 → 音色转换更彻底，但可能产生电流音

### rms_mix_rate (能量混合率)
- **范围**: 0.0 - 1.0
- **默认**: 0.25
- **作用**: 控制输出音频的能量包络
- **调整建议**:
  - 提高 → 保留源音频动态，减少失真
  - 降低 → 使用目标音色能量特征

### filter_radius (F0滤波半径)
- **范围**: 0 - 10
- **默认**: 1
- **作用**: 中值滤波平滑F0曲线
- **调整建议**:
  - 提高 → F0更平滑，减少破音，但可能过度平滑
  - 降低 → 保留F0细节，但可能有抖动

---

## 六、实验建议

### 测试流程
1. 选择一首有问题的歌曲片段(30-60秒)
2. 使用不同配置进行转换
3. 对比评估:
   - 音色转换彻底性
   - 高音段泄漏情况
   - 辅音清晰度
   - 整体自然度

### A/B测试配置
```bash
# 测试1: 当前配置
python run.py

# 测试2: 平衡型配置
# 修改 config.json 后运行

# 测试3: 音色优先配置
# 再次修改后运行
```

---

## 七、参考文献

1. [Mitigating Timbre Leakage with Universal Semantic Mapping](https://arxiv.org/html/2504.08524v1)
2. [Adaptive Refinements of Pitch Tracking](https://www.mdpi.com/2076-3417/9/12/2460/htm)
3. [RVC FAISS Tuning Tips](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/faiss-tuning-TIPS)
4. [Voice Conversion for Articulation Disorders](https://www.researchgate.net/publication/270463154_A_preliminary_demonstration_of_exemplar-based_voice_conversion_for_articulation_disorders_using_an_individuality-preserving_dictionary/download)
5. [RVC Audio Quality Optimization Guide 2026](https://www.apatero.com/blog/rvc-audio-quality-optimization-tips-guide-2026)

---

## 八、下一步优化方向

### 短期 (可立即实施)
1. 调整配置参数进行A/B测试
2. 针对不同歌曲类型使用不同配置预设
3. 添加后处理降噪模块

### 中期 (需要代码修改)
1. 实现自适应 protect 参数 (根据F0置信度动态调整)
2. 优化分块策略 (更大重叠，更平滑过渡)
3. 添加频谱后处理 (De-esser, Breath Reduction)

### 长期 (需要模型改进)
1. 使用更新的特征提取器 (减少音色泄漏)
2. 训练更大的模型 (更多数据，更好泛化)
3. 实现端到端的歌唱转换模型
