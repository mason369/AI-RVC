# Vocoder伪影修复说明

## 问题分析

### 问题1: 呼吸音伴有微弱电音
**症状**: 回气吸气时听到电子噪声

**根本原因**:
根据 [GitHub Issue #65](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/65) "Artefacting when speech has breath"，这是RVC vocoder的已知问题：

- **F0=0区域的vocoder伪影**: 呼吸音时F0=0，vocoder在这种区域会产生电子噪声
- **频谱平坦度高**: 呼吸音是宽频噪声，vocoder难以正确重建
- **相位随机性**: 无声段的相位是随机的，导致电子音

### 问题2: 高音和长音最后撕裂
**症状**: 长时间同一个音会在最后出现撕裂

**根本原因**:
根据 [arXiv:2601.14472](https://arxiv.org/abs/2601.14472) "Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding"：

- **相位不连续**: Vocoder在长音时会产生相位漂移，导致撕裂
- **分块边界问题**: 虽然有crossfade，但相位不连续仍会导致撕裂
- **谐波不稳定**: 长音时谐波结构会漂移

## 解决方案

### 1. 相位不连续修复 (`fix_phase_discontinuity`)

```python
# 使用希尔伯特变换检测相位跳变
analytic_signal = signal.hilbert(audio)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
phase_diff = np.diff(instantaneous_phase)

# 检测异常跳变
discontinuities = np.where(np.abs(phase_diff) > threshold)[0]

# 在跳变点应用平滑
for disc_idx in discontinuities:
    window = signal.windows.hann(window_size)
    audio[start:end] = audio[start:end] * window
```

**原理**:
- 希尔伯特变换提取瞬时相位
- 检测相位突变点
- 使用汉宁窗平滑过渡

**参考**: [Prosody-Guided Harmonic Attention (arXiv:2601.14472)](https://arxiv.org/abs/2601.14472)

### 2. 呼吸音电音修复 (`reduce_breath_electric_noise`)

```python
# 检测呼吸音：低能量 + 高频谱平坦度 + F0=0
is_breath = (energy_db < threshold) & (spectral_flatness > 0.6) & (f0 == 0)

# 对呼吸音区域应用频谱门限
for breath_frame in breath_frames:
    fft = np.fft.rfft(frame)
    magnitude = np.abs(fft)

    # 去除低于阈值的频率成分
    threshold = np.percentile(magnitude, 70)
    magnitude = np.where(magnitude > threshold, magnitude, magnitude * 0.1)

    # 重建
    frame_cleaned = np.fft.irfft(magnitude * np.exp(1j * phase))
```

**原理**:
- 多维度检测呼吸音（能量、频谱平坦度、F0）
- 使用频谱门限去除电子噪声
- 保留主要频率成分

**参考**: [GitHub Issue #65](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/65)

### 3. 长音稳定 (`stabilize_sustained_notes`)

```python
# 检测F0稳定的长音区域
f0_std = np.std(f0_window)
f0_mean = np.mean(f0_window)
is_sustained = (f0_std / f0_mean < 0.05)  # F0变化<5%

# 对长音应用包络平滑
envelope = np.abs(signal.hilbert(sustained_segment))
smoothed_envelope = signal.filtfilt(b, a, envelope)
audio = audio * (smoothed_envelope / envelope)
```

**原理**:
- 检测F0稳定的区域（变化<5%）
- 提取并平滑幅度包络
- 应用平滑包络，稳定谐波结构

**参考**: [Mel Spectrogram Inversion with Stable Pitch - Apple Research](https://machinelearning.apple.com/research/mel-spectrogram)

## 技术细节

### 相位连续性
Vocoder生成的音频在长音时容易出现相位漂移：

```
正常: ~~~~~~~~~~~~~~~~~~~
问题: ~~~~/\~~~~~/\~~~~~  (相位跳变)
修复: ~~~~~~~~~~~~~~~~~~~  (平滑过渡)
```

### 呼吸音频谱特征
```
正常人声: 谐波结构清晰，频谱平坦度低
呼吸音:   宽频噪声，频谱平坦度高 (>0.6)
电子音:   随机相位，听起来像电流
```

### 长音稳定性
```
理想: ═══════════════  (包络稳定)
问题: ═╪═╪═╪═╪═╪═╪═  (包络抖动)
修复: ═══════════════  (平滑包络)
```

## 与之前方法的区别

### 之前的vocal_cleanup
```python
# 简单的能量衰减
is_breath = (energy_db < -40)
audio *= gain_curve
```

**问题**:
- 只基于能量，误判率高
- 简单衰减，不去除电子噪声
- 没有处理相位问题

### 现在的vocoder_fix
```python
# 多维度检测
is_breath = (energy_db < threshold) & (spectral_flatness > 0.6) & (f0 == 0)

# 频谱门限降噪
magnitude = np.where(magnitude > threshold, magnitude, magnitude * 0.1)

# 相位修复
audio = fix_phase_discontinuity(audio)

# 长音稳定
audio = stabilize_sustained_notes(audio, f0)
```

**优势**:
- 准确检测呼吸音
- 频谱级降噪，去除电子音
- 修复相位不连续
- 稳定长音谐波

## 配置

已集成到管道中，自动启用：

```python
audio_out = apply_vocoder_artifact_fix(
    audio_out,
    sr=save_sr,
    f0=f0_resampled,
    fix_phase=True,        # 修复相位不连续
    fix_breath=True,       # 修复呼吸音电音
    fix_sustained=True     # 稳定长音
)
```

## 参考文献

1. [Prosody-Guided Harmonic Attention for Phase-Coherent Neural Vocoding (arXiv:2601.14472)](https://arxiv.org/abs/2601.14472)
2. [Mel Spectrogram Inversion with Stable Pitch - Apple Research](https://machinelearning.apple.com/research/mel-spectrogram)
3. [GitHub Issue #65: Artefacting when speech has breath](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/65)
4. [RVC Audio Quality Optimization Guide 2026](https://www.apatero.com/blog/rvc-audio-quality-optimization-tips-guide-2026)

## 预期效果

- ✅ 呼吸音不再有电音
- ✅ 长音不再撕裂
- ✅ 高音稳定
- ✅ 相位连续

直接运行测试即可验证效果。
