# TelkNet 同源音频更新记录

AI-RVC 1.6.0 移植了 TelkNet 项目的核心翻唱与音频处理改进，同时保持 AI-RVC 完全独立运行，保留固定版本的官方 RVC VC（`81eed5e8f68b6bed1789f682fe78cdd324495afc`）与 UVR5（`7ef19867780cf703841ebafb565a4e47d1ea86ff`）运行时。

## 核心同步项与本地适配

| 模块 | 同步内容与本地适配 |
|---|---|
| `infer/audio_splice.py`、`infer/rmvpe_boundaries.py`、`lib/rvc_chunking.py` | 同步绝对采样坐标裁剪与互补交叉淡化拼接算法，支持长歌曲与极短尾段无移位拼接 |
| `infer/rvc_checkpoint.py`、`infer/lib/applio_algorithm/` | 统一安全检查点加载、完整层校验与架构自动识别，支持标准 v1/v2、无 F0、多说话人及 Applio HiFi-GAN / MRF / RefineGAN |
| `infer/lib/rmvpe.py`、`infer/lib/infer_pack/` | 向量化 RMVPE 峰值解码、波形核验提前转音与短缺口，修复连续 F0 与激励相位低精度边带 |
| `infer/cover_pipeline.py` | 移植参考幅度 + 目标相位后处理、有界参考缓存、候选质量选择与单次间隙抑制 |
| 纯人声输入模式 | 新增 `CoverPipeline.process(input_mode="vocals")`，跳过分离与混音，直接导出输入人声与转换人声两轨 |
| `infer/separator.py` | 统一固定配置校验、唯一输出识别、静音主唱保留与短音频浮点精度补长裁切 |
| `lib/audio.py`、`lib/mixer.py`、`lib/vocoder_fix.py` | 全程保留 FLOAT32 / FLOAT64 浮点精度输出、混音与双精度 JUCE 混响 |
| 双索引角色包兼容 | 支持 `trained_*.index` + `added_*.index` 配对压缩包，验证空模板后自动选用推理索引 |

## 核心验证摘要

在 Windows（RTX 4070 Ti SUPER / PyTorch 2.11.0 / audio-separator 0.47.0 / Gradio 5.49.1）与 Linux 容器环境下完成以下实测验证：

- **长曲全流程验证**：使用同一花丸模型与默认参数连续完成 5 首 231–250 秒完整歌曲（含《Far far away》《梦的光点》《あこがれランララン》《SILENT PAIN》《だから僕は音楽を辞めた》）及 Rin 整曲测试；全部七类浮点 WAV 完整解码、无满幅削波，跨采样率时长误差低于 0.05 ms。
- **多路由覆盖**：默认整曲路线、官方 UVR5 路线、本地兼容入口、纯人声转换与 MCP `convert_voice` 均通过真实音频验证。
- **精度与回归覆盖**：补齐 GTX 16 系列显卡 FP32 规则、长音频分块拼接、RMVPE 边界、清辅音保护、同曲阶段复用与批量队列等全套回归测试。
