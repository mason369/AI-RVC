---
title: AI-RVC 一键 AI 翻唱
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🎤 AI-RVC 一键 AI 翻唱

AI-RVC 是一个基于 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的一键 AI 翻唱与声音转换 WebUI。上传歌曲后，它会自动分离人声与伴奏，使用角色 RVC 模型转换主唱音色，再把转换后的人声和伴奏混成完整作品。

English summary: AI-RVC is a one-click [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) AI cover WebUI. Non-WAV input is decoded once to 44.1 kHz stereo PCM16, then the same PCM is sent to [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) Leap XE 90 bands for vocals and [bgkb/bs_polarformer](https://huggingface.co/bgkb/bs_polarformer) BS PolarFormer public ONNX 62 bands for pure accompaniment. With Karaoke enabled, the [MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205) three-model BS-RoFormer `avg_wave` ensemble consumes the original full mix and returns lead vocals plus backing+instrumental. Pure backing vocals are derived by subtracting the lead from the Leap vocal mix. RVC converts only the lead; the final mix uses backing+instrumental directly so instruments are not doubled.

## 功能特点

- **AI 歌曲翻唱**：上传 MP3/WAV/FLAC，自动分离人声、转换音色、混合伴奏，一键生成 AI cover。
- **人声分离**：非 WAV 输入先统一解码为 44.1 kHz 双声道 PCM16；Leap XE 90 bands 提取人声，BS PolarFormer public ONNX 62 bands 提取纯伴奏并抑制孤立声道饱和；[MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205) 三模型 `avg_wave` 处理原始整曲，输出主唱与带和声伴奏，再通过人声差分导出纯和声。
- **音色转换**：采用 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 架构 + 官方兼容 VC 推理 + [FAISS](https://github.com/facebookresearch/faiss) 检索增强流程，搭配角色模型完成声线转换。
- **RMVPE 音高提取**：用于提取 F0 基频曲线；默认采用严格 [RMVPE](https://arxiv.org/abs/2306.15412) 路线，减少呼吸、齿音、换气声被误写成强音高。
- **角色模型**：内置 181 个可下载角色模型，支持系列筛选、关键词搜索和自定义模型导入。
- **混音效果**：支持人声混响、音量调节和 4 种混音预设。
- **卡拉OK模式**：MVSep 9205 从原始整曲分离主唱与带和声伴奏；Leap 人声减去主唱后另存纯和声。
- **VC预处理**：提供自动模式和严格 RoFormer De-Reverb 模式；不可用时显式停止，不静默降级到旧链路。
- **双VC管道**：支持当前实现和官方实现；默认兼顾官方兼容 VC 推理和项目当前后处理，可对比不同歌曲、不同模型下的效果。

## 默认输出文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `lead_vocals.wav` | MVSep 9205 主唱 | RVC 音色转换输入 |
| `backing_vocals.wav` | Leap 人声减去 MVSep 主唱得到的纯和声 | 单独试听或后期处理 |
| `accompaniment.wav` | MVSep `Back+Instrumental` | 默认成品混音 |
| `accompaniment_without_harmony.wav` | PolarFormer 纯伴奏 | 不需要和声时使用 |

## 使用方法

### 1. 下载角色模型

首次使用需要下载角色模型：
1. 进入「歌曲翻唱」标签页
2. 展开「下载角色模型」折叠面板
3. 选择并下载一个角色（推荐：星空凛、芙宁娜、纳西妲等）

### 2. 开始翻唱

1. 上传歌曲文件（支持 MP3/WAV/FLAC）
2. 选择已下载的角色
3. 调整参数：
   - 音调偏移：男转女 +12，女转男 -12
   - 混音预设：通用/人声突出/伴奏突出/现场感
   - 卡拉OK模式：启用 MVSep 9205 原曲主唱 / 带和声伴奏分离，并导出纯和声
4. 点击「🚀 开始翻唱」
5. 下载生成的翻唱作品

## 参数说明

### 基础参数

- **音调偏移**：半音数，正数升调，负数降调（男转女: +12, 女转男: -12）
- **索引率**：越高越像训练音色（建议 10-50%）
- **说话人ID**：多说话人模型的说话人选择（通常为 0）

### 混音预设

- **通用**：默认均衡设置
- **人声突出**：人声 +15%，伴奏 -10%，混响 -5%
- **伴奏突出**：人声 -10%，伴奏 +15%，混响 -5%
- **现场感**：默认音量，混响 +10%

### VC 预处理模式

- **自动**：使用当前默认的严格 RoFormer De-Reverb 路径（推荐）
- **严格 RoFormer De-Reverb**：明确指定同一条去混响/去回声预处理路径

## 可用角色模型（181 个）

| 系列 | 角色示例 |
|------|----------|
| Love Live! | 星空凛、园田海未、东条希、小泉花阳、南小鸟 |
| Love Live! Sunshine!! | 高海千歌、樱内梨子、黑泽黛雅、黑泽露比、国木田花丸 |
| Love Live! 虹咲学园 | 上原步梦、中须霞、天王寺璃奈、近江彼方、优木雪菜 |
| 原神 | 芙宁娜、枫原万叶、纳西妲、八重神子、雷电将军 |
| Hololive | Fuwawa、Mococo |
| 偶像大师 | 神崎兰子、梦见莉亚梦、双叶杏、本田未央、岛村卯月 |

> 完整列表请在 UI 中查看「下载角色模型」面板

## 技术架构

```
音频输入 → CoverPipeline
              ↓
          人声分离 (Leap XE vocals + PolarFormer pure accompaniment)
              ↓
          原曲主唱 / 带和声伴奏分离 (MVSep 9205 avg_wave ensemble)
              ↓
          人声差分导出纯和声；PolarFormer 纯伴奏单独保留
              ↓
          RoFormer De-Reverb 预处理
              ↓
          RVC 音色转换 (HuBERT + RMVPE + FAISS)
              ↓
          混音 (音量调节 + 混响)
              ↓
          AI 翻唱成品
```

## 模型定位 / Model Positioning

| 模块 | 项目状态 |
|------|----------|
| 人声分离 / 去混响 | 默认使用 Leap XE 90 vocals + BS PolarFormer public ONNX 62 pure accompaniment；[MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205) 从原始整曲输出主唱与带和声伴奏，纯和声由 Leap 人声减去主唱得到；去混响使用 RoFormer De-Reverb |
| [RMVPE](https://arxiv.org/abs/2306.15412) F0 | 论文支持的强默认，适合带伴奏歌声的 F0 提取 |
| [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 工程成熟、速度快、兼容现有角色模型；到 2026 已不是研究意义上的最新 SOTA |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) / [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) / [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf) / [SYKI-SVC](https://arxiv.org/abs/2501.02953) / [S2Voice](https://arxiv.org/abs/2601.13629) | 更前沿的零样本、扩散、流匹配或大模型式 SVC/SSC 方向；不是当前 RVC `.pth` 模型的直接替换 |

English note: [Seed-VC](https://github.com/Plachtaa/seed-vc), [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md)-like systems, [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf), [SYKI-SVC](https://arxiv.org/abs/2501.02953), and [S2Voice](https://arxiv.org/abs/2601.13629) are research-frontier directions for VC/SVC/SSC. Integrating them would require a new inference architecture, new model formats, new defaults, and a migration plan for existing character models.

## 常见问题

**Q: 首次运行很慢？**

A: Space 启动时会先准备 [HuBERT](https://arxiv.org/abs/2106.07447)、[RMVPE](https://arxiv.org/abs/2306.15412)、Leap XE vocals、BS PolarFormer pure accompaniment、MVSep 9205 子模型、RoFormer De-Reverb 和内置官方 RVC 源码。缺少模型或下载失败会直接停止并显示错误。

**Q: 高音断音/撕裂？**

A: 尝试降低保护系数（0.33 → 0.2），增大滤波半径（3 → 5）。

**Q: 转换后声音失真？**

A: 降低索引率，调整音调偏移，使用更高质量的输入音频。

**Q: 如何选择合适的角色？**

A: 建议选择与原唱性别、音色相近的角色，效果更自然。

## 性能说明

默认高质量路线包含 Leap XE、PolarFormer，以及 Karaoke 开启时的三个 MVSep 9205 子模型。单首歌曲的分离阶段最多执行 5 次整曲模型推理，因此负载和耗时明显高于旧单模型路线。

本 Space 的 `requirements_hf.txt` 使用 `audio-separator[cpu]`，入口会显式设置 `AI_RVC_DEVICE=cpu`。自建 GPU Space 需要改用 GPU extra，并显式设置 `AI_RVC_DEVICE=cuda`；设备或 Provider 不可用时会直接报错。

| 使用场景 | 推荐 GPU | 系统内存 | CPU / 存储 | 说明 |
|----------|----------|----------|-------------|------|
| 默认高质量路线 | NVIDIA CUDA，16GB 显存 | 64GB | 8 核以上；NVMe，至少 15GB 可用空间 | 适合 3～5 分钟歌曲，Karaoke 可保持开启 |
| 长音频或连续处理 | NVIDIA CUDA，24GB 显存以上 | 64GB 以上 | 12 核以上；NVMe，至少 30GB 可用空间 | WebUI 当前仍按任务串行处理 |
| 8～12GB 显存 | NVIDIA CUDA | 32GB 以上 | SSD，至少 15GB 可用空间 | 建议关闭 Karaoke；不要同时运行其他 GPU 程序 |
| CPU / 便携版 | 不需要 | 32GB 以上 | 8 核以上；SSD | 可以运行，但不适合批量处理 |

实际耗时取决于音频时长、GPU、Karaoke 开关和模型是否已加载，不再给出统一的“2～5 分钟”估算。PolarFormer 默认窗口上限为 `441000` 个采样点；减小窗口能降低峰值显存，但会增加分块数量和总耗时。

## 限制说明

- **音频长度**：建议单次处理不超过 10 分钟
- **文件大小**：建议上传文件不超过 50MB
- **并发处理**：同时只能处理一个任务

## 更多信息

- **GitHub 仓库**：https://github.com/mason369/AI-RVC
- **完整文档**：查看仓库中的 README.md
- **模型与 SOTA 说明**：README.md 的“使用的 AI 模型”章节列出当前默认、可选模型、研究前沿和论文依据
- **Colab 版本**：AI_RVC_Colab.ipynb
- **问题反馈**：GitHub Issues

## 免责声明

- 本仓库代码使用 MIT License，可以按许可证条款使用、修改和分发。
- 第三方模型、角色声音、歌曲和输入素材按各自许可证或授权条款使用，不随本仓库代码自动获得 MIT 授权。
- 不得将转换结果用于冒充、诈骗、误导、骚扰或其他违法侵权行为。

## 致谢

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 原始 RVC 项目
- [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) - 默认人声 stem 分离来源
- [bgkb BS PolarFormer ONNX](https://huggingface.co/bgkb/bs_polarformer) - 默认纯伴奏 stem 分离来源
- [MVSep Quality Checker 9205](https://www.mvsep.com/quality_checker/entry/9205) - 默认原曲主唱 / 带和声伴奏分离来源
- [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) - RoFormer / De-Reverb 路线的重要论文依据
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - 音源分离框架
- [RMVPE](https://arxiv.org/abs/2306.15412) - F0 提取
- [Gradio](https://gradio.app/) - Web 界面框架

---

**License**: MIT
**Version**: 2.0
**Last Updated**: 2026-07-12
