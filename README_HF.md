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

English summary: AI-RVC is a one-click [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) AI cover WebUI. It uses a practical cover pipeline with [RoFormer/Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) separation, [RMVPE](https://arxiv.org/abs/2306.15412) pitch extraction, [HuBERT](https://arxiv.org/abs/2106.07447) features, [FAISS](https://github.com/facebookresearch/faiss) retrieval, and [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) character models. The current pipeline focuses on local usability and model compatibility rather than claiming end-to-end 2026 research SOTA.

如果你想搜索或分享本项目，可以用这些关键词：AI 翻唱、RVC 翻唱、AI cover generator、RVC voice conversion、角色声线转换、人声分离、伴奏分离、HuBERT、RMVPE、FAISS、Gradio WebUI、Colab AI 翻唱。

## 功能特点

- **AI 歌曲翻唱**：上传 MP3/WAV/FLAC，自动分离人声、转换音色、混合伴奏，一键生成 AI cover。
- **人声分离**：默认 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 0.44.1 的 [ensemble:vocal_rvc](https://pypi.org/project/audio-separator/) 预设，属于 [RoFormer/Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 高质量实用路线，偏 RVC/AI cover 前处理。
- **音色转换**：采用 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 架构 + 官方兼容 VC 推理 + [FAISS](https://github.com/facebookresearch/faiss) 检索增强流程，搭配角色模型完成声线转换。
- **RMVPE 音高提取**：用于提取 F0 基频曲线；默认采用严格 [RMVPE](https://arxiv.org/abs/2306.15412) 路线，减少呼吸、齿音、换气声被误写成强音高。
- **角色模型**：内置 181 个可下载角色模型，支持系列筛选、关键词搜索和自定义模型导入。
- **混音效果**：支持人声混响、音量调节和 4 种混音预设。
- **卡拉OK模式**：分离主唱和伴唱轨道，方便对和声较多的歌曲做进一步处理。
- **VC预处理**：提供自动模式和严格 RoFormer De-Reverb 模式；不可用时显式停止，不静默降级到旧链路。
- **双VC管道**：支持当前实现和官方实现；默认兼顾官方兼容 VC 推理和项目当前后处理，可对比不同歌曲、不同模型下的效果。

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
   - 卡拉OK模式：启用主唱/伴唱分离
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
          人声分离 (RoFormer / Mel-Band RoFormer ensemble)
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

| 模块 | 当前判断 |
|------|----------|
| 人声分离 / 去混响 | 使用 [RoFormer/Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 系高质量开源路线，接近当前实用高端方案，适合 RVC 翻唱前处理，但不宣称所有数据集绝对 SOTA |
| [RMVPE](https://arxiv.org/abs/2306.15412) F0 | 论文支持的强默认，适合带伴奏歌声的 F0 提取 |
| [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 工程成熟、速度快、兼容现有角色模型；到 2026 已不是研究意义上的最新 SOTA |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) / [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) / [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf) / [SYKI-SVC](https://arxiv.org/abs/2501.02953) / [S2Voice](https://arxiv.org/abs/2601.13629) | 更前沿的零样本、扩散、流匹配或大模型式 SVC/SSC 方向；不是当前 RVC `.pth` 模型的直接替换 |

English note: [Seed-VC](https://github.com/Plachtaa/seed-vc), [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md)-like systems, [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf), [SYKI-SVC](https://arxiv.org/abs/2501.02953), and [S2Voice](https://arxiv.org/abs/2601.13629) are research-frontier directions for VC/SVC/SSC. Integrating them would require a new inference architecture, new model formats, new defaults, and a migration plan for existing character models.

## 常见问题

**Q: 首次运行很慢？**

A: 首次运行会自动下载模型文件（[HuBERT](https://arxiv.org/abs/2106.07447)、[RMVPE](https://arxiv.org/abs/2306.15412)、[RoFormer](https://arxiv.org/abs/2310.01809) 等），请耐心等待。

**Q: 高音断音/撕裂？**

A: 尝试降低保护系数（0.33 → 0.2），增大滤波半径（3 → 5）。

**Q: 转换后声音失真？**

A: 降低索引率，调整音调偏移，使用更高质量的输入音频。

**Q: 如何选择合适的角色？**

A: 建议选择与原唱性别、音色相近的角色，效果更自然。

## 性能说明

- **GPU 加速**：自动检测并使用 GPU（CUDA/ROCm）
- **处理时间**：一首 3-5 分钟的歌曲约需 2-5 分钟处理
- **显存需求**：建议 4GB 以上显存

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

本项目仅供学习研究和个人娱乐用途，不得用于任何商业目的。严禁使用本软件进行欺诈、传播虚假信息或侵犯他人权益。用户对使用本软件产生的所有内容和后果承担全部责任。

## 致谢

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 原始 RVC 项目
- [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) - 人声分离模型
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - 音源分离框架
- [RMVPE](https://arxiv.org/abs/2306.15412) - F0 提取
- [Gradio](https://gradio.app/) - Web 界面框架

---

**License**: MIT
**Version**: 2.0
**Last Updated**: 2026-07-04
