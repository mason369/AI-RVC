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

AI-RVC 是一个基于 [RVC v1/v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的一键 AI 翻唱与声音转换 WebUI。上传歌曲后，它会自动分离人声与伴奏，使用角色 RVC 模型转换主唱音色，再把转换后的人声和伴奏混成完整作品。

English summary: AI-RVC is a one-click [RVC v1/v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) AI cover WebUI. Separation input is normalized to 44.1 kHz stereo Float32 WAV, then the same PCM is sent to [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) Leap XE 90 bands for vocals and standard Leap Instrumental 62 bands from the same public repository for pure accompaniment. With Karaoke enabled, the [MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205) three-model BS-RoFormer `avg_wave` ensemble consumes the original full mix and returns lead vocals plus backing+instrumental. Pure backing vocals are derived by subtracting the lead from the Leap vocal mix. RVC converts only the lead; the final mix uses backing+instrumental directly so instruments are not doubled.

## 界面预览

![同步多轨、波形对照与七类原始音频输出](docs/多轨播放器.png)

本图采自 2026-09-06 本地实际翻唱结果，展示与 Space 共用的界面代码，不代表当前线上 Space 已更新。更多真实截图见[翻唱设置](docs/Windows界面.png)和[角色下载目录](docs/角色下载.png)。

## 功能特点

**部署验收状态（2026-09-06）：** 当前账户新建私有 Gradio CPU Space 返回 HTTP 402，平台要求 PRO；本次没有创建新实例，也没有更新既有线上 Space。Ubuntu/WSL 已完成 CPU 依赖安装、公共代码回归和默认六模型短音频完整流程，逐次测试数量见平台记录；这些记录不等于线上 Space 验收。资源权限以账户实际返回为准。

自有服务器可使用仓库新增的 [Docker / Compose 部署](https://github.com/mason369/AI-RVC/blob/master/docs/Docker使用指南.md)。CPU 与 NVIDIA CUDA 镜像的验证范围单独记录，不能替代本页 Gradio Space 的上线验收。

- **多轨试听**：与本地共用多轨播放器，支持同步播放、静音、独奏、逐轨增益、偏移、缩放和拖动添加对照轨；默认转换后人声＋伴奏。七种输出独立下载，试听调整不改写 WAV；所有前端资源内置，无需 CDN。

- **AI 歌曲翻唱**：上传 MP3/WAV/FLAC，自动分离人声、转换音色、混合伴奏，一键生成 AI cover。
- **人声分离**：分离输入统一解码为 44.1 kHz 双声道 Float32 WAV；Leap XE 90 bands 提取人声，标准 Leap Instrumental 62 bands 提取纯伴奏；[MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205) 三模型 `avg_wave` 处理原始整曲，输出主唱与带和声伴奏，再通过人声差分导出纯和声。
- **音色转换**：采用 [RVC v1/v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 架构 + 官方兼容 VC 推理 + [FAISS](https://github.com/facebookresearch/faiss) 检索增强流程，搭配角色模型完成声线转换。
- **RMVPE 音高提取**：用于提取 F0 基频曲线；默认使用 [RMVPE](https://arxiv.org/abs/2306.15412)。
- **角色模型**：保留 181 个注册条目，支持筛选、搜索和自定义导入；下载及导入均校验实际权重、索引和架构，列表明确区分资产校验与真实推理验证。
- **混音效果**：支持人声混响、音量调节和 4 种混音预设。
- **卡拉OK模式**：MVSep 9205 从原始整曲分离主唱与带和声伴奏；Leap 人声减去主唱后另存纯和声。
- **VC预处理**：固定使用 Stereo De-Reverb 22.5050；模型缺失或处理失败时停止并显示错误。
- **双VC管道**：提供标准翻唱和官方 RVC 路线；标准翻唱使用官方兼容 VC 推理及项目后处理，可对比不同歌曲、不同模型下的效果。

## 默认输出文件

分离引擎 0.47.0 的 MDXC/RoFormer 存在 DirectML 分配器限制；本项目加载时明确拒绝该组合，不自动改用 CPU。HF 默认 CPU 路线不受此限制。English: MDXC/RoFormer on DirectML is explicitly rejected; the HF CPU route is unaffected.

| 文件 | 内容 | 用途 |
|------|------|------|
| `lead_vocals.wav` | MVSep 9205 主唱 | RVC 音色转换输入 |
| `backing_vocals.wav` | Leap 人声减去 MVSep 主唱得到的纯和声 | 单独试听或后期处理 |
| `accompaniment.wav` | MVSep `Back+Instrumental` | 默认成品混音 |
| `accompaniment_without_harmony.wav` | Leap Instrumental 纯伴奏 | 不需要和声时使用 |

## 部署文件与依赖

部署到 Gradio Space 时，以本文件作为 Space 的 `README.md`，以 `requirements_hf.txt` 作为 Space 的 `requirements.txt`；同时保留根目录 `pre-requirements.txt`、`packages.txt`、`app.py` 及完整源码和前端资源。

`pre-requirements.txt` 固定安装工具版本，避免 pip 24.1 及以上拒绝 fairseq 依赖的旧元数据。HF 会先安装此文件，再处理 `requirements.txt`；系统依赖由 `packages.txt` 安装，详见 [Spaces 官方依赖文档](https://huggingface.co/docs/hub/spaces-dependencies)。CPU 清单包含 fairseq、Demucs、WORLD、音频解码和 MCP 所需依赖，不依赖运行中临时安装功能包。

本轮已在 Ubuntu 22.04 / Python 3.10 的隔离环境安装该清单并通过 `pip check` 及核心模块导入；线上 Space 的构建、资源配额与真实推理仍须在部署环境验收。完整范围见[平台适配与验收](docs/平台适配与验收.md)。

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
   - 音调偏移：默认 0，根据目标模型音域调整；无 F0 模型不支持移调
   - 混音预设：通用/人声突出/伴奏突出/现场感
   - 卡拉OK模式：启用 MVSep 9205 原曲主唱 / 带和声伴奏分离，并导出纯和声
4. 点击「🚀 开始翻唱」
5. 下载生成的翻唱作品

## 参数说明

### 基础参数

- **音调偏移**：半音数，正数升调，负数降调；默认 0，根据歌曲与目标模型音域调整
- **索引率**：同维度 FAISS 检索混合量；不是越高越好。没有索引时控件禁用并显示 0
- **说话人ID**：按权重真实说话人数限制范围；单说话人模型固定为 0

### 混音预设

- **通用**：默认均衡设置
- **人声突出**：人声 115%，伴奏 90%，混响 0%
- **伴奏突出**：人声 90%，伴奏 115%，混响 0%
- **现场感**：默认音量，混响 +10%

### 预处理与模型兼容

固定 Stereo De-Reverb 22.5050。支持标准 RVC v1/256 维、v2/768 维，原生 32/40/48 kHz，带 F0 / 无 F0。未知特征维度会报错。输出保留 Float32 WAV，允许 FP32/FP16 推理，不使用低比特量化。

官方诊断模式会关闭 Karaoke、源约束和静音门限后处理；相关控件会禁用。详见 [有效性与模型兼容性](docs/有效性与模型兼容性.md)。

FCPE 依赖 `torchfcpe==0.0.4`。离线官方 RVC 使用常规推理，避免变长音频反复捕获 CUDA Graph，不改变模型精度。默认整曲、v1/v2 和可选分离路线的真实验收目前来自本地 Windows CPU/CUDA；未把这些结果当作 HF Space 云端部署成功。

## 角色模型列表（181 个条目）

2026-09-06：178 个 HF 条目的配置文件在远端存在，2 个 Google Drive 角色已实下；本地 8 个角色带索引双路线推理共 16 次通过，不代表云端或全部 181 个权重均已验收。樱坂雫的 Mega 来源实测 `ENOENT`，保留失效提示，不能当作可下载模型。

Google Drive 使用 `gdown==6.0.0`。Mega 在 Windows/Linux x64 自动准备固定版、哈希校验的 Megatools；其他平台需要安装该工具。工具支持不代表失效链接恢复。原生 256/768 维检索保留完整 FP32 向量；IVFFlat 使用内存 FlatL2 精确搜索，不覆盖原索引，可能增加内存与耗时。

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
          人声分离 (Leap XE vocals + Leap Instrumental pure accompaniment)
              ↓
          原曲主唱 / 带和声伴奏分离 (MVSep 9205 avg_wave ensemble)
              ↓
          人声差分导出纯和声；Leap Instrumental 纯伴奏单独保留
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

2026-09-06 使用 0.47.0 环境重测同一份 30 秒人工混音：Leap Instrumental SI-SDR **19.2844 dB**，已移除的历史对照 PolarFormer **18.4565 dB**，Leap 高 **0.8280 dB**。这是本地 CUDA 的单一样本结果，不是 HF CPU 的运行成绩或原曲官方分数；完整方法和旧版记录见[伴奏模型实测](docs/伴奏模型实测.md)。

| 模块 | 项目状态 |
|------|----------|
| 人声分离 / 去混响 | 默认使用 Leap XE 90 vocals + standard Leap Instrumental 62 pure accompaniment；[MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205) 从原始整曲输出主唱与带和声伴奏，纯和声由 Leap 人声减去主唱得到；去混响使用 BS-RoFormer De-Reverb Stereo 22.5050 |
| [RMVPE](https://arxiv.org/abs/2306.15412) F0 | 用于复调音乐中的人声音高估计 |
| [RVC v1/v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 用于加载本项目的标准 RVC `.pth` 角色模型 |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) / [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) / [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf) / [SYKI-SVC](https://arxiv.org/abs/2501.02953) / [S2Voice](https://arxiv.org/abs/2601.13629) | 零样本、扩散、流匹配等 VC/SVC/SSC 研究；本项目未集成，无法直接载入 RVC `.pth` |

English note: [Seed-VC](https://github.com/Plachtaa/seed-vc), [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md)-like systems, [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf), [SYKI-SVC](https://arxiv.org/abs/2501.02953), and [S2Voice](https://arxiv.org/abs/2601.13629) are separate VC/SVC/SSC systems that are not integrated here. Integrating them would require a new inference architecture, new model formats, new defaults, and a migration plan for existing character models.

## 常见问题

**Q: 首次运行很慢？**

A: Space 启动时会先准备 [HuBERT](https://arxiv.org/abs/2106.07447)、[RMVPE](https://arxiv.org/abs/2306.15412)、Leap XE vocals、Leap Instrumental pure accompaniment、MVSep 9205 子模型、Stereo De-Reverb、新版 Transformers HuBERT 三文件目录与固定提交官方 RVC 源码（`_official_rvc_runtime/<commit>/`，保留旧源码目录）。缺少模型或下载失败会直接停止并显示错误。

**Q: 高音断音/撕裂？**

A: 分别试听分离主唱和转换人声，先定位问题环节。当前默认入口没有滤波半径参数。

**Q: 转换后声音失真？**

A: 降低索引率，调整音调偏移，使用更高质量的输入音频。

**Q: 如何选择合适的角色？**

A: 先用短片段试听目标模型，比较其音域、发音和音色是否适合歌曲。

## 性能说明

默认分离路线包含 Leap XE、Leap Instrumental，以及 Karaoke 开启时的三个 MVSep 9205 子模型。单首歌曲的分离阶段最多执行 5 次整曲模型推理，因此负载和耗时明显高于旧单模型路线。

本 Space 的 `requirements_hf.txt` 固定 `audio-separator[cpu]==0.47.0` 及完整 CPU PyTorch 栈，入口会显式设置 `AI_RVC_DEVICE=cpu`。自建 GPU Space 必须同时更换三项：将 `torch/torchvision/torchaudio` 的 CPU 固定版本及索引换成相互匹配的 CUDA 栈；将分离器改为同版本 `[gpu]` extra；设置 `AI_RVC_DEVICE=cuda`。仅切换环境变量或分离器 extra 不会把 CPU PyTorch 变成 CUDA 版。设备或 Provider 不可用时会直接报错。

RoFormer 采用模型 YAML 的分块长度、重叠次数和批量配置；默认 FP32，关闭 autocast、原生 FP16 和 `torch.compile`。0.47.0 修正重叠计算及尾部拼接，因此与旧版输出、耗时可能不同。静音轨仍保存为完整文件，导出失败明确报错。English: version 0.47.0 uses model-configured RoFormer overlap with FP32 eager inference; corrected chunk scheduling may change outputs and runtime. Silent stems remain valid files, and export failures propagate.

以下是容量规划参考，尚未逐档完成压力测试；实际资源占用取决于输入长度、模型和设备。

| 使用场景 | GPU | 系统内存 | CPU / 存储 | 说明 |
|----------|----------|----------|-------------|------|
| 默认分离路线 | NVIDIA CUDA，16GB 显存 | 64GB | 8 核以上；NVMe，至少 15GB 可用空间 | 适合 3～5 分钟歌曲，Karaoke 可保持开启 |
| 长音频或连续处理 | NVIDIA CUDA，24GB 显存以上 | 64GB 以上 | 12 核以上；NVMe，至少 30GB 可用空间 | WebUI 当前仍按任务串行处理 |
| 8～12GB 显存 | NVIDIA CUDA | 32GB 以上 | SSD，至少 15GB 可用空间 | 建议关闭 Karaoke；不要同时运行其他 GPU 程序 |
| CPU / 便携版 | 不需要 | 32GB 以上 | 8 核以上；SSD | 可以运行，但不适合批量处理 |

实际耗时取决于音频时长、设备、Karaoke 开关和模型加载状态。标准 Leap Instrumental 与 PolarFormer 的同输入分数、历史记录和评估方法见[伴奏模型实测](docs/伴奏模型实测.md)。该人工样本分数不能当作原曲真分轨或 HF CPU 运行速度。

## 界面与文件检查

选择语言并点击「保存语言」即时更新当前页面，保留手动参数和多轨试听设置。角色筛选后需重新选择有效角色。基础模型下载使用固定提交、文件大小及 SHA-256 校验；失败会显示原因，不以临时文件作为成功结果。Space 的硬件和持久化能力以部署配置为准。

## 限制说明

- **音频长度**：建议单次处理不超过 10 分钟
- **文件大小**：建议上传文件不超过 50MB
- **并发处理**：同时只能处理一个任务

## 更多信息

- **GitHub 仓库**：[mason369/AI-RVC](https://github.com/mason369/AI-RVC)
- **完整文档**：查看仓库中的 README.md
- **模型与评估说明**：README.md 的“使用的 AI 模型”章节列出当前默认、可选模型、研究前沿和论文依据
- **Colab 版本**：AI_RVC_Colab.ipynb
- **问题反馈**：GitHub Issues

## 免责声明

- 本仓库代码使用 MIT License，可以按许可证条款使用、修改和分发。
- 第三方模型、角色声音、歌曲和输入素材按各自许可证或授权条款使用，不随本仓库代码自动获得 MIT 授权。
- 不得将转换结果用于冒充、诈骗、误导、骚扰或其他违法侵权行为。

## 致谢

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 原始 RVC 项目
- [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) - 默认人声 stem 分离来源
- [MVSep Quality Checker 9205](https://www.mvsep.com/quality_checker/entry/9205) - 默认原曲主唱 / 带和声伴奏分离来源
- [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) - RoFormer / De-Reverb 路线的重要论文依据
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - 音源分离框架
- [RMVPE](https://arxiv.org/abs/2306.15412) - F0 提取
- [Gradio](https://gradio.app/) - Web 界面框架

---

**License**: MIT
**Documentation Updated**: 2026-09-06；云端部署状态以 Space 实际运行版本为准
