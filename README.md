# AI-RVC 一键 AI 翻唱 / RVC Voice Conversion WebUI

AI-RVC 是一个面向普通用户和创作者的 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) AI 翻唱与声音转换工具。上传一首歌，它会自动分离人声和伴奏，用角色 RVC 模型转换主唱音色，再把转换后的人声、伴奏和混响重新混成完整作品。

不用先手动拆音轨，也不用在一堆脚本里来回切。打开 Gradio WebUI，选歌、选角色、点开始，一首 AI cover 就能从原曲一路跑到成品。

> 在线体验：[https://telknet.cc/](https://telknet.cc/)

**平台支持：Windows / Linux / WSL2 / Google Colab / Hugging Face Spaces**

## 项目定位与搜索关键词

如果你在找 **AI 翻唱、RVC 翻唱、AI cover generator、RVC voice conversion、角色声线转换、人声分离、伴奏分离、HuBERT、RMVPE、FAISS、Gradio WebUI、Colab AI 翻唱** 这类工具，AI-RVC 的目标就是把这些零散步骤串成一条更省心的工作流。

适合放在 GitHub About 的仓库简介：

> 一键 AI 翻唱与 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 声音转换 WebUI：自动人声分离、[HuBERT](https://arxiv.org/abs/2106.07447) + [RMVPE](https://arxiv.org/abs/2306.15412) + [FAISS](https://github.com/facebookresearch/faiss) 音色转换、角色模型下载、混音预设，并支持 Windows、Linux、WSL2、Google Colab 和 Hugging Face Spaces。

推荐 GitHub Topics：

`rvc`, `rvc-v2`, `voice-conversion`, `ai-cover`, `song-cover`, `singing-voice-conversion`, `voice-changer`, `voice-cloning`, `vocal-separation`, `audio-separation`, `rmvpe`, `hubert`, `faiss`, `gradio`, `pytorch`, `colab`, `uvr`, `demucs`, `roformer`, `ai-music`

## 功能特点

- **AI 歌曲翻唱**：上传 MP3/WAV/FLAC，自动完成人声分离、RVC 音色转换、伴奏混合和结果导出，一首歌从原曲跑到 AI cover 成品。
- **人声分离**：默认使用 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 0.44.1 的 [ensemble:vocal_rvc](https://pypi.org/project/audio-separator/) 预设；这是偏 RVC/AI cover 前处理的 [RoFormer/Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 高质量实用路线，不把它夸成所有场景绝对 SOTA。
- **音色转换**：采用 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 架构 + 官方兼容 VC 推理，结合 [HuBERT](https://arxiv.org/abs/2106.07447) 特征、角色模型和 [FAISS](https://github.com/facebookresearch/faiss) 检索增强流程，让声线更贴近目标音色。
- **RMVPE 音高提取**：按 [RMVPE](https://arxiv.org/abs/2306.15412) 论文报告，在公开基准上优于 [CREPE](https://github.com/marl/crepe) / pYIN / SWIPE 等基线，并具备更好的噪声鲁棒性；项目默认采用严格 RMVPE 路线，减少呼吸、齿音被误写成强 F0。
- **角色模型**：内置可下载角色清单 181 项（以 `tools/character_models.py` 为准），支持系列筛选、关键词搜索和自定义模型导入。
- **混音效果**：支持人声混响、音量调节、原声混合，生成结果不用再额外开一套音频工程。
- **混音预设**：4 种预设（通用、人声突出、伴奏突出、现场感），想快一点就一键应用，想细一点也能继续手调。
- **卡拉OK模式**：分离主唱和伴唱轨道，支持独立处理和混合，适合和声多、伴唱明显的歌曲。
- **VC预处理**：提供自动模式和严格 RoFormer De-Reverb 模式；不可用时显式停止，不静默降级到旧链路。
- **双VC管道**：支持当前实现和官方实现；默认保留项目当前预处理/后处理，同时使用官方兼容 VC 推理，方便按歌曲素材、模型效果做 A/B 对比。
- **GPU 加速**：自动检测并使用 CUDA / ROCm / XPU / DirectML / MPS / CPU。
- **简洁界面**：基于 Gradio 的中文图形界面，支持本地 Web、Google Colab 和 Hugging Face Spaces。

## 平台支持

| 平台 | 状态 | 安装方式 | 说明 |
|------|------|---------|------|
| Windows 10/11 (x64) | ✅ 已支持 | 可执行文件 / 本地安装 | 推荐使用可执行文件，无需安装 Python |
| Linux (Ubuntu/Debian) | ✅ 支持 | 可执行文件 / 本地安装 | 推荐 Ubuntu 22.04+；GPU 版本请按本机驱动选择 PyTorch wheel |
| WSL2 (Windows 11) | ✅ 已支持 | 本地安装 | 可直接通过浏览器访问 `http://127.0.0.1:7860` |
| Google Colab | ✅ 支持 | 在线使用 | 使用独立 Python 3.10 环境，按 Notebook 顺序运行即可 |
| Hugging Face Spaces | ✅ 已支持 | 在线使用 | 免费 CPU / 付费 GPU |
| macOS | 实验性支持 | 本地安装 | 可尝试 CPU 模式；MPS 路径尚未适配 |

## 快速开始

> **💡 推荐方式**：
> - **新手用户**：使用方式 1（可执行文件），无需安装 Python，开箱即用
> - **开发者/频繁使用**：使用方式 4（本地安装），运行 `python install.py` 一键完成环境配置
> - **临时体验**：使用方式 2（Google Colab）或方式 3（Hugging Face Spaces）

### 方式 1：可执行文件（推荐新手，无需安装 Python）

#### Windows

1. 从 [Releases](https://github.com/mason369/AI-RVC/releases/latest) 下载 `AI-RVC-Windows-Portable.zip`
2. 解压到任意目录
3. 双击 `AI-RVC-Windows.exe` 启动
4. 浏览器自动打开 http://127.0.0.1:7860

#### Linux

1. 从 [Releases](https://github.com/mason369/AI-RVC/releases/latest) 下载 `AI-RVC-Linux-Portable.tar.gz`
2. 解压：`tar -xzf AI-RVC-Linux-Portable.tar.gz`
3. 添加执行权限：`chmod +x AI-RVC-Linux-Portable/AI-RVC-Linux`
4. 运行：`./AI-RVC-Linux-Portable/AI-RVC-Linux`
5. 浏览器访问 http://127.0.0.1:7860

**优势**：
- ✅ 无需安装 Python 和依赖
- ✅ 开箱即用，双击启动
- ✅ 包含所有必需模型
- 仅支持 CPU 推理（构建时使用 CPU 版 PyTorch 以控制包体积）
- 💡 如需 GPU 加速，请使用方式 4 本地安装（`python install.py`）
- 首次启动需要 5-10 分钟下载模型

### 方式 2：Google Colab（推荐临时使用）

1. 打开 Colab notebook：[AI_RVC_Colab.ipynb](https://colab.research.google.com/github/mason369/AI-RVC/blob/master/AI_RVC_Colab.ipynb)
2. 确保运行时类型设置为 **GPU**（菜单栏 → 代码执行程序 → 更改运行时类型 → T4 GPU）
3. 按顺序执行每个单元格
4. 启动 Gradio 界面后，点击生成的公共链接访问

**Colab 说明**：
- Notebook 会在 Colab 内创建独立 Python 3.10 环境，避免默认 Python 版本变化影响运行
- 安装流程会调用 `install.py --no-run`，并检查 `fairseq==0.12.2`、`audio-separator==0.44.1`、CUDA、HuBERT、RMVPE 等关键依赖和模型
- Gradio 启动前会检查环境和必需模型，缺少关键项时会直接提示错误

### 方式 3：Hugging Face Spaces（在线体验）

访问：https://huggingface.co/spaces/mason369/AI-RVC

**优势**：
- 无需安装，直接使用
- 随时随地访问
- 易于分享

**限制**：
- 免费版使用 CPU（处理较慢）
- 可升级到 GPU（付费）

### 方式 4：本地安装（推荐开发者和频繁使用）

#### 一键安装（推荐）

**Windows**

```powershell
# 1. 克隆仓库
git clone https://github.com/mason369/AI-RVC.git
cd AI-RVC

# 2. 运行一键安装脚本（自动创建虚拟环境、安装依赖）
python install.py

# 脚本会自动：
# - 检测并创建 Python 3.10 虚拟环境
# - 安装 PyTorch（自动检测 CUDA/CPU）
# - 安装所有项目依赖
# - 启动 Web 界面（首次运行时会准备必需模型和内置官方 RVC 源码）
```

**Linux / WSL2**

```bash
# 1. 克隆仓库
git clone https://github.com/mason369/AI-RVC.git
cd AI-RVC

# 2. 运行一键安装脚本
python3.10 install.py

# 或仅检查环境（不安装）
python3.10 install.py --check

# 或安装 CPU 版本
python3.10 install.py --cpu
```

**脚本选项**：
- 无参数：完整安装 + 自动启动
- `--check`：仅检查环境和依赖，不安装
- `--cpu`：安装 CPU 版本 PyTorch（无 GPU 加速）
- `--no-run`：安装完成后不自动启动

> 脚本会自动创建 `venv310` 虚拟环境并在其中安装所有依赖。安装后手动启动请使用虚拟环境中的 Python：
> - Windows：`venv310\Scripts\python run.py`
> - Linux：`venv310/bin/python run.py`

访问 http://127.0.0.1:7860 打开界面。

首次运行翻唱时，audio-separator 会自动下载分离模型并缓存在 `assets/separator_models/`（体积随上游模型版本变化，通常为数百 MB）。

---

#### 手动安装（高级用户）

如果需要自定义安装流程，可以手动执行以下步骤：

**Windows**

```powershell
# 1. 克隆仓库
git clone https://github.com/mason369/AI-RVC.git
cd AI-RVC

# 2. 创建虚拟环境
python -m venv venv310
.\venv310\Scripts\Activate.ps1

# 3. 安装 PyTorch（先在官方页面生成与你环境匹配的命令）
# https://pytorch.org/get-started/locally/
# 示例（CUDA 12.6，2026-03-06）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
# CPU 示例
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 准备必需模型与内置官方 RVC 源码
python tools/download_models.py

# 6. 启动
python run.py
```

**Linux / WSL2**

```bash
# 1. 克隆仓库
git clone https://github.com/mason369/AI-RVC.git
cd AI-RVC

# 2. 创建虚拟环境
python3.10 -m venv venv310
source venv310/bin/activate

# 3. 安装 PyTorch + 依赖
# 先在 https://pytorch.org/get-started/locally/ 生成命令
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 4. 准备必需模型与内置官方 RVC 源码 + 启动
python tools/download_models.py
python run.py
```

---

**Linux 兼容性说明**：
- ✅ 核心代码路径使用 `pathlib.Path` 和跨平台设备检测，按设计支持 Linux / WSL2
- ✅ 虚拟环境路径自动适配（`bin/python` vs `Scripts/python.exe`）
- ✅ 音频处理库（librosa, soundfile, ffmpeg）在 Linux 上通常表现稳定
- ✅ CUDA GPU 路径按 PyTorch Linux wheel 支持；ROCm 取决于本机 AMD 驱动、PyTorch ROCm wheel 与系统版本
- `fairseq==0.12.2`、`pyworld`、`audio-separator[gpu]` 等依赖在不同 Linux 发行版上可能需要编译工具链和系统音频/FFmpeg 依赖
- Linux / WSL2 首次使用建议先运行 `python3.10 install.py --check`，再用一小段音频试跑完整翻唱流程

**安装脚本说明**：
- `install.py` 会自动检测系统环境（Windows/Linux）并完成以下步骤：
  1. **检测 Python 3.10**：Windows 检查常见安装路径 + `py -3.10` 启动器；Linux 使用 `python3.10` 命令
  2. **创建虚拟环境**：在 `venv310/` 目录创建隔离的 Python 环境
  3. **安装 PyTorch**：自动检测 CUDA 可用性，安装对应版本（GPU/CPU）
  4. **安装项目依赖**：从 `requirements.txt` 安装所有必需包（包括 fairseq、audio-separator 等）
  5. **启动应用**：自动运行 `run.py` 启动 Web 界面（除非使用 `--no-run`）
- 必需模型（HuBERT、RMVPE、UVR5 HP2）和内置官方 RVC 源码会在首次运行或 `python tools/download_models.py` 时准备；缺少 git 或 `_official_rvc/` 不完整会显式报错停止
- 支持参数：`--check`（仅检查）、`--cpu`（CPU 版本）、`--no-run`（不自动启动）
- 如果虚拟环境已存在，会跳过创建步骤，直接检查依赖

## 依赖版本说明

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10+ | 推荐 3.10 |
| PyTorch | >= 2.0.0 | 语音转换 + 人声分离 |
| torchaudio | >= 2.0.0 | 与 PyTorch 版本对应 |
| CUDA | 与 torch wheel 匹配 | 常见 11.8 / 12.1 / 12.4 / 12.6（可选） |
| fairseq | 0.12.2 | HuBERT 特征提取 |
| [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | 0.44.1（requirements 锁定） | [RoFormer / Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 分离与 ensemble 预设；当前默认偏 RVC 翻唱前处理，不宣称所有场景绝对 SOTA |
| demucs | >= 4.0.0 | Demucs 人声分离（可选） |

> 建议使用 `python install.py` 安装依赖。当前依赖栈使用 Gradio 5 与 NumPy 2，以匹配 `audio-separator` 0.44.1 的 ensemble 预设和上游包元数据。

## 使用方法

### 歌曲翻唱（推荐）

1. 进入「歌曲翻唱」标签页
2. **下载角色模型**（首次使用）：
   - 展开「下载角色模型」折叠面板
   - 可按系列筛选或关键词搜索
   - 点击「下载选中角色」下载单个角色
   - 或点击「下载该分类全部」批量下载
3. **上传歌曲**：支持 MP3/WAV/FLAC 格式
4. **选择角色**：从已下载的角色列表中选择
5. **调整参数**：
   - 基础参数：音调偏移、索引率、说话人ID
   - 卡拉OK设置：启用主唱/伴唱分离
   - VC预处理模式：自动/严格 RoFormer De-Reverb
   - 源约束策略：自动/关闭/启用
   - VC管道模式：当前实现/官方实现
   - 混音预设：通用/人声突出/伴奏突出/现场感
   - 混音参数：人声音量、伴奏音量、混响、RMS混合率
6. **开始翻唱**：点击「🚀 开始翻唱」按钮
7. **下载结果**：
   - 最终翻唱（混合后的完整作品）
   - 转换后的人声
   - 原始人声
   - 主唱轨道（如启用卡拉OK）
   - 伴唱轨道（如启用卡拉OK）
   - 伴奏

### 角色模型管理

**查看可用角色**：
- 181 个角色，涵盖 Love Live!、原神、Hololive、偶像大师等系列
- 支持按系列筛选和关键词搜索
- 显示格式：【语言】角色名（出处）[内部名]

**下载方式**：
- 单个下载：选择角色后点击「下载选中角色」
- 批量下载：选择系列后点击「下载该分类全部」
- 全部下载：点击「下载全部角色模型」（需要较长时间）

**已下载角色**：
- 自动刷新列表
- 支持按系列筛选和关键词搜索
- 点击「刷新」按钮手动更新

## 支持的格式

**输入**：MP3, WAV, FLAC（UI 明确支持；其他格式取决于后端解码器）

**输出**：WAV（翻唱成品 + 分离人声 + 伴奏）

## 技术架构

```
音频输入 → CoverPipeline
              ↓
          ┌─ 步骤 1：人声分离 ─────────────────────────────┐
          │  Mel-Band Roformer (默认) / UVR5 / Demucs      │
          │      ↓                                         │
          │  人声 (vocals.wav) + 伴奏 (accompaniment.wav)  │
          └────────────────────────────────────────────────┘
              ↓
          ┌─ 步骤 2：RVC 语音转换 ─────────────────────────┐
          │  HuBERT 特征提取 → RMVPE F0 提取               │
          │      ↓                                         │
          │  RVC v2 推理（角色模型 + FAISS 索引检索）       │
          │      ↓                                         │
          │  转换后人声 (converted_vocals.wav)              │
          └────────────────────────────────────────────────┘
              ↓
          ┌─ 步骤 3：混音 ─────────────────────────────────┐
          │  转换人声 + 伴奏 → 音量调节 + 混响             │
          │      ↓                                         │
          │  AI 翻唱成品 (cover.wav)                       │
          └────────────────────────────────────────────────┘
```

### 使用的 AI 模型

本项目的翻唱效果不是由单个模型决定，而是由“分离 → 去混响/预处理 → F0 → RVC → 后处理 → 混音”整条链路共同决定。这里先给出结论，再列出当前默认、可选模型、研究前沿和依据。

**简要结论**：

- 默认链路是面向 AI cover 的质量优先 RVC 工作流，不是把论文榜单里每个任务的第一名硬拼到一起。
- 人声/伴奏分离和去混响使用 [RoFormer / Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 系路线，属于开源实用圈很强、接近当前高端实践的方案，尤其适合给 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 提供更干净的主唱。
- [RMVPE](https://arxiv.org/abs/2306.15412) 仍是合理默认。它的论文目标就是从带伴奏的复调音乐里估计人声音高，和 AI 翻唱场景高度相关。
- [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 不是 2026 年研究意义上的绝对 SOTA，但它速度快、可控、可本地运行，并且兼容现有 `.pth` / `.index` 角色模型，所以仍是本项目默认。
- [Seed-VC](https://github.com/Plachtaa/seed-vc)、[Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md)、[Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf)、[SYKI-SVC](https://arxiv.org/abs/2501.02953)、[S2Voice](https://arxiv.org/abs/2601.13629) 等属于更前沿的 VC / SVC / SSC 方向，但它们不是 RVC 模型的直接替换件，需要新模型格式、新推理代码和新的角色模型生态。

English summary: AI-RVC uses a practical RVC-cover pipeline rather than a single universal SOTA model. [RoFormer/Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) separation is a strong open-source choice for cover preprocessing, [RMVPE](https://arxiv.org/abs/2306.15412) is a well-supported default for vocal F0 extraction, and [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) remains the best fit for local character-model covers even though newer research systems such as [Seed-VC](https://github.com/Plachtaa/seed-vc), [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md), [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf), [SYKI-SVC](https://arxiv.org/abs/2501.02953), and [S2Voice](https://arxiv.org/abs/2601.13629) push the frontier of zero-shot VC, SVC, and singing style conversion.

---

### 默认质量链路

| 环节 | 当前默认 | 作用 | 为什么这样选 |
|------|----------|------|--------------|
| 人声/伴奏分离 | [ensemble:vocal_rvc](https://pypi.org/project/audio-separator/) | 从原曲中分离主唱和伴奏 | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 0.44.1 的 RVC 向 ensemble 预设，包含两个 [RoFormer / Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 人声模型，目标是给 RVC 提供更干净的主唱输入 |
| 卡拉OK分离 | [ensemble:karaoke](https://pypi.org/project/audio-separator/) | 从人声里继续分离主唱和伴唱 | 和声较多的歌曲会把伴唱混进主唱；三模型 karaoke ensemble 可以降低主唱被伴唱污染的概率 |
| 去混响/去回声 | [dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt](https://pypi.org/project/audio-separator/) | 给 VC 输入更干的人声 | 高回声或大混响输入会把原唱空间感带进转换结果，容易影响和伴奏重新混合后的清晰度 |
| 内容特征 | [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) | 提取 RVC v2 所需的语音内容特征 | [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 模型生态绑定 [HuBERT](https://arxiv.org/abs/2106.07447) 特征，不能随意换成 [ContentVec](https://proceedings.mlr.press/v162/qian22b.html) / [WavLM](https://github.com/microsoft/unilm/blob/master/wavlm/README.md) / [Whisper encoder](https://github.com/openai/whisper) |
| 音高提取 | [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)，默认 `f0_method=rmvpe` + `f0_hybrid_mode=off` | 保留原曲旋律和 F0 走向 | RMVPE 适合带伴奏歌声；严格默认路线会拒绝 `hybrid` / fallback 配置，避免呼吸、齿音、换气声被误写成强音高 |
| 语音转换 | [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，默认 `vc_pipeline_mode=current` + `use_official=true` | 把主唱转换成目标角色音色 | 先用官方兼容 VC 推理，再接项目当前清理、源约束和混音链路，兼顾上游一致性和本项目的翻唱后处理 |
| 混音 | `lib/mixer.py` | 转换人声 + 伴奏 + 伴唱 + 混响 | 让输出直接成为可听的完整 AI cover，而不是只导出干声 |

---

### SOTA 口径

不同任务的 SOTA 不能直接横比。Music Source Separation、Vocal Pitch Estimation、Voice Conversion、Singing Voice Conversion、Singing Style Conversion 用的数据集、指标和听感评测都不同。本 README 使用以下口径：

| 口径 | 含义 |
|------|------|
| 当前默认 | 项目开箱即用时实际会走的模型或策略 |
| 高质量实用 | 在开源生态里效果强、可本地跑、和当前依赖兼容，适合作为默认或候选 |
| 研究前沿 | 论文、挑战赛或新框架里更先进的方向，但不一定能直接接入本项目 |
| 未集成 | README 可作为调研方向说明，但 UI、推理代码和模型格式暂未支持 |

| 模块 | 当前判断 | English |
|------|----------|---------|
| 伴奏/人声分离、去混响 | [RoFormer / Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 系路线很强，适合 RVC 翻唱前处理；但不能写成所有场景绝对 SOTA | Strong RoFormer-family separation for RVC cover preprocessing, but not an absolute SOTA claim for every benchmark |
| [RMVPE](https://arxiv.org/abs/2306.15412) 音高提取 | 仍是非常合理的默认，论文面向复调音乐人声 F0，报告了 RPA/RCA 与抗噪优势 | A well-supported default for vocal pitch estimation in polyphonic music |
| [HuBERT](https://arxiv.org/abs/2106.07447) + [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 工程成熟、可控、兼容现有角色模型；研究前沿上已不是 2026 绝对 SOTA | Practical and compatible with existing RVC character models, but no longer the research frontier in 2026 |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) / [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) / [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf) / [SYKI-SVC](https://arxiv.org/abs/2501.02953) / [S2Voice](https://arxiv.org/abs/2601.13629) | 更前沿的零样本、扩散、流匹配或大模型式 SVC/SSC 方向；不是当前 RVC `.pth` 的直接替换件 | Research-frontier VC/SVC/SSC systems that require architecture and model-format changes |

---

### 当前项目在用的模型

| 模型 | 位置 | 用途 | 状态 |
|------|------|------|------|
| [ensemble:vocal_rvc](https://pypi.org/project/audio-separator/) | `infer/separator.py` / [audio-separator==0.44.1](https://pypi.org/project/audio-separator/) | 默认人声分离预设，包含 [melband_roformer_big_beta6x.ckpt](https://pypi.org/project/audio-separator/) + [mel_band_roformer_vocals_fv4_gabox.ckpt](https://pypi.org/project/audio-separator/)，算法 `avg_wave` | 使用中 |
| [ensemble:karaoke](https://pypi.org/project/audio-separator/) | `infer/separator.py` / [audio-separator==0.44.1](https://pypi.org/project/audio-separator/) | 默认卡拉OK分离预设，包含 [mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt](https://pypi.org/project/audio-separator/) + [mel_band_roformer_karaoke_gabox_v2.ckpt](https://pypi.org/project/audio-separator/) + [mel_band_roformer_karaoke_becruily.ckpt](https://pypi.org/project/audio-separator/)，算法 `avg_wave` | 使用中 |
| [dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt](https://pypi.org/project/audio-separator/) | `infer/separator.py` | 严格 DeEcho / 去混响 | 使用中 |
| [htdemucs_ft](https://github.com/facebookresearch/demucs) | `configs/config.json` / [Demucs](https://github.com/facebookresearch/demucs) | 当前 Demucs 可选默认值 | 可选 |
| [HP2_all_vocals.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/HP2_all_vocals.pth) | `configs/config.json` / `tools/download_models.py` | UVR5 主人声模型；同时在 `REQUIRED_MODELS` 下载清单中 | 可选 / 需要基础下载 |
| [HP3_all_vocals.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/HP3_all_vocals.pth) / [HP5_only_main_vocal.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/HP5_only_main_vocal.pth) | `tools/download_models.py` | UVR5 主人声模型 | 可选下载 |
| [VR-DeEchoNormal.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/VR-DeEchoNormal.pth) / [VR-DeEchoAggressive.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/VR-DeEchoAggressive.pth) / [VR-DeEchoDeReverb.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/VR-DeEchoDeReverb.pth) | `tools/download_models.py` | 旧版 UVR DeEcho / DeReverb | 可选下载 |
| [onnx_dereverb_By_FoxJoy/vocals.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx) | `tools/download_models.py` | 旧版 ONNX 去混响 | 可选下载 |
| [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) | `tools/download_models.py` | [HuBERT](https://arxiv.org/abs/2106.07447) 内容特征 | 需要下载 |
| [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) | `tools/download_models.py` | [RMVPE](https://arxiv.org/abs/2306.15412) 音高提取 | 需要下载 |
| [_official_rvc/](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | `tools/download_models.py` / `run.py` | 默认 `use_official=true` 时使用的内置官方 RVC 源码 | 需要准备；脚本会 clone，失败即停止 |
| [f0G48k.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/pretrained_v2/f0G48k.pth) / [f0D48k.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/pretrained_v2/f0D48k.pth) / [f0G40k.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/pretrained_v2/f0G40k.pth) / [f0D40k.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/pretrained_v2/f0D40k.pth) | `tools/download_models.py` | [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 训练相关预训练权重 | 可选下载 |

---

### 人声分离与去混响模型

本项目当前默认使用 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 的 ensemble 预设，而不是旧版单一 ckpt。RoFormer 子模型没有稳定的逐 ckpt 论文页或仓库页时，表格链接指向 `audio-separator` 的公开模型来源；对应架构论文单独列在后面的研究依据里。以下模型名以当前代码和本地 `audio-separator==0.44.1` 包内预设为准。

| 模型/预设 | 类型 | 用途与接入状态 | 取舍 |
|-----------|------|----------------|------|
| [ensemble:vocal_rvc](https://pypi.org/project/audio-separator/) | RoFormer ensemble | 默认 AI cover / RVC 前处理；使用中 | 偏向干净主唱，适合后续 RVC；不是普通听歌分离的唯一最优 |
| [melband_roformer_big_beta6x.ckpt](https://pypi.org/project/audio-separator/) | [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) | `vocal_rvc` 子模型；使用中 | 高质量人声分离候选，作为 ensemble 的一部分 |
| [mel_band_roformer_vocals_fv4_gabox.ckpt](https://pypi.org/project/audio-separator/) | [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) | `vocal_rvc` 子模型；使用中 | 与 Beta6X 平均融合，降低单模型偏差 |
| [ensemble:karaoke](https://pypi.org/project/audio-separator/) | RoFormer ensemble | 主唱/伴唱分离；使用中 | 三模型 ensemble，适合和声明显的歌曲 |
| [mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt](https://pypi.org/project/audio-separator/) | [Mel-Band RoFormer karaoke](https://arxiv.org/abs/2310.01809) | `ensemble:karaoke` 子模型；使用中 | 公开带 SDR 标识的 karaoke 候选 |
| [mel_band_roformer_karaoke_gabox_v2.ckpt](https://pypi.org/project/audio-separator/) | [Mel-Band RoFormer karaoke](https://arxiv.org/abs/2310.01809) | `ensemble:karaoke` 子模型；使用中 | 和其他 karaoke 模型互补 |
| [mel_band_roformer_karaoke_becruily.ckpt](https://pypi.org/project/audio-separator/) | [Mel-Band RoFormer karaoke](https://arxiv.org/abs/2310.01809) | `ensemble:karaoke` 子模型；使用中 | 和其他 karaoke 模型互补 |
| [dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt](https://pypi.org/project/audio-separator/) | [Mel-Band RoFormer dereverb](https://arxiv.org/abs/2310.01809) | 去混响/去回声；使用中 | 用于严格 DeEcho 路径，目标是降低主唱回声进入 RVC |
| [vocal_balanced](https://pypi.org/project/audio-separator/) | RoFormer ensemble | 泛用高质量人声分离参考；未接入 UI 默认 | `audio-separator` 预设，偏整体平衡；适合未来做 A/B 候选 |
| [vocal_clean](https://pypi.org/project/audio-separator/) | RoFormer ensemble | 更少伴奏泄漏的人声；未接入 UI 默认 | 可能牺牲部分和声、尾音和气声 |
| [vocal_full](https://pypi.org/project/audio-separator/) | RoFormer ensemble | 尽量保留完整人声与和声；未接入 UI 默认 | 可能带来更多伴奏残留 |
| [htdemucs_ft](https://github.com/facebookresearch/demucs) | [Hybrid Demucs](https://arxiv.org/abs/2111.03600) | 分离后端对比；可选 | Demucs 曾是 2021 Music Demixing Challenge 优胜路线；现在更多作为稳定备选 |
| [UVR5 / VR DeEcho 系列](https://github.com/Anjok07/ultimatevocalremovergui) | UVR/VR | 老模型兼容与对照；可选 | 保留用于旧工作流和对比，不作为质量优先默认 |

研究依据上，[Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) 论文报告了它在 MUSDB18HQ 上对 BS-RoFormer 的提升；[audio-separator](https://pypi.org/project/audio-separator/) 的 `vocal_rvc` 预设明确面向 RVC / AI voice training 数据；[MVSEP](https://mvsep.com/en/algorithms) 一类平台的高质量 ensemble 往往还会组合 BS-RoFormer、MelBand RoFormer、SCNet 等更多模型。因此，本项目默认可以称为“RVC 翻唱场景下的高质量开源实用路线”，但不写成“所有数据集绝对第一”。

---

### 语音转换模型：RVC v2 与未来方向

当前项目使用 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)（Retrieval-based Voice Conversion v2）进行人声音色转换。它不是 2026 研究意义上的最新 SOTA，但仍是本项目默认，因为它和现有角色 `.pth` / `.index` 模型兼容，推理速度快，本地部署成本低，参数可控，适合普通用户做 AI cover。

| 项目 | 详情 |
|------|------|
| 模型全称 | Retrieval-based Voice Conversion v2 |
| 来源 | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| 架构 | HuBERT 特征提取 → F0 条件 → 生成器 + FAISS 索引检索 |
| 特征提取器 | [HuBERT Base](https://arxiv.org/abs/2106.07447)（[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt)） |
| 推理权重 | 用户选择的 RVC `.pth` 声线模型 |
| 索引文件 | 可选 `.index`，通过 FAISS 做检索增强 |
| 当前默认路由 | `vc_pipeline_mode=current` + `use_official=true`，先使用官方兼容 VC 推理，再走项目当前后处理 |
| 许可证 | MIT |

#### 同领域语音转换框架对比

| 框架/方向 | 状态 | 架构/特点 | 与本项目关系 |
|-----------|------|-----------|--------------|
| [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 当前采用 | HuBERT + F0 + FAISS 检索增强生成 | 默认，兼容现有 181 个可下载角色模型体系 |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) / [VITS](https://arxiv.org/abs/2106.06103) 系 SVC | 开源常见路线 | VITS/扩展 VITS | 可作为同类参考，不直接兼容 RVC `.pth` |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) | 研究/开源前沿 | 零样本 VC / SVC，部分版本使用 DiT、Whisper/内容编码器与 BigVGAN | 零样本和咬字等能力更前沿；不是当前 RVC 模型的直接替换 |
| [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) / [Vevo 1.5](https://huggingface.co/amphion/Vevo) | 研究前沿 | 大模型式 VC/SVC 基线，SVCC 2025 中被多个系统使用或微调 | 未来可调研；需要新推理栈和新模型格式 |
| [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf) / [SYKI-SVC](https://arxiv.org/abs/2501.02953) | 研究前沿 | 扩散或歌声转换系统 | 可作为未来 SVC/SSC 方向参考 |
| [S2Voice](https://arxiv.org/abs/2601.13629) | 研究前沿 | 基于 Vevo 思路的歌唱风格转换系统，面向 SVCC 2025 | 偏 singing style conversion，不是普通 RVC cover 的直接替换 |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 开源常见路线 | GPT + VITS few-shot | 更偏 TTS / 语音克隆，不是本项目当前 AI cover 主线 |
| [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) | 开源常见路线 | DDSP / 神经声码器方向 | 轻量实时方向参考，不是本项目当前默认 |

English note: replacing [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) with [Seed-VC](https://github.com/Plachtaa/seed-vc), [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md)-like systems, [Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf), [SYKI-SVC](https://arxiv.org/abs/2501.02953), or [S2Voice](https://arxiv.org/abs/2601.13629) would be a major architecture change. It would require new model formats, new inference code, new UI defaults, new licensing checks, and a migration plan for existing character models.

> **结论**：在“已有角色模型 + 本地可跑 + 默认不用手调 + 快速生成翻唱”的工作流里，[RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 仍然是工程上最合适的默认。若目标改成“论文/竞赛意义上的 2026 绝对 SOTA”，就需要另起一条 [Seed-VC](https://github.com/Plachtaa/seed-vc) / [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) / 扩散或流匹配式 SVC 管线。

---

### F0 提取模型：RMVPE

使用 [RMVPE](https://arxiv.org/abs/2306.15412) 从人声中提取基频（F0）曲线，用于保持转换后的音高/旋律。项目配置里根级和 `cover.f0_method` 都默认是 `rmvpe`，`f0_hybrid_mode` 默认是 `off`；翻唱链路会显式拒绝 `hybrid` / fallback 配置，避免把呼吸、换气、齿音、气声尾音错误写成强音高，造成电音化或机械感。

| 项目 | 详情 |
|------|------|
| 模型全称 | Robust Model for Vocal Pitch Estimation in Polyphonic Music |
| 论文 | [arXiv:2306.15412](https://arxiv.org/abs/2306.15412) |
| 检查点 | [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) |
| 核心优势 | 直接从多声道混音中提取人声音高，噪声鲁棒性强 |
| 指标 | 论文报告在 RPA/RCA 等指标上优于 [CREPE](https://github.com/marl/crepe)、pYIN、SWIPE、[Harvest](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf) 等基线 |

#### 同领域 F0 提取模型对比

| 模型 | 来源 | 说明 | 本项目用法 |
|------|------|------|------------|
| [RMVPE](https://arxiv.org/abs/2306.15412) | Dream-High | 面向复调音乐人声 F0，兼顾精度与抗噪 | 默认主路径 |
| [CREPE](https://github.com/marl/crepe) | NYU MARL | 经典 CNN 方案，生态成熟 | 只适合作为保守补洞参考，不适合无门控覆盖呼吸、齿音和无声音段 |
| [Harvest](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf) | [WORLD](https://github.com/mmorise/World) | 传统信号处理方案，部署简单 | 可用于对照，不作为默认 |
| [FCPE 等后续 F0 方向](https://arxiv.org/html/2509.15140) | 研究 / 社区路线 | 可能在部分数据集或实时场景有优势 | 未来可评估，当前未作为默认 |

> **结论**：质量优先时默认使用 [RMVPE](https://arxiv.org/abs/2306.15412) 严格路线。[CREPE](https://github.com/marl/crepe) 只保留为研究、诊断或非默认实验方向，不参与默认一键翻唱链路。

---

### 特征提取模型：HuBERT Base

| 项目 | 详情 |
|------|------|
| 模型全称 | [Hidden-Unit BERT](https://arxiv.org/abs/2106.07447) |
| 来源 | [Meta AI / fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) |
| 检查点 | [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) |
| 用途 | 提取语音内容特征，供 RVC 生成器使用 |
| 当前约束 | [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 架构和现有角色模型绑定 [HuBERT Base](https://arxiv.org/abs/2106.07447)；不能直接把 `.pth` 模型切到其他 encoder |

#### 特征模型对比

| 特征/编码器 | 说明 | 与当前项目关系 |
|-------------|------|----------------|
| [HuBERT Base](https://arxiv.org/abs/2106.07447) | RVC v2 的经典内容特征基础 | 当前默认，兼容现有角色模型 |
| [ContentVec](https://proceedings.mlr.press/v162/qian22b.html) | 在 HuBERT 基础上强调内容和说话人解耦 | 新框架中可能更合适，但不能直接替换现有 RVC v2 模型输入 |
| [WavLM](https://github.com/microsoft/unilm/blob/master/wavlm/README.md) | 更强的通用语音表征之一 | 可作为未来 VC/SVC 架构参考 |
| [Whisper encoder](https://github.com/openai/whisper) | 在部分零样本 VC/SVC 系统中用于内容或语义条件 | Seed-VC 等方向会用到，不是当前 RVC `.pth` 直接可用输入 |
| [离散语音 tokenizer / codec](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) | [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md)、[S2Voice](https://arxiv.org/abs/2601.13629) 等大模型式系统常见方向 | 属于未来架构迁移，不是当前链路的小参数调整 |

---

### 研究依据与维护原则

| 来源 | 类型 | 本 README 采用的结论 |
|------|------|----------------------|
| [Mel-Band RoFormer for Music Source Separation](https://arxiv.org/abs/2310.01809) | 论文 | Mel-RoFormer 在 MUSDB18HQ 上报告了对 BS-RoFormer 的提升，是 RoFormer / Mel-Band RoFormer 分离路线的重要依据 |
| [Mel-RoFormer for Vocal Separation and Vocal Melody Transcription](https://arxiv.org/abs/2409.04702) | 论文 | 继续说明 Mel-RoFormer 可同时服务人声分离和旋律相关任务，支持本项目把分离与 F0 质量一起看 |
| [Sound Demixing Challenge 2023](https://transactions.ismir.net/articles/10.5334/tismir.171) | 挑战赛论文 | 说明音乐拆分任务的 SOTA 依赖统一数据集、评分协议和听感测试，不能和 VC/SVC 指标混排 |
| [Hybrid Demucs](https://arxiv.org/abs/2111.03600) | 论文 | Demucs 曾在音乐拆分挑战中表现很强，但现在更多作为稳定备选或对照 |
| [audio-separator](https://pypi.org/project/audio-separator/) | 工程来源 | 本项目默认的 RoFormer ensemble、`vocal_rvc`、`karaoke` 等预设来自这个推理框架和模型表 |
| [MVSEP algorithms](https://mvsep.com/en/algorithms) | 榜单 / 模型页 | 高质量通用分离 ensemble 往往会组合 BS-RoFormer、MelBand RoFormer、SCNet 等模型，因此本项目不把 `vocal_rvc` 写成所有场景绝对第一 |
| [RVC 官方 README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md) | 上游项目 | RVC v2 的 HuBERT 特征、F0 条件和检索增强是本项目当前 VC 主线 |
| [RMVPE Interspeech 2023](https://www.isca-archive.org/interspeech_2023/wei23b_interspeech.html) / [arXiv](https://arxiv.org/abs/2306.15412) | 论文 | RMVPE 面向复调音乐人声 F0，适合 AI cover 的音高提取需求 |
| [HuBERT](https://arxiv.org/abs/2106.07447) | 论文 | HuBERT 是 RVC v2 内容特征基础 |
| [ContentVec](https://arxiv.org/abs/2204.09224) | 论文 | 说明后续研究仍在改进内容特征和说话人解耦 |
| [Seed-VC paper](https://arxiv.org/html/2411.09943v1) / [Seed-VC repo](https://github.com/Plachtaa/seed-vc) | 论文 / 开源项目 | 代表更前沿的零样本 VC / SVC 路线，公开资料包含与 RVCv2 的 singing voice conversion 对比 |
| [Singing Voice Conversion Challenge 2025](https://www.vc-challenge.org/) / [SVCC 2025 论文](https://arxiv.org/pdf/2509.15629) | 挑战赛 / 论文 | 歌声转换研究已经从 singer identity conversion 进一步走向 singing style conversion |
| [S2Voice](https://arxiv.org/html/2601.13629) | 论文 | 代表 SVCC 2025 后续的歌唱风格转换前沿方向 |
| [SI-SDR](https://arxiv.org/abs/1811.02508) / [museval](https://github.com/sigsep/sigsep-mus-eval) | 评估指标 / 工具 | 分离模型的量化对比需要参考 stem 和统一评估协议，不能只靠文件名或单曲听感判断 |

维护原则：

- README 只把当前代码真正支持的模型写成“使用中”。
- 研究前沿模型可以写入“未来方向”，但不能写成当前默认或可直接替换。
- 分离、F0、VC、SVC、SSC 的分数不混排成总榜。
- 如果未来接入 [Seed-VC](https://github.com/Plachtaa/seed-vc)、[Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) 类系统，应新增独立推理后端、模型下载策略、许可证说明和 A/B 评估，而不是把 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的 `.pth` 模型伪装成通用格式。

## 参数说明

### 转换参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| 音调偏移 | 半音数，正数升调，负数降调 | 男转女: +12, 女转男: -12 |
| F0 提取方法 | 音高提取算法 | rmvpe（默认严格路线）；`f0_hybrid_mode=off` |
| 索引比率 | 越高越像训练音色 | 0.1-0.5 (10-50%) |
| 滤波半径 | 中值滤波，减少气音抖动 | 3 |
| 保护系数 | 防止撕裂伪影，越小保护越强 | 0.33 |
| RMS 混合率 | 音量包络匹配程度 | 0.0（默认）；需要贴近源动态时再少量提高 |

### 混音参数（翻唱）

| 参数 | 说明 | 建议值 |
|------|------|--------|
| 人声音量 | 转换后人声的音量 | 100% |
| 伴奏音量 | 背景伴奏的音量 | 100% |
| 人声混响 | 为人声添加空间感 | 10-20% |
| 伴唱混合率 | 伴唱在最终输出中的比例 | 0-100% |

### 混音预设

| 预设 | 人声音量 | 伴奏音量 | 混响 | 说明 |
|------|---------|---------|------|------|
| 通用 | 100% | 100% | 0% | 默认均衡设置 |
| 人声突出 | 115% | 90% | 0% | 突出人声，适合清唱风格 |
| 伴奏突出 | 90% | 115% | 0% | 突出伴奏，适合背景音乐丰富的歌曲 |
| 现场感 | 100% | 100% | 10% | 增加混响，模拟现场演出效果 |

### VC 预处理模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| 自动 | 使用当前默认的严格 RoFormer De-Reverb 路径 | 推荐默认；失败会显式报错，不用静默降级掩盖问题 |
| 严格 RoFormer De-Reverb | 明确指定 RoFormer De-Reverb | 需要去除混响和回声，且希望固定使用同一条预处理路径 |

### 源约束策略

| 模式 | 说明 |
|------|------|
| 自动 | 根据场景自动决定 |
| 关闭 | 不使用源约束 |
| 启用 | 强制启用源约束 |

### VC 管道模式

| 模式 | 说明 | 特点 |
|------|------|------|
| 当前实现 | 使用项目自定义 VC 流程 | 支持完整的预处理和后处理 |
| 官方实现 | 使用内置官方 RVC 路线 | 强制官方 UVR5 分离 + RoFormer De-Reverb + 官方 VC；关闭 Karaoke 与当前项目源约束/静音门限后处理，适合做 A/B 对照 |

### 人声分离参数 (config.json)

| 参数 | 说明 | 建议值 |
|------|------|--------|
| separator | 分离器类型 | roformer（推荐）、uvr5 或 demucs |
| uvr5_model | UVR5 模型 | [HP2_all_vocals](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/HP2_all_vocals.pth) |
| uvr5_agg | UVR5 激进度 (1-10) | 6-8（高音问题可降低） |
| demucs_model | Demucs 模型 | [htdemucs](https://github.com/facebookresearch/demucs) |
| karaoke_model | 卡拉OK分离模型 | [ensemble:karaoke](https://pypi.org/project/audio-separator/) |

### 分离质量评估

真实量化指标需要参考 stem。项目提供 `tools/evaluate_karaoke_models.py` 用于对比本地 Karaoke 模型：

```powershell
python tools/evaluate_karaoke_models.py --vocals-path vocals.wav --output-dir outputs/karaoke_eval
```

无参考 stem 时，报告里的 `score` 只是诊断代理分数，用于检查重建误差、主唱/伴唱相关性、能量比例和长度覆盖率，不能代表最终听感。若有人工标注或数据集参考 stem，可加入参考主唱/伴唱，此时报告会输出论文中常用的 SI-SDR / SDR：

```powershell
python tools/evaluate_karaoke_models.py `
  --vocals-path vocals.wav `
  --reference-lead refs/lead.wav `
  --reference-backing refs/backing.wav `
  --output-dir outputs/karaoke_eval
```

实践建议：当前默认使用偏 RVC 翻唱前处理的 [ensemble:vocal_rvc](https://pypi.org/project/audio-separator/) 和 [ensemble:karaoke](https://pypi.org/project/audio-separator/)。若某首歌出现主唱变薄、和声泄漏或伴奏残留，可以在评估工具中加入参考 stem 使用 [SI-SDR](https://arxiv.org/abs/1811.02508) / SDR 排名，再对 [vocal_rvc](https://pypi.org/project/audio-separator/)、[vocal_balanced](https://pypi.org/project/audio-separator/)、[vocal_clean](https://pypi.org/project/audio-separator/)、[vocal_full](https://pypi.org/project/audio-separator/) 或单个 karaoke 模型做 A/B；不要只凭模型名里的分数判断最终听感。

## 配置文件

主要配置在 `configs/config.json`：

```json
{
  "device": "cuda",
  "f0_method": "rmvpe",
  "index_rate": 0.1,
  "filter_radius": 1,
  "protect": 0.28,
  "f0_hybrid_mode": "off",
  "use_official_vc": true,
  "cover": {
    "separator": "roformer",
    "roformer_model": "ensemble:vocal_rvc",
    "karaoke_separation": true,
    "karaoke_model": "ensemble:karaoke",
    "use_official": true,
    "uvr5_model": "HP2_all_vocals",
    "uvr5_agg": 10,
    "demucs_model": "htdemucs_ft",
    "f0_method": "rmvpe",
    "f0_hybrid_mode": "off",
    "index_rate": 0.50,
    "filter_radius": 3,
    "protect": 0.33,
    "rms_mix_rate": 0.0,
    "backing_mix": 0.0,
    "vc_preprocess_mode": "auto",
    "source_constraint_mode": "auto",
    "vc_pipeline_mode": "current"
  }
}
```

## 可用角色模型（100+，当前清单 181）

| 系列 | 角色示例 |
|------|----------|
| Love Live! | 星空凛、园田海未、东条希、小泉花阳、南小鸟 |
| Love Live! Sunshine!! | 高海千歌、樱内梨子、黑泽黛雅、黑泽露比、国木田花丸、津岛善子、小原鞠莉、渡边曜、松浦果南 |
| Love Live! 虹咲学园 | 上原步梦、中须霞、天王寺璃奈、近江彼方、优木雪菜、三船栞子、米雅·泰勒 |
| Love Live! Superstar!! | 唐可可、平安名堇 |
| 偶像大师 | 神崎兰子、梦见莉亚梦、双叶杏、本田未央、岛村卯月 |
| 原神 | 芙宁娜、枫原万叶、纳西妲、八重神子、雷电将军 |
| 碧蓝航线 | 埃塞克斯 |
| Hololive | Fuwawa、Mococo |
| 原创 | 爱美 (Aimi) |

> 完整列表请在 UI 中查看「下载角色模型」面板

## 项目结构

```
AI-RVC/
├── venv310/                 # 虚拟环境 (Python 3.10)
├── assets/                  # 模型文件
│   ├── hubert/              # HuBERT 模型 (~190 MB)
│   ├── rmvpe/               # RMVPE 模型
│   ├── uvr5_weights/        # UVR5 人声分离模型
│   ├── separator_models/    # Roformer 人声分离模型 (自动下载)
│   └── weights/             # 用户语音模型
│       └── characters/      # 角色模型 (100+，自动下载)
├── configs/                 # 配置文件
│   └── config.json          # 主配置
├── infer/                   # 推理模块
│   ├── pipeline.py          # 自定义 RVC 推理管道
│   ├── cover_pipeline.py    # 翻唱流水线
│   ├── separator.py         # 人声分离 (Roformer/Demucs)
│   └── modules/             # 官方 VC 模块
│       ├── vc/              # 官方 VC 管道
│       └── uvr5/            # UVR5 人声分离
├── lib/                     # 核心库
│   ├── audio.py             # 音频处理
│   ├── mixer.py             # 混音模块
│   └── logger.py            # 日志系统
├── models/                  # 模型定义
├── tools/                   # 工具脚本
│   ├── download_models.py   # 必需模型与内置官方 RVC 源码准备
│   └── character_models.py  # 角色模型管理
├── ui/                      # Gradio 界面
├── outputs/                 # 输出文件
├── temp/                    # 临时文件
└── run.py                   # 主入口
```

## 常见问题

**Q: CUDA out of memory**

人声分离通常需要约 4GB 以上显存（取决于音频时长和模型），尝试：
- 关闭其他占用显存的程序
- 使用较短的音频（建议 < 5 分钟）
- 在 config.json 中切换 separator 为 demucs 或 uvr5

**Q: 首次运行很慢**

首次运行会自动下载模型文件（大小随模型版本变化），请耐心等待。

**Q: 高音断音/撕裂**

这通常是 F0 提取不稳定导致的，尝试：
- 降低 UVR5 激进度（`uvr5_agg`: 8 → 6-7）
- 降低保护系数（`protect`: 0.33 → 0.2）
- 增大滤波半径（`filter_radius`: 3 → 5）
- 使用更干净的输入音频

**Q: 转换后声音失真**

尝试：降低索引比率、调整音调偏移、使用更高质量的输入音频。

**Q: 角色模型下载失败**

检查网络连接，或手动下载：
```bash
python -c "from tools.character_models import download_character_model; download_character_model('rin')"
```

**Q: faiss AVX512 警告**

正常的回退机制，faiss 会自动使用 AVX2，不影响功能。

**Q: CUDA 不可用**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: torchaudio DLL 加载失败 / 路径相关报错**

项目路径中不能包含中文或特殊字符（如 `C:\新建文件夹\AI-RVC`），否则 PyTorch/torchaudio 的 C++ 库无法正确加载。请将项目放在纯英文路径下，例如 `C:\AI-RVC` 或 `D:\AI-RVC`。

## 数据核验说明（2026-07-04）

以下外部数据已在 2026-07-04 复核。README 中涉及模型定位、论文依据和“是否 SOTA”的判断以这些来源为准；若上游模型榜单变化，应先重新核验再改文档。

- [MVSEP 算法页](https://mvsep.com/en/algorithms)（Multisong 指标与模型分数）
- [audio-separator 公开模型表与预设说明](https://pypi.org/project/audio-separator/)
- [MVSEP 算法详情（KimberleyJensen 模型）](https://mvsep.com/algorithms/49)
- [Mel-Band RoFormer 论文](https://arxiv.org/abs/2310.01809)
- [Mel-RoFormer vocal separation / melody transcription 论文](https://arxiv.org/abs/2409.04702)
- [Sound Demixing Challenge 2023 论文](https://transactions.ismir.net/articles/10.5334/tismir.171)
- [Hybrid Demucs 论文](https://arxiv.org/abs/2111.03600)
- [SI-SDR 指标讨论（Le Roux et al., 2019）](https://arxiv.org/abs/1811.02508)
- [BSS Eval / museval 源分离评估工具链](https://github.com/sigsep/sigsep-mus-eval)
- [RVC 官方仓库与许可证](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RVC 官方英文说明](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md)
- [第三方模型聚合计数（voice-models 首页）](https://voice-models.com/)
- [RMVPE 论文](https://www.isca-archive.org/interspeech_2023/wei23b_interspeech.html)
- [RMVPE arXiv](https://arxiv.org/abs/2306.15412)
- [HuBERT 论文](https://arxiv.org/abs/2106.07447)
- [ContentVec 论文](https://proceedings.mlr.press/v162/qian22b.html)
- [Seed-VC 论文](https://arxiv.org/html/2411.09943v1)
- [Seed-VC 仓库与评测说明](https://github.com/Plachtaa/seed-vc)
- [Vevo 仓库说明](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md)
- [Vevo 预训练模型页](https://huggingface.co/amphion/Vevo)
- [Serenade 论文](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf)
- [SYKI-SVC 论文](https://arxiv.org/abs/2501.02953)
- [Singing Voice Conversion Challenge 2025](https://www.vc-challenge.org/)
- [SVCC 2025 论文](https://arxiv.org/pdf/2509.15629)
- [S2Voice 论文](https://arxiv.org/html/2601.13629)
- [FCPE 论文](https://arxiv.org/abs/2509.15140)
- [PyTorch 安装页面（当前 CUDA wheel 选择）](https://pytorch.org/get-started/locally/)

## 贡献

欢迎提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 许可证

MIT License

## 致谢

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 原始 RVC 项目
- [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) - 人声分离模型架构论文
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - 音源分离推理框架
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) - Roformer 预训练权重
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) - Ultimate Vocal Remover
- [Demucs](https://github.com/facebookresearch/demucs) - Meta 人声分离
- [RMVPE](https://arxiv.org/abs/2306.15412) - 高质量 F0 提取
- [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) - 语音特征提取
- [Gradio](https://gradio.app/) - Web 界面框架

## 免责声明

**重要提示：使用本软件前请仔细阅读以下声明**

1. **仅供学习研究**：本项目仅供学习、研究和个人娱乐用途，不得用于任何商业目的。

2. **禁止非法使用**：严禁使用本软件进行以下行为：
   - 未经授权模仿他人声音进行欺诈、诈骗
   - 制作虚假音频用于传播谣言或误导公众
   - 侵犯他人肖像权、名誉权或其他合法权益
   - 任何违反当地法律法规的行为

3. **版权声明**：
   - 使用本软件转换的音频版权归原作者所有
   - 用户需自行获取原始音频和模型的使用授权
   - 本项目内置的角色模型仅供技术演示，请勿用于商业用途

4. **用户责任**：用户对使用本软件产生的所有内容和后果承担全部责任。开发者不对任何滥用行为负责。

5. **无担保声明**：本软件按"原样"提供，不提供任何明示或暗示的担保。

**使用本软件即表示您已阅读、理解并同意以上声明。**
