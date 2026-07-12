# AI-RVC 一键 AI 翻唱 / RVC Voice Conversion WebUI

AI-RVC 是一个开源的 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 翻唱 WebUI。项目包含人声与伴奏分离、主唱与和声分离、F0 提取、RVC 音色转换、FAISS 检索和混音导出，可直接处理 MP3、WAV 和 FLAC 歌曲。

> 在线体验：[TelkNet AI 翻唱](https://telknet.cc/tools/ai-rvc)

**平台支持：Windows / Linux / WSL2 / Google Colab / Hugging Face Spaces**

## 默认模型速览

| 环节 | 当前默认 | 输入与输出 | 来源 |
|------|----------|------------|------|
| 人声 | [BS-RoFormer Leap XE 90 bands（pcunwa）](https://huggingface.co/pcunwa/BS-Roformer-Leap) | 统一 PCM 整曲 → `vocals.wav` | [MVSep 10178](https://mvsep.com/quality_checker/entry/10178)：Vocals SDR 11.7577、SI-SDR 11.3936 |
| 纯伴奏 | [BS PolarFormer public ONNX 62 bands（bgkb/ZFTurbo）](https://huggingface.co/bgkb/bs_polarformer) | 同一 PCM 整曲 → `accompaniment_without_harmony.wav` | [MVSep 10009](https://www.mvsep.com/quality_checker/entry/10009)：Instrumental SDR 18.0650、SI-SDR 17.9756 |
| 主唱 / 带和声伴奏 | `BS-Kar-Gabox_IS + BS-Kar-Frazer&Becruily + BS-Kar-Anvuew (AVG)` | 原始整曲 → `lead_vocals.wav` + `accompaniment.wav`；第二路为 `Back+Instrumental` | [MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205)：三个模型使用 `avg_wave` |
| 和声 | Leap 人声与 MVSep 9205 主唱差分 | `vocals.wav - lead_vocals.wav` → `backing_vocals.wav` | 与 TelkNet 生产链路一致；输出为纯和声，不含乐器 |
| 去混响 | [RoFormer De-Reverb](https://huggingface.co/anvuew/dereverb_mel_band_roformer) | 主唱 → 较干的人声 | `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` |
| 内容特征 | [HuBERT Base](https://arxiv.org/abs/2106.07447) | 人声 → RVC 内容特征 | `hubert_base.pt` |
| 音高 | [RMVPE](https://arxiv.org/abs/2306.15412) | 人声 → F0 曲线 | `rmvpe.pt` |
| 音色转换 | [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) + [FAISS](https://github.com/facebookresearch/faiss) | 主唱 + `.pth` / `.index` → 转换后人声 | 兼容现有 RVC 角色模型 |

MP3、FLAC 等非 WAV 输入会先统一解码为 44.1 kHz 双声道 PCM16，Leap XE 与 PolarFormer 使用同一份输入。PolarFormer 输出还会执行孤立声道饱和抑制。默认 Karaoke 路线只转换主唱；最终混音直接使用 MVSep 的 `Back+Instrumental`，不会再叠加 PolarFormer 纯伴奏。

![Windows 界面](docs/Windows界面.png)

## 项目定位与搜索关键词

本仓库提供完整的本地 RVC 翻唱推理流程，重点是现有 `.pth` / `.index` 角色模型的使用与管理。训练 RVC 模型、文本转语音、实时直播变声和零样本声音克隆不在当前 WebUI 的功能范围内。

GitHub Topics：

`rvc`, `rvc-v2`, `voice-conversion`, `ai-cover`, `song-cover`, `singing-voice-conversion`, `voice-changer`, `voice-cloning`, `vocal-separation`, `audio-separation`, `rmvpe`, `hubert`, `faiss`, `gradio`, `pytorch`, `colab`, `uvr`, `demucs`, `roformer`, `ai-music`

## 功能特点

- **完整翻唱流程**：上传歌曲后依次完成人声分离、主唱提取、去混响、F0 提取、RVC 推理和混音导出。
- **TelkNet 分离链路**：非 WAV 输入先统一解码为 PCM；Leap XE 90 提取人声，PolarFormer 62 提取纯伴奏并抑制孤立声道饱和，MVSep 9205 从原始整曲输出主唱与 `Back+Instrumental`，再由人声差分得到纯和声。
- **RVC v2 兼容**：支持角色 `.pth`、FAISS `.index`、多说话人模型和当前/官方两套 VC 路由。
- **角色模型管理**：注册表包含 181 个条目，支持筛选、搜索、下载、自定义导入和版本信息缓存。
- **可导出的中间结果**：成品、转换后人声、原始人声、主唱、纯和声、带和声伴奏、纯伴奏均可单独保存。
- **混音控制**：4 种预设，并提供人声、伴奏、混响和原主人声混入参数。
- **运行后端**：支持 CUDA、ROCm、XPU、DirectML、MPS 和 CPU 检测；具体模型仍受上游运行时和设备算子支持限制。
- **部署入口**：提供 Gradio WebUI、Windows/Linux 打包配置、Google Colab 和 Hugging Face Spaces 入口。

## 平台支持

| 平台 | 状态 | 安装方式 | 说明 |
|------|------|---------|------|
| Windows 10/11 (x64) | 支持 | 可执行文件 / 本地安装 | 便携包使用 CPU；本地安装可使用 CUDA 或 DirectML |
| Linux (Ubuntu/Debian) | 支持 | 可执行文件 / 本地安装 | GPU 环境需安装与驱动匹配的 PyTorch wheel |
| WSL2 | 支持 | 本地安装 | WebUI 默认地址为 `http://127.0.0.1:7860` |
| Google Colab | 支持 | Notebook | Notebook 创建独立 Python 3.10 环境 |
| Hugging Face Spaces | 支持 | Space | CPU 或付费 GPU 硬件 |
| macOS | 实验性 | 本地安装 | 代码可检测 MPS；默认分离模型组合尚未完整验证 |

## 快速开始

| 方式 | 本地环境 | 加速能力 |
|------|----------|----------|
| 可执行文件 | 不需要 Python | 便携包使用 CPU |
| Google Colab | 浏览器 + Google 账号 | 由 Colab 运行时提供 GPU |
| Hugging Face Spaces | 浏览器 | 取决于 Space 硬件 |
| 本地安装 | Python 3.10 | 可配置 CUDA、ROCm、XPU、DirectML、MPS 或 CPU |

### 推荐配置

默认高质量分离会依次运行 Leap XE、PolarFormer 和 MVSep 9205 的三个子模型。开启 Karaoke 时，单首歌曲仅分离阶段就包含 5 次整曲模型推理，耗时会明显高于旧单模型路线。

| 使用场景 | GPU | 系统内存 | CPU / 存储 | 说明 |
|----------|-----|----------|------------|------|
| 默认高质量路线 | NVIDIA CUDA，16GB 显存 | 64GB | 8 核以上；NVMe，至少 15GB 可用空间 | 适合 3～5 分钟歌曲，Karaoke 可保持开启 |
| 长音频或连续处理 | NVIDIA CUDA，24GB 显存以上 | 64GB 以上 | 12 核以上；NVMe，至少 30GB 可用空间 | 适合更长音频和连续任务；当前 WebUI 仍按任务串行处理 |
| 8～12GB 显存 | NVIDIA CUDA | 32GB 以上 | SSD，至少 15GB 可用空间 | 建议关闭 Karaoke，减少三次 MVSep 9205 推理；不要同时运行其他 GPU 程序 |
| CPU / 便携版 | 不需要 | 32GB 以上 | 8 核以上；SSD | 功能可运行，但 RoFormer/PolarFormer 整曲推理会很慢，不适合批量处理 |

PolarFormer 默认把单次窗口限制为 `441000` 个采样点，与 TelkNet 运行配置一致。需要进一步压低峰值显存时，可在启动前设置 `POLARFORMER_MAX_CHUNK_SIZE=220500`；窗口更小会增加分块数量和总耗时。设为 `0` 会取消限制并使用模型配置值，不建议在 24GB 以下显存上使用。

### 方式 1：可执行文件（无需安装 Python）

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

**运行说明**：
- 无需单独安装 Python 和项目依赖
- 仅支持 CPU 推理（构建时使用 CPU 版 PyTorch 以控制包体积）
- GPU 推理需使用方式 4 本地安装（`python install.py`）
- 首次启动会下载模型，耗时取决于网络和磁盘速度

### 方式 2：Google Colab

![Colab 演示](docs/Colab演示.png)

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

**运行说明**：
- 无需安装，直接使用
- 随时随地访问
- 易于分享

**限制**：
- 免费版使用 CPU（处理较慢）
- 可升级到 GPU（付费）

### 方式 4：本地安装

#### 一键安装

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
# - 启动 Web 界面（首次运行时会准备 HuBERT/RMVPE、默认分离模型和内置官方 RVC 源码）
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
- `--cpu`：安装 CPU 版本 PyTorch，并把运行配置明确写为 `device=cpu`；默认 GPU 安装会写为 `device=cuda`，设备不可用时不会自动切换
- `--no-run`：安装完成后不自动启动

> 脚本会自动创建 `venv310` 虚拟环境并在其中安装所有依赖。安装后手动启动请使用虚拟环境中的 Python：
> - Windows：`venv310\Scripts\python run.py`
> - Linux：`venv310/bin/python run.py`

访问 http://127.0.0.1:7860 打开界面。

首次运行或执行 `python tools/download_models.py` 时，项目会下载 HuBERT、RMVPE、UVR5 HP2、Leap XE vocals、BS PolarFormer public ONNX、MVSep 9205 子模型、RoFormer De-Reverb 和内置官方 RVC 源码；分离模型缓存在 `assets/separator_models/`。默认运行资源合计约 2～3GB，下载全部可选模型后可能超过 10GB。

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
# GPU：复制 PyTorch 页面生成的命令
# CPU：
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 准备必需模型、默认分离模型与内置官方 RVC 源码
python tools/download_models.py
# 仅检查状态：
# python tools/download_models.py --check
# 只准备默认分离模型：
# python tools/download_models.py --separator

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
# 在 https://pytorch.org/get-started/locally/ 生成并执行对应 CUDA、ROCm 或 CPU 命令
pip install -r requirements.txt

# 4. 准备必需模型、默认分离模型与内置官方 RVC 源码 + 启动
python tools/download_models.py
# 可选：python tools/download_models.py --check / --separator
python run.py
```

---

**Linux 兼容性说明**：
- 路径处理使用 `pathlib.Path`，虚拟环境入口按 `bin/python` 和 `Scripts/python.exe` 区分
- CUDA 取决于 NVIDIA 驱动与 PyTorch wheel；ROCm 取决于 AMD 驱动、系统版本和 PyTorch ROCm wheel
- `fairseq==0.12.2`、`pyworld`、`audio-separator[gpu]` 等依赖在不同 Linux 发行版上可能需要编译工具链和系统音频/FFmpeg 依赖
- `python3.10 install.py --check` 可检查 Python、依赖、设备和模型状态

**安装脚本说明**：
- `install.py` 会自动检测系统环境（Windows/Linux）并完成以下步骤：
  1. **检测 Python 3.10**：Windows 检查常见安装路径 + `py -3.10` 启动器；Linux 使用 `python3.10` 命令
  2. **创建虚拟环境**：在 `venv310/` 目录创建隔离的 Python 环境
  3. **安装 PyTorch**：自动检测 CUDA 可用性，安装对应版本（GPU/CPU）
  4. **安装项目依赖**：从 `requirements.txt` 安装所有必需包（包括 fairseq、audio-separator 等）
  5. **启动应用**：自动运行 `run.py` 启动 Web 界面（除非使用 `--no-run`）
- 必需模型（HuBERT、RMVPE、UVR5 HP2）、默认分离模型（Leap XE vocals、BS PolarFormer public ONNX、MVSep 9205、RoFormer De-Reverb）和内置官方 RVC 源码会在首次运行或 `python tools/download_models.py` 时准备；缺少 git、`audio-separator` 或 `_official_rvc/` 不完整会显式报错停止
- 支持参数：`--check`（仅检查）、`--cpu`（CPU 版本）、`--no-run`（不自动启动）
- 如果虚拟环境已存在，会跳过创建步骤，直接检查依赖

## 依赖版本说明

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10 | 安装脚本和 Colab 固定使用 3.10 |
| PyTorch | >= 2.0.0 | 语音转换 + 人声分离 |
| torchaudio | >= 2.0.0 | 与 PyTorch 版本对应 |
| CUDA / ROCm | 与 PyTorch wheel 和本机驱动匹配 | 可选 |
| fairseq | 0.12.2 | HuBERT 特征提取 |
| [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | 0.44.1（requirements 锁定） | 加载 RoFormer/BS-RoFormer `.ckpt`，用于 MVSep 9205 主唱 / `Back+Instrumental` ensemble、DeEcho 和旧预设对照 |
| [ONNX Runtime](https://onnxruntime.ai/) / [einops](https://github.com/arogozhnikov/einops) / [PyYAML](https://pyyaml.org/) | ONNX Runtime 由 `audio-separator[cpu/gpu/dml]` 安装对应版本 | 运行默认 [BS PolarFormer public ONNX 62 bands](https://huggingface.co/bgkb/bs_polarformer) 伴奏分离；不会同时安装 CPU 与 GPU Runtime |
| demucs | >= 4.0.0 | Demucs 人声分离（可选） |

`python install.py` 会按项目约束安装依赖。当前依赖栈使用 Gradio 5 与 NumPy 2，并固定 `audio-separator==0.44.1`。

## 使用方法

### 歌曲翻唱

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
   - 卡拉OK设置：启用 MVSep 9205 原曲主唱 / `Back+Instrumental` 分离
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
   - 纯和声轨道
   - 带和声伴奏（默认 MVSep Karaoke 路线）
   - 纯伴奏（PolarFormer）

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
          │  Leap XE vocals + PolarFormer pure accomp (默认) / UVR5 / Demucs│
          │      ↓                                         │
          │  人声 (vocals.wav) + 纯伴奏 (accompaniment_without_harmony.wav) │
          └────────────────────────────────────────────────┘
              ↓
          ┌─ 步骤 1.5：主唱 / 带和声伴奏分离（可选）────────┐
          │  MVSep 9205 avg_wave，输入仍是原始整曲          │
          │      ↓                                         │
          │  主唱 (lead_vocals.wav) + 带和声伴奏 (accompaniment.wav) │
          │  人声 - 主唱 → 纯和声 (backing_vocals.wav)      │
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
          │  转换人声 + 最终伴奏轨 → 音量调节 + 混响       │
          │      ↓                                         │
          │  AI 翻唱成品 (cover.wav)                       │
          └────────────────────────────────────────────────┘
```

### 使用的 AI 模型

当前运行时由六类模型组成：音源分离、去混响、内容特征、F0、RVC 生成器和 FAISS 索引。后文中的“未集成”表示论文或上游代码可查，但本项目没有对应的下载器、推理后端或 UI 入口。

术语：

- **VC（Voice Conversion）**：转换说话人或歌手音色。
- **SVC（Singing Voice Conversion）**：转换歌手身份，并保留歌词与旋律。
- **SSC（Singing Style Conversion）**：改变气声、颤音、滑音等演唱风格；任务定义与普通 SVC 不同。

---

### 默认质量链路

| 阶段 | 模型或实现 | 运行状态 | 输出 |
|------|------------|----------|------|
| 人声分离 | Leap XE 90 bands | 已集成，默认 | `vocals.wav` |
| 纯伴奏分离 | BS PolarFormer public ONNX 62 bands | 已集成，默认 | `accompaniment_without_harmony.wav` |
| 主唱 / 带和声伴奏 | MVSep 9205 三模型 `avg_wave` | 已集成，Karaoke 默认开启 | `lead_vocals.wav`、`accompaniment.wav` |
| 纯和声推导 | Leap 人声减去 MVSep 主唱 | 已集成，Karaoke 默认开启 | `backing_vocals.wav` |
| 去混响 | RoFormer De-Reverb | 已集成，`auto` 与 `uvr_deecho` 使用同一严格模型 | `vocals_for_vc.wav` |
| 内容与音高 | HuBERT Base + RMVPE | 已集成，默认 | 内容特征与 F0 |
| 音色转换 | RVC v2 + FAISS | 已集成，默认使用官方兼容推理和项目后处理 | `converted_vocals.wav` |
| 混音 | `lib/mixer.py` + pedalboard | 已集成 | `cover.wav` |

---

### SOTA 口径

音源分离、F0、VC、SVC 和 SSC 使用不同数据集与指标。README 不把 SDR、F0 准确率、说话人相似度、自然度 MOS 和风格相似度合并成总排名。

| 状态 | 定义 |
|------|------|
| 已集成 | 当前代码、配置、下载器和 UI 均可使用 |
| 可选 | 当前代码支持，但不属于默认路线 |
| 未集成 | 只有论文或上游实现，本项目尚无推理入口 |

公开榜单已有 [PolarFormer 124 bands](https://www.mvsep.com/quality_checker/entry/10147) 和 BS-RoFormer 2025.07 等更高分条目。本项目仍使用可核验公开权重和配置的 62-band PolarFormer ONNX；榜单模型名称不等于本地已有权重。

---

### 当前项目在用的模型

| 模型或资源 | 代码位置 | 用途 | 状态 |
|------------|----------|------|------|
| `hybrid:leap_xe90_vocals+polarformer62_instrumental` | `infer/separator.py` | 默认整曲人声 / 伴奏路由 | 已集成 |
| [bs_leap_xe_voc.ckpt](https://huggingface.co/pcunwa/BS-Roformer-Leap) | `assets/separator_models/Xe/` | Leap XE 90 人声输出 | 已集成 |
| [bs_polarformer.onnx](https://huggingface.co/bgkb/bs_polarformer) | `assets/separator_models/bs_polarformer/` | PolarFormer 62 纯伴奏输出；默认窗口上限 `441000` | 已集成 |
| [ensemble:mvsep_9205_avg](https://www.mvsep.com/quality_checker/entry/9205) | `infer/separator.py` | Gabox_IS、Frazer&Becruily、Anvuew 三模型主唱 / `Back+Instrumental` 分离 | 已集成 |
| [RoFormer De-Reverb](https://huggingface.co/anvuew/dereverb_mel_band_roformer) | `infer/separator.py` | VC 前去混响 | 已集成 |
| [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) | `assets/hubert/` | RVC 内容特征 | 已集成 |
| [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) | `assets/rmvpe/` | F0 提取 | 已集成 |
| [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | `_official_rvc/`、`infer/pipeline.py` | `.pth` / `.index` 音色转换 | 已集成 |
| 181 项角色模型注册表 | `tools/character_models.py` | 下载、导入、筛选和版本信息 | 已集成 |
| [htdemucs_ft](https://github.com/facebookresearch/demucs) | `configs/config.json` | Demucs 分离后端 | 可选 |
| [HP2_all_vocals.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/HP2_all_vocals.pth) | `assets/uvr5_weights/` | UVR5 分离后端 | 可选 |
| `ensemble:vocal_rvc`、`ensemble:karaoke` | `infer/separator.py` | 旧 RoFormer ensemble 回归对照 | 可选 |
| RVC v2 `f0G*` / `f0D*` 预训练权重 | `tools/download_models.py` | RVC 训练资源；当前 WebUI 不提供训练流程 | 可选下载 |

---

### 人声分离与去混响模型

默认人声、纯伴奏和 Karaoke 两路都基于原始整曲。非 WAV 输入先统一解码为 44.1 kHz 双声道 PCM16，Leap XE 和 PolarFormer 读取同一份 PCM；PolarFormer 在相减前抑制孤立声道饱和。`accompaniment.wav` 是 MVSep 9205 的 `Back+Instrumental`，`accompaniment_without_harmony.wav` 是 PolarFormer 纯伴奏，`backing_vocals.wav` 则是 Leap 人声减去 MVSep 主唱得到的纯和声。

| 模型或路线 | 任务 | 状态 | 公开依据 |
|------------|------|------|----------|
| Leap XE 90 | 人声 | 已集成，默认 | [MVSep 10178](https://mvsep.com/quality_checker/entry/10178) |
| PolarFormer public ONNX 62 | 纯伴奏 | 已集成，默认 | [ZFTurbo v1.0.20](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.20)、[ONNX 模型页](https://huggingface.co/bgkb/bs_polarformer)、[MVSep 10009](https://mvsep.com/quality_checker/entry/10009) |
| MVSep 9205 `avg_wave` | 主唱 / `Back+Instrumental` | 已集成，默认 | [Quality Checker 9205](https://www.mvsep.com/quality_checker/entry/9205) |
| RoFormer De-Reverb | 去混响 | 已集成，默认 | [模型页](https://huggingface.co/anvuew/dereverb_mel_band_roformer) |
| `htdemucs_ft` | 人声 / 伴奏 | 可选 | [Demucs](https://github.com/facebookresearch/demucs)、[Hybrid Demucs](https://arxiv.org/abs/2111.03600) |
| UVR5 HP2 | 人声 / 伴奏 | 可选 | [RVC UVR5 权重](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) |
| PolarFormer 124 bands | 人声 / 伴奏 | 未集成 | [MVSep 10147](https://www.mvsep.com/quality_checker/entry/10147) 有榜单结果；本仓库没有对应权重和运行配置 |
| BS-RoFormer 2025.07 | 人声 / 伴奏 | 未集成 | [MVSep 算法页](https://mvsep.com/algorithms/34) 有结果；本仓库没有可核验的对应权重 |

MVSep 的 SDR、SI-SDR、bleedless 和 fullness 来自指定测试集。它们用于同任务、同协议下的比较，不代表每首输入歌曲都能达到相同数值。

---

### 语音转换模型：RVC v2 与未来方向

当前项目只集成 [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 推理。现有角色资产使用 RVC `.pth` 和 FAISS `.index`；其他 VC、SVC 或 SSC 模型不能直接载入这套运行时。

| 项目 | 详情 |
|------|------|
| 模型全称 | Retrieval-based Voice Conversion v2 |
| 来源 | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| 架构 | HuBERT 特征提取 → F0 条件 → 生成器 + FAISS 索引检索 |
| 特征提取器 | [HuBERT Base](https://arxiv.org/abs/2106.07447)（[hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt)） |
| 推理权重 | 用户选择的 RVC `.pth` 声线模型 |
| 索引文件 | 可选 `.index`，通过 FAISS 做检索增强 |
| 当前默认路由 | `vc_pipeline_mode=current` + `use_official=true`；官方兼容 VC 推理后执行项目后处理 |
| 许可证 | MIT |

#### 同领域语音转换框架对比

| 框架 | 主要任务与技术 | 上游状态（2026-07-11） | 本项目状态 |
|------|----------------|-------------------------|------------|
| [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | HuBERT、F0、VITS 系生成器、FAISS 检索 | MIT；官方仓库仍可访问 | 已集成 |
| [so-vits-svc 4.1](https://github.com/svc-develop-team/so-vits-svc) | 基于 VITS 的 SVC | AGPL-3.0；官方仓库已归档 | 未集成 |
| [Seed-VC](https://github.com/Plachtaa/seed-vc) | 零样本 VC / SVC；v1 使用扩散 Transformer 与 F0 条件，v2 增加 AR + CFM 路线 | GPL-3.0；官方仓库已归档；提供 44.1 kHz SVC 检查点 | 未集成 |
| [Vevo](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md) | 自监督离散 token、AR 内容-风格建模、flow-matching 声学模型 | ICLR 2025；Amphion 提供实现与权重 | 未集成 |
| [Vevo1.5](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevosing) | 统一语音与歌声生成；SVCC 2025 开源基线之一 | Amphion 提供实现 | 未集成 |
| [Vevo2](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2) | 统一韵律学习；prosody tokenizer、content-style tokenizer、Qwen2.5-0.5B AR、flow-matching Transformer、Vocos | 2026 年公开实现与预训练模型；支持 VC、SVC、SSC、编辑与旋律控制 | 未集成 |
| [S²Voice](https://arxiv.org/abs/2601.13629) | 基于 Vevo1.5；FiLM、风格 cross-attention、全局说话人条件、DPO | SVCC 2025 两个 SSC 赛道第一名系统；论文和演示公开 | 未集成 |
| [Serenade](https://github.com/lesterphillip/serenade) | 基于 audio infilling 的扩散式 SSC | EUSIPCO 2025；代码公开 | 未集成 |
| [SYKI-SVC](https://arxiv.org/abs/2501.02953) | ContentVec + Whisper 内容特征、F0 与高频后处理 | ICASSP 2025 论文；面向歌手身份转换 | 未集成 |

接入这些系统需要独立模型下载、配置解析、推理后端、显存策略、许可证检查和输出评估。RVC 角色 `.pth` 不能转换成上述模型的通用权重。

---

### F0 提取模型：RMVPE

项目使用 [RMVPE](https://arxiv.org/abs/2306.15412) 提取歌声 F0。根配置和 `cover.f0_method` 均为 `rmvpe`，`f0_hybrid_mode` 为 `off`；默认翻唱路线不组合多个 F0 估计器。

| 项目 | 详情 |
|------|------|
| 模型全称 | Robust Model for Vocal Pitch Estimation in Polyphonic Music |
| 论文 | [arXiv:2306.15412](https://arxiv.org/abs/2306.15412) |
| 检查点 | [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) |
| 适用任务 | 复调音乐中的人声音高估计 |
| 指标 | 论文报告在 RPA/RCA 等指标上优于 [CREPE](https://github.com/marl/crepe)、pYIN、SWIPE、[Harvest](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf) 等基线 |

#### 同领域 F0 提取模型对比

| 模型 | 任务与结构 | 状态 |
|------|------------|------|
| [RMVPE](https://github.com/Dream-High/RMVPE) | 复调音乐人声 F0；Mel 频谱与深度网络 | 已集成，默认 |
| [CREPE](https://github.com/marl/crepe) | 单音高估计；时域 CNN | 已集成，可选 |
| [Harvest](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf) | WORLD 传统 F0 估计 | 已集成，可选 |
| [FCPE](https://github.com/CNChTu/FCPE) | Fast Context-based Pitch Estimation；Lynx-Net 与深度可分离卷积 | 未集成；MIT 上游代码和检查点公开 |
| [SwiftF0](https://github.com/lars76/swift-f0) | 轻量单音高估计；STFT + 2D CNN，面向 CPU 实时处理 | 未集成；MIT 上游代码公开 |

RMVPE、FCPE 和 SwiftF0 的论文使用不同数据集、噪声条件和速度测量方法。未在同一评估协议下复测前，不在此给出跨论文排名。

---

### 特征提取模型：HuBERT Base

| 项目 | 详情 |
|------|------|
| 模型全称 | [Hidden-Unit BERT](https://arxiv.org/abs/2106.07447) |
| 来源 | [Meta AI / fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) |
| 检查点 | [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) |
| 用途 | 提取语音内容特征，供 RVC 生成器使用 |
| 模型约束 | 现有 RVC `.pth` 在训练时使用 HuBERT 特征；更换编码器需要重新训练转换模型 |

#### 特征模型对比

| 特征或表示 | 用途 | 本项目状态 |
|------------|------|------------|
| [HuBERT Base](https://arxiv.org/abs/2106.07447) | RVC v2 内容特征 | 已集成，默认 |
| [ContentVec](https://proceedings.mlr.press/v162/qian22b.html) | 弱化说话人信息的内容表示 | 未集成 |
| [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) | 通用自监督语音表示 | 未集成 |
| [Whisper encoder](https://github.com/openai/whisper) | 语义或内容条件；Seed-VC、SYKI-SVC 等系统使用 | 未集成 |
| 离散内容 / 风格 / 韵律 tokenizer | Vevo 系列用信息瓶颈拆分内容、音色、风格与韵律 | 未集成 |
| [ASTRAL-Quantization](https://github.com/Plachtaa/ASTRAL-quantization) | Seed-VC v2 使用的说话人解耦语音 tokenizer | 未集成 |

---

### 研究依据与维护原则

| 主题 | 一手来源 |
|------|----------|
| 音源分离架构 | [BS-RoFormer](https://arxiv.org/abs/2309.02612)、[Mel-Band RoFormer](https://arxiv.org/abs/2310.01809)、[Mel-RoFormer vocal separation](https://arxiv.org/abs/2409.04702)、[Hybrid Demucs](https://arxiv.org/abs/2111.03600) |
| 当前分离权重与指标 | [Leap XE](https://huggingface.co/pcunwa/BS-Roformer-Leap)、[PolarFormer release](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.20)、[PolarFormer ONNX](https://huggingface.co/bgkb/bs_polarformer)、[MVSep 10178](https://mvsep.com/quality_checker/entry/10178)、[10009](https://mvsep.com/quality_checker/entry/10009)、[9205](https://www.mvsep.com/quality_checker/entry/9205) |
| RVC 内容与音高 | [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)、[HuBERT](https://arxiv.org/abs/2106.07447)、[ContentVec](https://arxiv.org/abs/2204.09224)、[RMVPE](https://arxiv.org/abs/2306.15412) |
| 零样本 VC / SVC | [Seed-VC](https://arxiv.org/abs/2411.09943)、[Vevo](https://openreview.net/forum?id=anQDiQZhDP)、[Vevo2](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2) |
| 歌唱风格转换 | [SVCC 2025](https://www.vc-challenge.org/)、[挑战总结](https://arxiv.org/abs/2509.15629)、[S²Voice](https://arxiv.org/abs/2601.13629)、[Serenade](https://eusipco2025.org/wp-content/uploads/pdfs/0000411.pdf) |
| 歌声转换与后处理 | [SYKI-SVC](https://arxiv.org/abs/2501.02953) |
| 新 F0 方向 | [FCPE](https://arxiv.org/abs/2509.15140)、[SwiftF0](https://arxiv.org/abs/2508.18440) |
| 分离评估 | [SI-SDR](https://arxiv.org/abs/1811.02508)、[museval](https://github.com/sigsep/sigsep-mus-eval) |

维护原则：

- “已集成”必须能在当前代码、配置和测试中找到对应入口。
- 论文模型与榜单模型统一标为“未集成”，直到权重、许可证、推理代码和评估流程全部落地。
- 不跨任务比较分数；同一模型也应标明数据集、指标和评测版本。
- 模型页没有明确许可证时，README 不推断其商用授权。
- 更新前沿模型表时同步更新下方“数据核验说明”的日期与来源。

## 参数说明

### 转换参数

| 参数 | 说明 | 翻唱默认值 |
|------|------|------------|
| 音调偏移 | 半音数，正数升调，负数降调 | 0 |
| F0 提取方法 | 音高提取算法 | `rmvpe`；`f0_hybrid_mode=off` |
| 索引比率 | FAISS 检索特征混合比例 | 0.50 |
| 滤波半径 | F0 中值滤波半径 | 3 |
| 保护系数 | 清辅音与呼吸段保护 | 0.33 |
| RMS 混合率 | 源人声音量包络混合比例 | 0.0 |

### 混音参数（翻唱）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 人声音量 | 转换后人声的音量 | 100% |
| 伴奏音量 | PolarFormer 纯伴奏或 MVSep `Back+Instrumental` 的音量 | 100% |
| 人声混响 | 应用于转换后人声的混响量 | 0% |
| 原主唱混入 | 原主唱与转换后主唱的混合比例 | 0% |

### 混音预设

| 预设 | 人声音量 | 伴奏音量 | 混响 | 说明 |
|------|---------|---------|------|------|
| 通用 | 100% | 100% | 0% | 默认 |
| 人声突出 | 115% | 90% | 0% | 提高人声、降低伴奏 |
| 伴奏突出 | 90% | 115% | 0% | 降低人声、提高伴奏 |
| 现场感 | 100% | 100% | 10% | 增加人声混响 |

### VC 预处理模式

| 模式 | 说明 |
|------|------|
| 自动 | 使用严格 RoFormer De-Reverb；模型缺失或推理失败时停止处理 |
| 严格 RoFormer De-Reverb | 显式指定同一 RoFormer De-Reverb 路线 |

### 源约束策略

| 模式 | 说明 |
|------|------|
| 自动 | 严格 DeEcho 成功后启用源引导约束 |
| 关闭 | 不使用源约束 |
| 启用 | VC 预处理成功后启用源引导约束 |

### VC 管道模式

| 模式 | 说明 | 特点 |
|------|------|------|
| 当前实现 | 使用项目自定义 VC 流程 | 支持完整的预处理和后处理 |
| 官方实现 | 使用内置官方 RVC 路线 | 强制官方 UVR5 分离 + RoFormer De-Reverb + 官方 VC；关闭 Karaoke 与当前项目源约束/静音门限后处理，用于 A/B 对照 |

### 人声分离参数 (config.json)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| separator | 分离器类型 | `roformer` |
| roformer_model | 默认人声/纯伴奏分离模型 | `hybrid:leap_xe90_vocals+polarformer62_instrumental` |
| uvr5_model | UVR5 模型 | [HP2_all_vocals](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/uvr5_weights/HP2_all_vocals.pth) |
| uvr5_agg | UVR5 激进度（1-10） | 10 |
| demucs_model | Demucs 模型 | [htdemucs_ft](https://github.com/facebookresearch/demucs) |
| karaoke_model | 卡拉OK分离模型 | [ensemble:mvsep_9205_avg](https://www.mvsep.com/quality_checker/entry/9205) |

### 分离质量评估

真实量化指标需要参考 stem。项目提供 `tools/evaluate_karaoke_models.py` 用于对比本地 Karaoke 模型：

```powershell
python tools/evaluate_karaoke_models.py --vocals-path vocals.wav --output-dir outputs/karaoke_eval
```

无参考 stem 时，报告里的 `score` 只用于检查重建误差、主唱/第二路相关性、能量比例和长度覆盖率。提供人工标注或数据集参考 stem 后，报告会输出 SI-SDR / SDR：

```powershell
python tools/evaluate_karaoke_models.py `
  --vocals-path vocals.wav `
  --reference-lead refs/lead.wav `
  --reference-backing refs/backing.wav `
  --output-dir outputs/karaoke_eval
```

当前默认评估对象为 `hybrid:leap_xe90_vocals+polarformer62_instrumental` 和 [ensemble:mvsep_9205_avg](https://www.mvsep.com/quality_checker/entry/9205)。提供参考 stem 时，评估工具会计算 SI-SDR / SDR；没有参考 stem 时，报告只提供重建误差、相关性、能量比例和长度覆盖率等诊断值。

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
    "roformer_model": "hybrid:leap_xe90_vocals+polarformer62_instrumental",
    "karaoke_separation": true,
    "karaoke_model": "ensemble:mvsep_9205_avg",
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

## 可用角色模型（当前清单 181）

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
│   ├── separator_models/    # Leap XE / BS PolarFormer / RoFormer 分离模型 (自动下载)
│   └── weights/             # 用户语音模型
│       └── characters/      # 角色模型（当前注册表 181 项）
├── configs/                 # 配置文件
│   └── config.json          # 主配置
├── infer/                   # 推理模块
│   ├── pipeline.py          # 自定义 RVC 推理管道
│   ├── cover_pipeline.py    # 翻唱流水线
│   ├── separator.py         # 人声/纯伴奏、主唱/带和声伴奏分离与纯和声推导
│   └── modules/             # 官方 VC 模块
│       ├── vc/              # 官方 VC 管道
│       └── uvr5/            # UVR5 人声分离
├── lib/                     # 核心库
│   ├── audio.py             # 音频处理
│   ├── mixer.py             # 混音模块
│   └── logger.py            # 日志系统
├── models/                  # 模型定义
├── tools/                   # 工具脚本
│   ├── download_models.py   # 必需模型、默认分离模型与内置官方 RVC 源码准备
│   └── character_models.py  # 角色模型管理
├── ui/                      # Gradio 界面
├── outputs/                 # 输出文件
├── temp/                    # 临时文件
└── run.py                   # 主入口
```

## 常见问题

**Q: CUDA out of memory**

默认高质量路线建议使用 16GB 显存。出现显存不足时：

- 关闭浏览器硬件加速、游戏和其他 GPU 程序后重启 AI-RVC
- 关闭 Karaoke，避免额外运行 MVSep 9205 三模型
- 启动前设置 `POLARFORMER_MAX_CHUNK_SIZE=220500`；显存会下降，分块数量和耗时会增加
- 选择 Demucs 或 UVR5 属于显式更换分离路线，输出不会与默认模型相同

**Q: 首次运行很慢**

首次运行会自动下载模型文件（大小随模型版本变化），请耐心等待。

**Q: 日志长时间停在 PolarFormer，但 GPU 一直满载**

PolarFormer 会逐块处理整首歌曲。日志会显示 `BS PolarFormer 正在处理分块 x/y`。GPU 持续有计算负载且分块编号前进，说明任务仍在运行；分块编号不再变化、GPU 利用率长期为 0，或进程退出时才按异常排查。开启 Karaoke 后还会继续运行三个 MVSep 9205 子模型。

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

## 数据核验说明（2026-07-12）

本节记录 README 使用的一手来源。运行状态以当前仓库代码和测试为准，论文模型的功能以论文与官方仓库为准。

### 当前运行链路

- [Leap XE 权重](https://huggingface.co/pcunwa/BS-Roformer-Leap)；[MVSep 10178](https://mvsep.com/quality_checker/entry/10178)
- [BS PolarFormer v1.0.20 权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.20)；[62-band ONNX](https://huggingface.co/bgkb/bs_polarformer)；[MVSep 10009](https://mvsep.com/quality_checker/entry/10009)
- [MVSep 9205](https://www.mvsep.com/quality_checker/entry/9205)：Gabox_IS、Frazer&Becruily、Anvuew 三模型 `avg_wave`
- [RoFormer De-Reverb](https://huggingface.co/anvuew/dereverb_mel_band_roformer)
- [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)、[RMVPE](https://github.com/Dream-High/RMVPE)、[HuBERT](https://arxiv.org/abs/2106.07447)、[FAISS](https://github.com/facebookresearch/faiss)
- [audio-separator 0.44.1](https://pypi.org/project/audio-separator/)、[Demucs](https://github.com/facebookresearch/demucs)、[UVR5](https://github.com/Anjok07/ultimatevocalremovergui)

### 分离架构与评估

- [BS-RoFormer](https://arxiv.org/abs/2309.02612)、[Mel-Band RoFormer](https://arxiv.org/abs/2310.01809)、[Mel-RoFormer vocal separation](https://arxiv.org/abs/2409.04702)
- [PolarFormer 124 bands / MVSep 10147](https://www.mvsep.com/quality_checker/entry/10147)、[BS-RoFormer 算法页](https://mvsep.com/algorithms/34)
- [Hybrid Demucs](https://arxiv.org/abs/2111.03600)、[Sound Demixing Challenge 2023](https://transactions.ismir.net/articles/10.5334/tismir.171)
- [SI-SDR](https://arxiv.org/abs/1811.02508)、[museval](https://github.com/sigsep/sigsep-mus-eval)

### VC、SVC 与 SSC 前沿

- [Seed-VC 论文](https://arxiv.org/abs/2411.09943)与[官方仓库](https://github.com/Plachtaa/seed-vc)；仓库在核验日为 GPL-3.0 且已归档
- [Vevo](https://openreview.net/forum?id=anQDiQZhDP)、[Vevo1.5](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevosing)、[Vevo2](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2)
- [SVCC 2025](https://www.vc-challenge.org/)与[挑战总结](https://arxiv.org/abs/2509.15629)
- [S²Voice](https://arxiv.org/abs/2601.13629)、[Serenade](https://github.com/lesterphillip/serenade)、[SYKI-SVC](https://arxiv.org/abs/2501.02953)
- [FCPE](https://github.com/CNChTu/FCPE)、[SwiftF0](https://github.com/lars76/swift-f0)

PyTorch 与 CUDA 安装命令不固定写死在 README 中，使用 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 生成与本机驱动匹配的命令。

## 贡献

欢迎提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

### 发布说明要求

新版本的 Release 说明必须列出推荐 GPU、显存、系统内存、存储空间、默认分离模型和实测音频时长。模型或分块策略变化时，还要记录测试硬件、峰值显存、总耗时和相对上一版本的变化。没有完成实测的项目写“未验证”，不要填写估算值。

## 许可证

本仓库代码使用 [MIT License](LICENSE)。第三方源码、模型权重、角色模型、数据集和输入音频按各自许可证或授权条款使用，不随本仓库代码自动获得 MIT 授权。

## 致谢

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 原始 RVC 项目
- [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap) - 默认人声 stem 分离来源
- [bgkb BS PolarFormer ONNX](https://huggingface.co/bgkb/bs_polarformer) - 默认纯伴奏 stem 分离来源
- [MVSep Quality Checker 9205](https://www.mvsep.com/quality_checker/entry/9205) - 默认原曲主唱 / `Back+Instrumental` 分离来源
- [Mel-Band RoFormer](https://arxiv.org/abs/2310.01809) - RoFormer / De-Reverb 路线的重要论文依据
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - 音源分离推理框架
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) - BS PolarFormer / RoFormer 预训练权重
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) - Ultimate Vocal Remover
- [Demucs](https://github.com/facebookresearch/demucs) - Meta 人声分离
- [RMVPE](https://arxiv.org/abs/2306.15412) - 高质量 F0 提取
- [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) - 语音特征提取
- [Gradio](https://gradio.app/) - Web 界面框架

## 免责声明

- 只处理你有权使用的音频、模型和声音素材。
- 不得将转换结果用于冒充、诈骗、误导、骚扰或其他违法侵权行为。
- 声音、歌曲、角色和模型权重可能涉及版权、邻接权、人格权、商标权或单独的模型许可证；使用者负责取得所需授权。
- 本项目不会改变输入素材或第三方模型原有的权利归属。
- 软件按 MIT License 的“原样”条款提供，不附带适销性、特定用途适用性或不侵权保证。
