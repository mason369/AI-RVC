# Changelog

## 1.7.0 — 2026-11-26

### 中文

#### 界面、双主题与文案优化
- **全新工作室界面排版**：重构主界面卡片层级与双栏工作台布局，将「歌曲翻唱」设为默认首屏，压缩无效留白并优化筛选、角色信息卡片与手动参数的分组排列；桌面与窄屏（390px）完整响应式自适应。
- **日间 / 夜间双主题系统**：新增页首日间/夜间即时切换按钮，支持跟随系统配色并本地持久化记忆；主界面与离线内嵌多轨播放器通过消息实时同步主题且不中断播放。
- **全面清理文案与翻译**：精简界面提示、更新全部 README 与技术文档，移除冗余说明与过程性日志；批量修正 `i18n/console_en_US.json` 中 282 项角色名及专业音频术语翻译错误。

#### 核心新功能
- **同曲调参分轨复用**：在同一会话中修改音调或更换角色时自动复用已分离音轨；仅调整混音音量、混响、原唱混入或和声开关时，直接复用已有转换人声快速重混。
- **批量队列翻唱**：支持一次上传多首歌曲或多段干声按顺序批量处理，单首异常自动隔离并保留详细诊断；完成后支持逐首切换试听分轨及整批成品 ZIP 一键下载。
- **纯人声转换模式**：配置与界面新增 `song` / `vocals` 输入模式；已有干声直接进入 RVC 换声并导出输入人声与转换人声两轨对照。
- **Colab 批量翻唱单元**：更新 `AI_RVC_Colab.ipynb`，支持笔记本内直接上传、复用会话参数与下载成品 ZIP。

#### 音频质量与精度修复
- **同步 TelkNet 同源音频更新**：统一检查点加载与完整层校验（支持标准 v1/v2、无 F0、多说话人及 Applio HiFi-GAN / MRF / RefineGAN）；改进 RMVPE 峰值解码、连续 F0 精度、参考幅度/目标相位后处理与长歌曲绝对坐标交叉淡化拼接。
- **修复 GTX 16 系列半精度选择**：RVC 与 RMVPE 共用的精度选择器同时核对 CUDA 型号与计算能力，按官方设备规则将 GTX 16 系列（含 GTX 1660 SUPER）的 FP16 权重无损升为 FP32 运行，保留符合条件 Tensor Core 显卡的原生 FP16 路径。
- **完整浮点与精确检索**：全程保留 FLOAT32 / FLOAT64 浮点输出与双精度混响；FAISS 检索统一使用完整 FP32 向量的内存 `FlatL2` 精确搜索；兼容标准 `trained_` / `added_` 双索引角色压缩包。

#### 打包与运行稳定性
- **内置 FFmpeg 与 Numba 缓存修复**：便携包自动暂存并校验 `ffmpeg` / `ffprobe`；`lib.mixer`、`librosa`、`resampy` 按真实包内源码收集，彻底解决异机及中文路径下的 Numba 磁盘缓存定位报错。
- **精简运行时资源与冻结程序修复**：便携包仅收集界面 Python 与内联 `player.html`，排除前端 `node_modules`；修复冻结程序官方 UVR5 路由的模块路径缓存失效问题；Docker 镜像补齐 `libatomic1` 依赖。
- **跨语言事件稳定化**：翻唱模式、原曲结构贴合与混音预设统一使用固定内部值配本地化标签，解决语言切换与页面刷新后的状态残留问题。
- **续传与批量状态修复**：Windows 文件暂时锁定时重试已验证模型的安装，重启下载后可重新校验并安装完整的本地文件。单首结果更新播放器时清空批量选中项，保留批次内的试听与下载。
- **错误提示与发布检查**：先检查角色选择及模型文件，再加载推理后端。发布流程会从原始构建分卷启动便携包，用 Chrome 检查批量入口、中英文切换及播放器主题同步。

### English

#### UI Redesign, Light/Dark Themes & Localization
- **Studio Workspace Redesign**: Rebuilt all tabs with a clean card hierarchy and two-column workspace layout, promoting Song Cover to the default landing tab with responsive desktop and mobile (390px) ergonomics.
- **Light & Dark Theme System**: Added an instant header theme toggle with system preference detection and `localStorage` persistence; synchronizes themes live with the embedded offline multitrack player without interrupting playback.
- **Copy & Translation Overhaul**: Polished bilingual UI copy, README files, and documentation; corrected 282 mistranslated character names and audio engineering terms in `i18n/console_en_US.json`.

#### New Features
- **Same-Song Stage Reuse**: Automatically reuses separated stems when changing pitch or voice in the same session, and reuses converted vocals when adjusting mix levels, reverb, or original lead blend.
- **Batch Cover Queue**: Process multiple songs or vocal clips sequentially with per-item error isolation, interactive track preview switching, and one-click ZIP archive download.
- **Vocals-Only Mode**: Added `song` / `vocals` input selection across config, pipeline, and UI for direct dry-vocal conversion.
- **Colab Batch Workflow**: Updated `AI_RVC_Colab.ipynb` for in-notebook multi-file upload, session reuse, and ZIP export.

#### Audio & Precision Improvements
- **TelkNet Shared Audio Sync**: Unified checkpoint validation across standard v1/v2, non-F0, multi-speaker, and Applio HiFi-GAN / MRF / RefineGAN models; improved vectorized RMVPE decoding, continuous F0 precision, reference-magnitude/target-phase post-processing, and absolute-position long-song splicing.
- **GTX 16-Series Precision Fix**: Precision selection now checks both CUDA device name and compute capability so GTX 16-series cards (including GTX 1660 SUPER) losslessly widen FP16 weights to FP32 under upstream device rules while eligible Tensor Core GPUs retain native FP16.
- **Float Preservation & Exact Retrieval**: Maintains Float32/Float64 audio and reverb precision, exact full-vector FP32 `FlatL2` FAISS retrieval, and support for paired `trained_`/`added_` character archives.

#### Packaging & Stability
- **Bundled FFmpeg & Relocatable Numba Cache**: Bundles verified `ffmpeg`/`ffprobe` binaries and collects cached audio modules (`lib.mixer`, `librosa`, `resampy`) from bundled source files so Numba disk caching works cleanly across relocated Unicode paths.
- **Clean Runtime Packaging**: Ships only runtime UI files and the self-contained `player.html` (excluding `node_modules`), fixes frozen official UVR5 `inspect` module cache invalidation, and adds `libatomic1` to Docker images.
- **Cross-Locale Event Stability**: Uses stable internal values with localized labels for conversion routes, source alignment, and mix presets across language switches and page reloads.
- **Download Recovery and Batch State**: Retries publishing verified model files after temporary Windows file locks. Rerunning the downloader can verify and install a completed local file. Generating a single result clears the previous batch selection while keeping the batch available for preview and download.
- **Validation and Release Checks**: Validates voice selection and model files before loading the inference backend. The release workflow opens the original portable build in Chrome to check the batch entry point, both languages and player theme synchronization.

## 1.6.0 — 2026-10-23

同步 TelkNet 的翻唱音频更新，提供纯人声转换和 Windows/Linux 便携包。10 月 26 日更新修复了冻结程序的 Numba 缓存定位、打包范围及语言切换问题。[原版发布说明](docs/releases/v1.6.0.md)

Imported TelkNet's cover audio updates and introduced dry-vocal conversion and Windows/Linux portable builds. The October 26 rebuild fixed frozen Numba caching, bundle contents and language switching. [Original release notes](docs/releases/v1.6.0.md)

---

## 1.5.1 — 2026-09-06

### 中文
- 修复 UVR5 重建频谱缓冲区未初始化导致的偶发 `NaN` 音频缺陷；补齐 PyTorch 2.11 经由 `cuda-toolkit` 引入的 NVIDIA 动态库收集及 NVRTC 实际编译验证。
- 统一升级分离引擎至 `audio-separator==0.47.0`，默认纯伴奏切换为标准 **Leap Instrumental 62 bands**，移除 PolarFormer 路线与旧配置；新增显式 DirectML 分配器检查。
- 新增 Linux x86-64 CPU / NVIDIA CUDA Docker 镜像与 Compose 部署（支持持久化卷、非 root 运行、可选登录认证、子路径反向代理与健康检查）。
- 新增离线内嵌同步多轨播放器（支持静音、独奏、增益、时间偏移、波形缩放与对照轨拖入），并引入 Gradio 5.49.1 上传 ID 初始化修复与 Windows `SelectorEventLoop` 适配。
- 实现页首中英文即时切换、原子化配置保存、角色资产内容哈希校验、清辅音保护无声帧修复及 `rvc_mcp` 独立 MCP 服务包。

### English
- Fixed uninitialized UVR5 spectrum bins causing intermittent `NaN` audio and completed NVIDIA library/NVRTC collection for PyTorch 2.11 GPU bundles.
- Upgraded `audio-separator` to `0.47.0`, switched default instrumental separation to standard **Leap Instrumental 62 bands**, removed PolarFormer, and added explicit DirectML allocator checks.
- Added Linux x86-64 CPU/CUDA Docker images and Compose setups with persistent volumes, non-root execution, optional auth, subpath proxy support, and health checks.
- Added the offline embedded synchronized multitrack player, Gradio 5.49.1 upload ID fix, and Windows `SelectorEventLoop` server runtime.
- Added instant bilingual UI switching, atomic config persistence, content-hashed character validation, unvoiced-frame protection fixes, and the standalone `rvc_mcp` package.

---

## 1.4.0 — 2026-07-12

### 中文
- 对齐默认分离链路：统一解码输入、Leap XE 90 人声提取、MVSep 9205 三模型 `avg_wave` 原曲主唱与带和声伴奏分离、纯和声差分导出。
- 完善显式计算设备选择策略、UTF-8 控制台日志与离线中英文控制台本地化词典；统一 PyTorch 索引安装与 `pip check` 校验。

### English
- Aligned the default separation pipeline with unified decoding, Leap XE 90 vocals, MVSep 9205 three-model `avg_wave` lead/backing+instrumental separation, and pure harmony derivation.
- Added explicit compute device enforcement, UTF-8 console logging, offline English console localization, and unified PyTorch dependency checks.

---

## 1.3.0 — 2026-07-04

### 中文
- 确立严格 RMVPE 默认翻唱质量路线与官方兼容推理路径，新增权重形状自动识别 v1/v2 与 FAISS 索引维度校验。
- 新增自定义模型导入（`.pth` / `.index` / `.zip`）、181 项内置角色模型清单、干净下载文件名处理与单轨下载按钮。

### English
- Established the strict RMVPE default cover workflow with shape-based RVC v1/v2 detection and FAISS index dimension validation.
- Added custom model import (`.pth`/`.index`/`.zip`), 181 built-in character entries, clean download filenames, and per-stem download buttons.
