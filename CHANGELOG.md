# Changelog

## v1.4.0 - 2026-07-12

### 中文更新说明

- TelKNet 默认分离链路已对齐：非 WAV 输入统一解码为 44.1 kHz 双声道 PCM16，Leap XE 与 PolarFormer 使用同一份输入；PolarFormer 增加孤立声道饱和抑制；MVSep 9205 三模型 `avg_wave` 从原始整曲分离主唱与带和声伴奏。
- 输出语义与生产环境保持一致：`backing_vocals.wav` 为 Leap 人声减去 MVSep 主唱得到的纯和声，`accompaniment.wav` 为 MVSep `Back+Instrumental`，`accompaniment_without_harmony.wav` 为 PolarFormer 纯伴奏。
- 移除未调用的旧 Karaoke 和声回混链路，以及会把 `bs_polarformer_124bands_fp16` 错当成 62-band ONNX 模型的兼容别名；错误模型 ID 现在会显式失败。
- README、Hugging Face README、Colab、依赖注释、诊断脚本和 Web UI 文案已同步说明 Leap XE + BS PolarFormer + MVSep 9205 + RoFormer De-Reverb 默认链路及三类输出语义。
- `tools/download_models.py`、`run.py`、HF `app.py`、MCP 模型状态和 GitHub Release 打包流程现在都会显式准备/检查 Leap XE vocals、BS PolarFormer pure accompaniment、MVSep 9205 子模型与 RoFormer De-Reverb。
- UI 路由状态现在明确展示统一 PCM、Leap/PolarFormer、MVSep 9205、纯和声推导、RoFormer De-Reverb、RVC 和混音的真实顺序。
- MVSep 9205 现在按榜单口径直接处理原始整曲，短音频会补到三模型所需的最大窗口，再把输出裁回原时长。
- 单卡流程会在 MVSep 前卸载 Leap XE 与 PolarFormer；最终混音直接使用 MVSep 的带和声伴奏轨，PolarFormer 纯伴奏仍单独导出，不会重复叠加乐器。
- 运行日志不再把 MVSep `Back+Instrumental` 写成 `backing_vocals.wav`：默认 Karaoke 临时输出改为 `karaoke/accompaniment.wav`，纯和声仍为会话根目录的 `backing_vocals.wav`；进度、模型加载和混音日志统一使用“纯伴奏 / 带和声伴奏 / 纯和声”。
- 显式选择 CUDA、XPU、DirectML 或 MPS 后，设备不可用会直接报错；只有 `device=auto` 才会自动选设备。CPU 安装和 CPU 便携包会显式写入 `device=cpu`。
- Karaoke 输出无法唯一识别主唱与第二路、或检测到疑似反转时会停止，不再按文件顺序猜测或自动交换。
- Karaoke 开启时，`accompaniment.wav` 始终导出 MVSep `Back+Instrumental`；关闭成品和声时只改变混音输入，不改变公开输出语义。
- ONNX Runtime 改由 `audio-separator` 的 CPU/GPU/DML extra 安装，避免 CPU 与 GPU 发行包同时写入同一模块；GPU 便携包不再宣称缺少 CUDA 时自动回退 CPU。
- Windows 控制台和重定向日志统一使用 UTF-8，避免中文/English 混合输出在 CI、Codex 或发布诊断日志中乱码。
- 控制台消息现在严格跟随 `zh_CN` / `en_US`：英文模式使用随包发布的离线目录，漏译会显式报错，不再混入中文。
- `docs` 目录中的 Markdown 已从仓库移除并加入忽略规则；发布要求保留在 README 与 Changelog 中。
- PyTorch、torchvision 与 torchaudio 现在从同一 PyTorch 索引一次性安装；`install.py --check` 会执行 `pip check`，依赖版本不一致时直接失败。
- 2026-07-12 本地验证：完整 170 项测试全部通过；RTX 4070 Ti SUPER 真实 CUDA 完整翻唱已覆盖 Leap XE、PolarFormer、MVSep 9205 三模型、RoFormer De-Reverb、RVC 与最终混音，英文控制台全程无汉字。

### English Notes

- Default vocals/instrumental separation now uses `hybrid:leap_xe90_vocals+polarformer62_instrumental`: Leap XE 90 bands extracts vocals from the original full mix and BS PolarFormer public ONNX 62 bands extracts pure accompaniment; default lead / backing+instrumental separation uses `ensemble:mvsep_9205_avg` on the original full mix.
- README, Hugging Face README, Colab, dependency comments, diagnostics, and Web UI copy now describe the Leap XE + BS PolarFormer + MVSep 9205 + RoFormer De-Reverb default route and its three output stems.
- `tools/download_models.py`, `run.py`, HF `app.py`, MCP model status, and GitHub release packaging now explicitly prepare/check Leap XE vocals, BS PolarFormer pure accompaniment, the MVSep 9205 submodels, and RoFormer De-Reverb.
- The TelKNet production separation route is now aligned end to end: non-WAV input is decoded once to 44.1 kHz stereo PCM16, Leap XE and PolarFormer share that input, PolarFormer applies isolated-channel saturation suppression, and MVSep 9205 runs its three-model `avg_wave` ensemble on the original mix.
- Export semantics now match production: `backing_vocals.wav` is pure harmony derived as Leap vocals minus MVSep lead, `accompaniment.wav` is the MVSep `Back+Instrumental` stem, and `accompaniment_without_harmony.wav` is the pure PolarFormer accompaniment.
- Removed the unused legacy Karaoke backing-remix path and the alias that treated `bs_polarformer_124bands_fp16` as the 62-band ONNX model; invalid model IDs now fail explicitly.
- MVSep 9205 now processes the original full mix, pads short inputs to the longest model window, and trims both outputs back to the source duration.
- Single-GPU runs unload Leap XE and PolarFormer before MVSep. The final mix uses MVSep backing+instrumental directly while keeping the pure PolarFormer accompaniment as a separate export.
- Runtime logs no longer label the MVSep `Back+Instrumental` stem as `backing_vocals.wav`: the default Karaoke temporary output is now `karaoke/accompaniment.wav`, while pure harmony remains `backing_vocals.wav` at the session root.
- Explicit CUDA, XPU, DirectML, and MPS selections now fail when unavailable; only `device=auto` performs automatic selection. CPU installs and portable CPU builds write `device=cpu` explicitly.
- Karaoke output classification now stops on unknown, ambiguous, or apparently reversed stems instead of guessing by file order or swapping tracks.
- With Karaoke enabled, `accompaniment.wav` always exports MVSep `Back+Instrumental`; disabling harmony in the final mix changes only the mix input, not the public output contract.
- ONNX Runtime now comes from the matching `audio-separator` CPU/GPU/DML extra, avoiding simultaneous CPU and GPU runtime distributions.
- Windows console and redirected project logs now use UTF-8 so mixed Chinese/English output remains readable in CI and release diagnostics.
- Console messages now strictly follow `zh_CN` / `en_US`. English mode uses a bundled offline catalog and fails explicitly on missing translations instead of mixing in Chinese.
- Markdown files under `docs` were removed from version control and are now ignored; release requirements remain in the README and changelog.
- PyTorch, torchvision, and torchaudio are now installed together from one PyTorch index, and `install.py --check` fails when `pip check` finds an inconsistent environment.
- Verified locally on July 12, 2026: all 170 tests passed; a real end-to-end CUDA cover on an RTX 4070 Ti SUPER covered Leap XE, PolarFormer, all three MVSep 9205 models, RoFormer De-Reverb, RVC, and the final mix with no Han characters in the English console log.

## v1.3.0 - 2026-07-04

AI-RVC v1.3.0 重点整理默认翻唱质量路线、模型兼容性和发布打包流程。

### 中文更新说明

#### 亮点

- 默认翻唱路线改为严格 RMVPE，并关闭 `f0_hybrid_mode`；混合 F0 和 fallback F0 不再静默改结果，出问题会直接报错。
- 默认启用更贴近官方 RVC 的推理路径，同时保留本项目的翻唱预处理、清理、源约束和混音流程。
- RVC 权重会根据模型形状识别 v1/v2，不再只依赖可能缺失或写错的 `version` 元数据。
- 转换前会校验 FAISS 索引维度是否匹配当前 RVC 模型。
- 支持导入自定义角色模型：单个 `.pth`、`.pth + .index`，或只包含一个模型的 `.zip`。
- 可下载角色模型扩展到 181 个，并优化了系列和分类展示。
- 清理 Gradio 临时前缀生成的下载文件名，并给每个输出结果单独提供下载按钮。

#### 质量路线

- RoFormer De-Reverb 现在是默认翻唱的严格 VC 预处理路线。
- 源音清理和过渡平滑会尽量保留有效人声，同时压掉尾音回声、低电平杂音和切换尖刺。
- 新增默认质量审计工具，用来阻止隐藏参数覆盖，并要求最终质量结论必须经过听感确认。
- 移除了默认流程里的 UI singing-repair 开关，因为它依赖 F0 fallback 行为。

#### 安装和运行

- `run.py` 会检查必需基础模型，并在默认路线需要时准备 `_official_rvc/` 官方源码树。
- `tools/download_models.py` 支持准备 `_official_rvc/`，源码树不完整会直接报错。
- 依赖检查会严格校验 Gradio、fairseq、audio-separator 等关键版本。
- Hugging Face Hub 限制在 1.0 以下，避免和 Space 运行环境冲突。
- GitHub Release 打包已对齐 Gradio 5.49.1，按 CPU/GPU 版本固定 audio-separator 0.44.1，预下载当时的 RoFormer 默认模型，并把 `_official_rvc/` 打进便携包。

#### UI 和文档

- 翻唱参数现在会校验输入值，不再自动夹取或静默降级。
- 混音预设会真正同步到音量和混响滑块，同时仍可手动微调。
- README 和 Hugging Face README 已更新当前模型定位、SOTA 边界、严格默认参数和官方 RVC 源码准备方式。
- 本地 agent 指令文件和生成的审计产物已从 Git 跟踪中移除；被 README 引用的文档和界面演示图继续保留。

#### 测试

- 新增或补齐了音频清理保护、RVC 版本识别、官方 adapter 导出、索引维度校验、自定义模型导入、下载文件名清理、UI 下载按钮、默认质量审计策略、安装依赖检查和严格翻唱配置的测试。
- 2026-07-04 本地已验证：`python -m unittest discover -s tests`，116 个测试通过。

#### 下载说明

- Windows 包是 `.7z` 分卷，下载同一版本的所有 `.001`、`.002` 等文件后，用 7-Zip 从 `.001` 解压。
- Linux 包是 `.tar.gz.part*` 分卷，下载同一版本的全部分卷后再合并解压。
- CPU 版不需要显卡；GPU 版面向 NVIDIA CUDA 12.1 环境。

### English Release Notes

AI-RVC v1.3.0 focuses on the default cover quality route, model compatibility, and release hygiene.

#### Highlights

- Changed the default cover route to strict RMVPE with `f0_hybrid_mode=off`; hybrid/fallback F0 paths now fail loudly instead of silently changing behavior.
- Enabled the official-compatible RVC inference path by default while keeping the project cover preprocessing, cleanup, source constraint, and mixing chain.
- Added RVC checkpoint version detection from weight shape, so v1/v2 models with missing or wrong `version` metadata are handled explicitly.
- Validates FAISS index dimensions against the loaded RVC model before conversion.
- Added custom character model import from `.pth`, `.pth + .index`, or a single-model `.zip`.
- Expanded the downloadable character registry to 181 entries and improved series/category grouping.
- Cleaned generated cover download filenames by removing Gradio temp prefixes and adding per-output download buttons.

#### Quality Route

- RoFormer De-Reverb is now the strict VC preprocessing path for default covers.
- Source cleanup and transition smoothing now preserve more active vocal body while still suppressing echo tails, quiet artifacts, and transition spikes.
- Added a default-quality audit tool that blocks hidden parameter overrides and requires listening review for final quality verdicts.
- Removed the UI singing-repair toggle from the default flow because it depended on F0 fallback behavior.

#### Installation And Runtime

- `run.py` now verifies required base models and prepares the vendored official RVC source tree when the default route needs it.
- `tools/download_models.py` can prepare `_official_rvc/` and reports incomplete trees as hard errors.
- Dependency checks now enforce exact versions for Gradio, fairseq, and audio-separator where the project depends on pinned behavior.
- Hugging Face Hub is constrained below 1.0 to stay aligned with the Space runtime.
- GitHub release packaging now builds against Gradio 5.49.1, pins audio-separator 0.44.1 per CPU/GPU variant, preloads the then-current separator defaults, and bundles `_official_rvc/` into portable artifacts.

#### UI And Documentation

- Cover controls now validate values instead of clamping or silently falling back.
- Mix presets update actual mix sliders and still allow manual adjustment.
- README and Hugging Face README now document the current model positioning, SOTA boundaries, strict defaults, and official RVC source preparation.
- Local agent instruction files and generated audit artifacts are removed from Git tracking, while referenced docs and demo images remain tracked.

#### Tests

- Added coverage for audio cleanup guards, RVC version detection, official adapter export, index dimension validation, custom model import, clean output filenames, UI download buttons, default quality audit policy, install requirement checks, and strict cover configuration.
- Verified locally with `python -m unittest discover -s tests` on July 4, 2026: 116 tests passed.

#### Download Notes

- Windows packages are split `.7z` archives. Download every volume for the same edition, then extract from `.001` with 7-Zip.
- Linux packages are split `.tar.gz.part*` archives. Download every part for the same edition, then concatenate and extract.
- CPU builds do not require a GPU. GPU builds target NVIDIA CUDA 12.1 environments.
