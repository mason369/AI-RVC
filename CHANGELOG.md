# Changelog

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
- GitHub Release 打包已对齐 Gradio 5.49.1，按 CPU/GPU 版本固定 audio-separator 0.44.1，预下载当前 RoFormer 默认模型，并把 `_official_rvc/` 打进便携包。

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
- GitHub release packaging now builds against Gradio 5.49.1, pins audio-separator 0.44.1 per CPU/GPU variant, preloads the current RoFormer defaults, and bundles `_official_rvc/` into portable artifacts.

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
