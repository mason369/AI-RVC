# RVC v2 语音转换

基于 RVC v2 + RMVPE 的高质量语音转换系统，提供简洁的 Gradio 图形界面。

## 功能特性

- **高质量转换**: 使用 RVC v2 架构，48kHz 采样率输出
- **RMVPE 音高提取**: 目前质量最高的 F0 提取方法
- **简洁界面**: 基于 Gradio 的中文图形界面
- **自动下载**: 首次运行自动下载所需模型
- **MCP 支持**: 可作为 Claude Code 的 MCP 服务器使用

## 系统要求

- Python 3.8+
- CUDA 11.7+ (推荐，CPU 也可运行但较慢)
- 4GB+ 显存 (GPU 模式)
- 8GB+ 内存

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/AI-RVC.git
cd AI-RVC
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 版本
pip install torch torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

### 4. 下载模型

```bash
python tools/download_models.py
```

### 5. 启动应用

```bash
# Windows
run.bat

# 或直接运行
python run.py
```

访问 http://127.0.0.1:7860 打开界面。

## 使用说明

### 添加语音模型

1. 将 `.pth` 模型文件放入 `assets/weights/` 目录
2. 如果有对应的 `.index` 文件，使用相同的文件名放入同一目录
3. 在界面中点击「刷新」按钮

### 转换参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| 音调偏移 | 半音数，正数升调，负数降调 | 男转女: +12, 女转男: -12 |
| F0 提取方法 | 音高提取算法 | rmvpe (质量最高) |
| 索引比率 | 越高越像训练音色 | 0.5 |
| 中值滤波 | 减少气息噪声 | 3 |
| 响度混合 | 输出响度控制 | 0.25 |
| 清辅音保护 | 保护清辅音和呼吸声 | 0.33 |

## 项目结构

```
AI-RVC/
├── assets/              # 模型文件
│   ├── hubert/          # HuBERT 模型
│   ├── rmvpe/           # RMVPE 模型
│   ├── pretrained_v2/   # 预训练权重
│   └── weights/         # 用户语音模型
├── configs/             # 配置文件
├── i18n/                # 语言包
├── infer/               # 推理模块
├── lib/                 # 核心库
├── models/              # 模型定义
├── mcp/                 # MCP 服务器
├── tools/               # 工具脚本
├── ui/                  # Gradio 界面
├── run.py               # 主入口
└── run.bat              # Windows 启动脚本
```

## MCP 服务器

本项目可作为 Claude Code 的 MCP 服务器使用，提供以下工具：

- `convert_voice`: 语音转换
- `list_models`: 列出可用模型
- `download_model`: 下载基础模型
- `check_models`: 检查模型状态

配置文件位于 `.claude/mcp.json`。

## 常见问题

### Q: CUDA out of memory

A: 尝试以下方法：
- 使用较短的音频
- 关闭其他占用显存的程序
- 使用 CPU 模式 (在 .env 中设置 `DEVICE=cpu`)

### Q: 转换后声音失真

A: 尝试以下方法：
- 降低索引比率
- 使用更高质量的输入音频
- 确保输入音频是干声 (无伴奏)

### Q: 模型下载失败

A: 尝试以下方法：
- 检查网络连接
- 使用代理或镜像源
- 手动下载模型文件放入对应目录

## 致谢

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 原始 RVC 项目
- [RMVPE](https://github.com/Dream-High/RMVPE) - 高质量 F0 提取
- [Gradio](https://gradio.app/) - Web 界面框架

## 许可证

MIT License
