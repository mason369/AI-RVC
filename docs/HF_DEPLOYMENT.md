# Hugging Face Spaces 部署总结

## 部署信息

- **Space 名称**: mason369/AI-RVC
- **Space URL**: https://huggingface.co/spaces/mason369/AI-RVC
- **SDK**: Gradio 3.50.2
- **硬件**: CPU (免费版)
- **部署时间**: 2026-03-10
- **状态**: ✅ 已成功部署

## 部署内容

### 上传的文件

**主要文件**:
- `app.py` - HF Spaces 入口文件
- `README.md` - Space 主页说明（从 README_HF.md 复制）
- `requirements.txt` - 项目依赖
- `configs/config.json` - 配置文件

**项目文件夹**:
- `ui/` - Gradio 界面代码
- `infer/` - 推理模块（RVC、人声分离、翻唱流水线）
- `lib/` - 核心库（音频处理、混音、设备管理）
- `models/` - 模型定义
- `tools/` - 工具脚本（模型下载、角色管理）
- `i18n/` - 国际化语言包

### 功能支持

✅ **完整功能**:
- AI 歌曲翻唱（完整流水线）
- 117 个角色模型下载
- 4 种混音预设
- 卡拉OK模式（主唱/伴唱分离）
- 4 种 VC 预处理模式
- 源约束策略
- 双 VC 管道
- 模型管理
- 设备选择

## 使用说明

### 访问 Space

直接访问: https://huggingface.co/spaces/mason369/AI-RVC

### 首次启动

⚠️ **重要提示**:
- 首次启动需要 5-10 分钟下载基础模型（HuBERT、RMVPE）
- 角色模型需要在界面中手动下载
- 请耐心等待模型下载完成

### 使用流程

1. **等待 Space 启动**（首次约 5-10 分钟）
2. **下载角色模型**：
   - 进入「歌曲翻唱」标签页
   - 展开「下载角色模型」
   - 选择并下载一个角色（推荐：星空凛、芙宁娜、纳西妲）
3. **上传歌曲**：支持 MP3/WAV/FLAC
4. **选择角色**：从已下载的角色中选择
5. **调整参数**：音调、混音预设、卡拉OK等
6. **开始翻唱**：点击「🚀 开始翻唱」
7. **下载结果**：下载生成的翻唱作品

## 性能说明

### 当前配置（免费 CPU）

- **硬件**: CPU (免费版)
- **处理速度**: 较慢（一首 3-5 分钟的歌曲约需 10-20 分钟）
- **并发限制**: 同时只能处理一个任务
- **适用场景**: 测试、演示、轻度使用

### 升级到 GPU（付费）

如需更快的处理速度，可以升级到 GPU：

1. 访问 Space 设置页面
2. 选择硬件：**T4 small** 或更高
3. 保存更改
4. 重启 Space

**GPU 性能**:
- **T4 small**: 一首 3-5 分钟的歌曲约需 2-5 分钟
- **A10G small**: 更快的处理速度
- **费用**: 按小时计费，详见 HF 定价页面

## 限制和注意事项

### 免费版限制

- ⚠️ **处理速度慢**: CPU 处理速度较慢
- ⚠️ **并发限制**: 同时只能处理一个任务
- ⚠️ **超时限制**: 长时间无操作会自动休眠
- ⚠️ **存储限制**: 生成的文件会定期清理

### 建议

1. **测试用途**: 免费版适合测试和演示
2. **生产用途**: 建议升级到 GPU 版本
3. **本地部署**: 对于频繁使用，建议本地部署（Windows/Linux）
4. **Colab 替代**: 也可以使用 Google Colab（免费 GPU）

## 故障排除

### 问题 1: Space 启动失败

**可能原因**:
- 依赖安装失败
- 模型下载超时
- 内存不足

**解决方案**:
1. 刷新页面重试
2. 查看 Space 日志（Logs 标签）
3. 如果持续失败，提交 Issue

### 问题 2: 处理速度很慢

**原因**: 使用免费 CPU 版本

**解决方案**:
1. 升级到 GPU 版本（付费）
2. 使用 Google Colab（免费 GPU）
3. 本地部署（Windows/Linux）

### 问题 3: 角色模型下载失败

**可能原因**:
- 网络连接问题
- HuggingFace Hub 访问限制

**解决方案**:
1. 重试下载
2. 尝试下载其他角色
3. 检查网络连接

### 问题 4: 翻唱处理失败

**可能原因**:
- 输入文件格式不支持
- 音频文件过大
- 内存不足

**解决方案**:
1. 使用支持的格式（MP3/WAV/FLAC）
2. 压缩音频文件（< 10 分钟）
3. 降低音频质量
4. 升级到 GPU 版本

## 更新和维护

### 更新 Space

要更新 Space 中的代码：

```bash
# 1. 在本地更新代码
git pull origin master

# 2. 使用 Python 脚本上传
python -c "
from huggingface_hub import HfApi
api = HfApi(token='YOUR_TOKEN')
api.upload_folder(
    folder_path='.',
    repo_id='mason369/AI-RVC',
    repo_type='space',
    ignore_patterns=['*.pyc', '__pycache__', '.git*', 'venv*', 'outputs', 'temp']
)
"
```

### 监控 Space

- **日志**: 查看 Space 的 Logs 标签
- **状态**: 查看 Space 的 Status 标签
- **使用情况**: 查看 Space 的 Analytics 标签

## 相关链接

- **Space URL**: https://huggingface.co/spaces/mason369/AI-RVC
- **GitHub 仓库**: https://github.com/mason369/AI-RVC
- **Colab Notebook**: AI_RVC_Colab.ipynb
- **完整文档**: README.md
- **问题反馈**: GitHub Issues

## 成本估算

### 免费版

- **费用**: $0
- **限制**: CPU 处理，速度慢
- **适用**: 测试、演示

### GPU 版本（T4 small）

- **费用**: 约 $0.60/小时（具体以 HF 官网为准）
- **性能**: 快速处理
- **适用**: 生产、频繁使用

### 建议

- **偶尔使用**: 免费 CPU 版本
- **频繁使用**: 本地部署（Windows/Linux）
- **临时需求**: Google Colab（免费 GPU）
- **生产环境**: GPU 版本（付费）

## 总结

✅ **部署成功**: AI-RVC 已成功部署到 Hugging Face Spaces

**优势**:
- 无需本地安装
- 随时随地访问
- 自动更新
- 易于分享

**限制**:
- 免费版处理速度慢
- GPU 版本需要付费
- 存储空间有限

**推荐使用场景**:
- 快速测试和演示
- 分享给他人使用
- 无本地环境的用户

**其他选择**:
- **本地部署**: 最佳性能，完全控制
- **Google Colab**: 免费 GPU，适合临时使用
- **HF Spaces GPU**: 付费，适合生产环境

---

**部署日期**: 2026-03-10
**部署人**: Claude Opus 4.6
**Space 状态**: ✅ 运行中
