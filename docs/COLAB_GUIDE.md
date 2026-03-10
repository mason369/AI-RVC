# Google Colab 使用指南

## 简介

Google Colab 是一个免费的云端 Jupyter notebook 环境，提供免费的 GPU 加速。使用 Colab 可以无需本地安装即可体验 AI-RVC 的所有功能。

## 优势

- ✅ **免费 GPU**：T4 GPU（16GB 显存）或 V100 GPU（Colab Pro）
- ✅ **无需安装**：无需本地安装 Python、CUDA、依赖库
- ✅ **开箱即用**：自动配置环境，一键启动
- ✅ **功能完整**：支持所有 Web UI 功能
- ✅ **随时随地**：只需浏览器即可使用

## 快速开始

### 1. 打开 Notebook

1. 访问 GitHub 仓库：https://github.com/mason369/AI-RVC
2. 点击 `AI_RVC_Colab.ipynb` 文件
3. 点击「在 Colab 中打开」按钮（或直接访问 Colab 链接）

### 2. 设置 GPU 运行时

**重要**：必须启用 GPU 才能正常运行

1. 点击菜单栏：**代码执行程序** → **更改运行时类型**
2. 在「硬件加速器」下拉框中选择 **GPU**
3. 推荐选择 **T4 GPU**（免费）
4. 点击「保存」

### 3. 执行单元格

按顺序执行每个单元格（点击单元格左侧的播放按钮，或按 `Shift + Enter`）：

#### 单元格 1：环境检查
- 检查 GPU 是否可用
- 显示 Python 和 PyTorch 版本
- 显示 CUDA 版本和 GPU 型号

#### 单元格 2：克隆仓库
- 从 GitHub 克隆 AI-RVC 项目
- 切换到项目目录

#### 单元格 3：安装依赖
- 安装 PyTorch（CUDA 12.1）
- 安装项目依赖（约 5-10 分钟）

#### 单元格 4：下载基础模型
- 下载 HuBERT 模型（~190 MB）
- 下载 RMVPE 模型
- 检查模型状态

#### 单元格 5：下载角色模型（可选）
- 查看可用角色列表（117 个）
- 下载单个角色
- 下载指定系列的所有角色
- 下载全部角色（需要较长时间）

#### 单元格 6：启动 Gradio 界面
- 启动 Web 界面
- 生成公共链接（可分享）
- 点击链接访问界面

#### 单元格 7：命令行翻唱（可选）
- 通过代码直接处理翻唱
- 适合批量处理或自动化

#### 单元格 8：下载输出文件
- 列出生成的文件
- 下载到本地

## 详细使用说明

### 下载角色模型

#### 方式 1：下载单个角色

```python
from tools.character_models import download_character_model

# 修改角色名称
character_name = "rin"  # 星空凛

print(f"正在下载角色: {character_name}")
success = download_character_model(character_name)
if success:
    print(f"✅ {character_name} 下载完成")
```

**可用角色名称示例**：
- `rin` - 星空凛
- `umi` - 园田海未
- `nozomi` - 东条希
- `chika` - 高海千歌
- `riko` - 樱内梨子
- `furina` - 芙宁娜
- `nahida` - 纳西妲
- `raiden` - 雷电将军

#### 方式 2：下载指定系列

```python
from tools.character_models import download_all_character_models

# 修改系列名称
series_name = "Love Live!"  # 可选: "原神", "Hololive" 等

print(f"正在下载系列: {series_name}")
result = download_all_character_models(series=series_name)
print(f"✅ 成功: {len(result['success'])} 个")
```

**可用系列**：
- `Love Live!`
- `Love Live! Sunshine!!`
- `Love Live! 虹咲学园`
- `Love Live! Superstar!!`
- `原神`
- `Hololive`
- `偶像大师`
- `碧蓝航线`

#### 方式 3：下载全部角色

```python
from tools.character_models import download_all_character_models

print("正在下载全部 117 个角色模型...")
result = download_all_character_models()
print(f"✅ 成功: {len(result['success'])} 个")
```

**注意**：下载全部角色需要 10-20 分钟，建议只下载需要的角色。

### 使用 Gradio 界面

启动界面后，会生成两个链接：

1. **本地链接**：`http://127.0.0.1:7860`（仅 Colab 内部可访问）
2. **公共链接**：`https://xxxxx.gradio.live`（可分享，72 小时有效）

点击公共链接即可访问完整的 Web 界面，功能与本地运行完全一致：

- **歌曲翻唱**：上传歌曲，选择角色，调整参数，生成翻唱
- **角色模型管理**：下载、筛选、搜索角色模型
- **混音预设**：通用、人声突出、伴奏突出、现场感
- **卡拉OK模式**：分离主唱和伴唱
- **VC预处理**：自动、直通、学习型DeEcho、旧版手工链
- **模型管理**：下载基础模型和 DeEcho 模型
- **设置**：查看设备信息，选择计算后端

### 命令行翻唱

如果你想通过代码直接处理翻唱（适合批量处理）：

```python
from infer.cover_pipeline import get_cover_pipeline
from tools.character_models import get_character_model_path

# 配置参数
input_audio = "/content/your_song.mp3"  # 你的歌曲路径
character_name = "rin"  # 角色名称
output_dir = "/content/outputs"

# 获取角色模型
model_info = get_character_model_path(character_name)

# 获取翻唱流水线
pipeline = get_cover_pipeline("cuda")

# 执行翻唱
result = pipeline.process(
    input_audio=input_audio,
    model_path=model_info["model_path"],
    index_path=model_info.get("index_path"),
    pitch_shift=0,  # 音高偏移（-12 ~ 12）
    index_ratio=0.35,  # 索引率（0 ~ 1）
    vocals_volume=1.0,  # 人声音量（0 ~ 2）
    accompaniment_volume=1.0,  # 伴奏音量（0 ~ 2）
    reverb_amount=0.1,  # 混响量（0 ~ 1）
    karaoke_separation=True,  # 启用卡拉OK分离
    vc_preprocess_mode="auto",  # VC预处理模式
    output_dir=output_dir
)

print(f"✅ 翻唱完成！")
print(f"最终翻唱: {result['cover']}")
```

### 下载输出文件

生成的文件保存在 `/content/AI-RVC/outputs/` 目录：

```python
# 列出文件
!ls -lh outputs/

# 下载文件
from google.colab import files
files.download('outputs/your_cover.wav')
```

## 参数说明

### 基础参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `pitch_shift` | 音调偏移（半音数） | 男转女: +12, 女转男: -12 |
| `index_ratio` | 索引比率（越高越像训练音色） | 0.1-0.5 |
| `filter_radius` | 中值滤波（减少气音抖动） | 3 |
| `protect` | 保护系数（防止撕裂伪影） | 0.33 |
| `rms_mix_rate` | RMS 混合率（音量包络匹配） | 0.15 |

### 混音参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `vocals_volume` | 人声音量 | 1.0 (100%) |
| `accompaniment_volume` | 伴奏音量 | 1.0 (100%) |
| `reverb_amount` | 人声混响 | 0.1-0.2 |
| `backing_mix` | 伴唱混合率 | 0.0-1.0 |

### VC 预处理模式

| 模式 | 说明 |
|------|------|
| `auto` | 自动选择（推荐） |
| `direct` | 主唱直通 RVC |
| `uvr_deecho` | 使用学习型 DeEcho/DeReverb |
| `legacy` | 旧版手工链（仅用于对比） |

### 人声分离器

| 分离器 | 说明 |
|--------|------|
| `roformer` | Mel-Band Roformer（默认，质量最高） |
| `demucs` | Demucs（速度较快） |
| `uvr5` | UVR5（兼容性好） |

## 常见问题

### Q: 如何上传歌曲文件？

A: 有两种方式：

1. **通过 Gradio 界面**：启动界面后，直接在「上传歌曲」区域拖拽或点击上传
2. **通过 Colab 文件管理器**：
   ```python
   from google.colab import files
   uploaded = files.upload()  # 点击选择文件上传
   ```

### Q: CUDA out of memory 怎么办？

A: 尝试以下方法：
1. 使用较短的音频（< 5 分钟）
2. 重启运行时（菜单栏 → 代码执行程序 → 重启运行时）
3. 切换到 `demucs` 分离器（显存占用更小）
4. 升级到 Colab Pro（更大显存）

### Q: 会话超时怎么办？

A: Colab 免费版有以下限制：
- 空闲 90 分钟后自动断开
- 连续运行 12 小时后自动断开
- 建议及时下载生成的文件

解决方案：
1. 定期点击页面保持活跃
2. 使用 Colab Pro（更长会话时间）
3. 及时下载输出文件到本地

### Q: 公共链接失效怎么办？

A: Gradio 公共链接有效期为 72 小时，失效后：
1. 重新运行「启动 Gradio 界面」单元格
2. 会生成新的公共链接

### Q: 如何保存进度？

A: Colab 环境是临时的，每次重启会清空。建议：
1. 下载生成的翻唱文件到本地
2. 如需保存角色模型，可以挂载 Google Drive：
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # 将模型复制到 Drive
   !cp -r assets/weights/characters /content/drive/MyDrive/AI-RVC-models/
   ```

### Q: 如何批量处理多首歌曲？

A: 使用命令行翻唱方式：

```python
songs = [
    "/content/song1.mp3",
    "/content/song2.mp3",
    "/content/song3.mp3"
]

for song in songs:
    result = pipeline.process(
        input_audio=song,
        model_path=model_info["model_path"],
        # ... 其他参数
    )
    print(f"✅ {song} 处理完成")
```

### Q: 如何使用自己的 RVC 模型？

A: 上传你的 `.pth` 和 `.index` 文件：

```python
from google.colab import files

# 上传模型文件
uploaded = files.upload()

# 移动到正确位置
!mkdir -p assets/weights/my_model
!mv *.pth assets/weights/my_model/
!mv *.index assets/weights/my_model/

# 使用模型
model_path = "assets/weights/my_model/your_model.pth"
index_path = "assets/weights/my_model/your_model.index"
```

## 性能优化

### 1. 使用更快的分离器

```python
# 在 process() 中设置
separator="demucs"  # 比 roformer 快，但质量略低
```

### 2. 减少 Demucs shifts

```python
# 在 process() 中设置
demucs_shifts=1  # 默认是 2，减少可提速但质量略降
```

### 3. 关闭卡拉OK分离

```python
# 在 process() 中设置
karaoke_separation=False  # 跳过主唱/伴唱分离，节省时间
```

### 4. 使用直通模式

```python
# 在 process() 中设置
vc_preprocess_mode="direct"  # 跳过 DeEcho 预处理
```

## 限制和注意事项

### Colab 免费版限制

- **GPU 时长**：每天约 12 小时
- **会话时长**：空闲 90 分钟或连续 12 小时后断开
- **存储空间**：临时存储，重启后清空
- **网络速度**：下载模型可能较慢

### Colab Pro 优势

- **更长 GPU 时长**：每天约 24 小时
- **更快 GPU**：V100 或 A100
- **更长会话**：空闲 24 小时后断开
- **优先访问**：高峰期优先分配资源

### 建议

1. **及时下载文件**：生成的翻唱文件及时下载到本地
2. **分批处理**：避免一次处理太多歌曲导致超时
3. **保持活跃**：定期点击页面防止空闲断开
4. **使用 Drive**：挂载 Google Drive 保存重要文件

## 故障排除

### 问题 1：GPU 不可用

**症状**：`torch.cuda.is_available()` 返回 `False`

**解决**：
1. 检查运行时类型是否设置为 GPU
2. 重启运行时
3. 检查 GPU 配额是否用完（免费版每天约 12 小时）

### 问题 2：依赖安装失败

**症状**：`pip install` 报错

**解决**：
1. 重启运行时
2. 检查网络连接
3. 尝试单独安装失败的包：`!pip install package_name`

### 问题 3：模型下载失败

**症状**：下载模型时超时或失败

**解决**：
1. 检查网络连接
2. 重试下载
3. 手动下载模型并上传：
   ```python
   from google.colab import files
   uploaded = files.upload()
   !mv *.pt assets/hubert/
   ```

### 问题 4：翻唱处理失败

**症状**：执行翻唱时报错

**解决**：
1. 检查输入文件格式（支持 MP3/WAV/FLAC）
2. 检查角色模型是否下载完整
3. 尝试使用较短的音频测试
4. 查看完整错误信息并根据提示调整

## 更多资源

- **GitHub 仓库**：https://github.com/mason369/AI-RVC
- **完整文档**：查看仓库中的 README.md
- **问题反馈**：GitHub Issues
- **角色模型列表**：117 个可下载角色（Love Live!, 原神, Hololive 等）

## 更新日期

2026-03-10
