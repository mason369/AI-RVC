# 多轨播放器

基于 TelkNet 的 `MultiTrackAudioMixer.tsx` 适配，完整保留同步时间轴、逐轨增益、静音、独奏、时间偏移、缩放、拖动添加与移除交互。宿主为 Gradio 5.49.1，运行时直接读取随程序分发的离线单文件 `dist/player.html`，无需 Node.js、CDN 或外部服务。

## 功能与交互

- **整曲与纯人声模式**：整曲模式默认载入转换后人声与伴奏，其余分轨按需点击或拖入对比（新加入对照轨默认静音）；纯人声模式载入输入人声与转换后人声两轨。
- **日夜间双主题同步**：通过 `rvc-theme` 消息与宿主界面实时同步浅色 / 深色主题，切换外观时保持当前播放状态与音轨参数不变。
- **中英文即时切换**：通过 `rvc-language` 消息同步 `i18n` 词条，保留当前播放会话、音轨选择、增益与时间偏移；仅在生成新结果（`rvc-tracks`）时重建播放会话。
- **互斥试听与无损下载**：独立单轨试听与多轨混音自动互斥；所有混音台调节仅用于实时试听，下载按钮保存原始浮点 WAV。

## 开发构建

在本目录执行：

```bash
npm ci
npm run build
```

构建脚本将 Tailwind CSS 与 React 代码内联打包为单文件 `dist/player.html`。发行包仅收集 `dist/player.html`，不包含 `node_modules`。

## Gradio 与 Windows 运行时适配

- `ui/gradio_assets.py`：引入 Gradio #12637 上传 ID 初始化修复与流连接取消支持，在响应层按版本精确补丁，不修改 `site-packages`。
- `ui/server_runtime.py`：在 Windows 下为 Gradio HTTP 服务显式使用 `SelectorEventLoop`，避免媒体播放客户端中断 Range 请求时触发 Proactor 套接字清理异常。
- `lib/logger.py`：日志格式器完整保留异常堆栈，便于排查运行错误。
