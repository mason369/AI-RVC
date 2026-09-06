# 多轨播放器

基于 TelkNet 的 `MultiTrackAudioMixer.tsx` 适配（来源快照：2026-09-06），保留时间轴、逐轨增益、静音、独奏、时间偏移、缩放、拖动添加和移除交互。宿主改为 Gradio 5.49.1，运行时只读取随程序分发的 `dist/player.html`，无需 Node.js、CDN 或外部音频服务。

默认加入转换后人声和本次导出的伴奏；其余五种输出只在实际存在时显示。手动添加的对照轨默认静音。独立播放器与多轨播放互斥。所有调整仅用于试听，下载不改变原始 Float32 WAV。播放器通过 Gradio Audio 的文件授权及缓存接口取得 URL，保留原有七音轨 API 输出顺序。

本地适配：默认整曲适配可用宽度；高度随内容变化；中英文来自项目 `i18n`；不使用参考站的签名 URL 刷新接口；文件加载、波形解码、Web Audio 初始化和播放失败会显示错误并停止混音，用户可显式重新加载。浏览器因主动暂停而中止尚未完成的 `play()` 请求属于取消，不报告成网络错误。

开发构建：在本目录执行 `npm ci`、`npm run build`。提交源码、锁文件和 `dist/player.html`；`node_modules` 不进入源码或便携包。`dist/player.js`、`dist/player.css` 是可检查的中间构建产物。播放器与 Gradio 之间通过校验来源的 `postMessage` 传递数据，结果更新后重新创建播放会话并释放旧音轨。

上游来源：`mason369/telknet` → `apps/web/features/tasks/components/MultiTrackAudioMixer.tsx`。本地不引用在线网站的脚本；React、React DOM、Lucide 和 Tailwind 的版权声明保留在构建产物中。

## Gradio 与 Windows 运行时

- `ui/gradio_assets.py` 按 Gradio #12637 补正上传 ID 赋值，并通过 AbortController 取消流连接。仅对固定 5.49.1 的已核对资源在响应时转换，不修改 site-packages，资源不匹配就报错。
- `ui/server_runtime.py` 在 Windows 的浏览器服务器中显式配置 SelectorEventLoop，处理媒体客户端取消连接。只影响服务器套接字；RVC 使用同步 subprocess 工作线程。Windows Selector 的标准限制是最多 512 个套接字，不适用于大规模并发服务；已有运行记录仅覆盖本地单用户场景。
- 日志格式器保留异常 traceback，不过滤服务端错误。

语言更新使用独立的 `rvc-language` 消息，保留当前播放会话、音轨选择、增益和时间偏移。只有新结果 `rvc-tracks` 才释放并重建会话。宿主通过 Gradio 原生语言目录更新静态文字，并对同一组件合并候选项与标签更新，避免重复输出互相覆盖。
