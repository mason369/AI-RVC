# Docker 使用指南

AI-RVC 1.7.0 提供 Linux x86-64 的 **CPU** 与 **NVIDIA CUDA** 官方镜像：
- NVIDIA CUDA：`ghcr.io/mason369/ai-rvc:1.7.0-cuda`（对应 `compose.yaml`）
- CPU：`ghcr.io/mason369/ai-rvc:1.7.0-cpu`（对应 `compose.cpu.yaml`）

两套镜像均内置完整 Python 3.10 运行环境、PyTorch 2.11.0、audio-separator 0.47.0、`libatomic1`（Pedalboard 混响依赖）、FFmpeg、固定版官方 RVC 源码与离线多轨播放器，并以非 root 用户（UID/GID `1000:1000`）运行。

## 前置要求

- 宿主机已安装 Docker Engine / Docker Desktop 及 Compose 插件。
- NVIDIA GPU 版本需安装宿主机显卡驱动与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。Windows 用户推荐使用 Docker Desktop 的 WSL2 后端。

> 若在 WSL 发行版中手动运行 Docker Engine，建议使用前台 `docker compose up` 会话，避免关闭 WSL 窗口导致后台服务随发行版休眠停止。

## 快速部署

在项目根目录执行（两个基础 Compose 文件按需二选一，请勿叠加）：

```bash
# NVIDIA CUDA（推荐）
docker compose pull
docker compose run --rm ai-rvc prepare
docker compose up -d
docker compose logs -f --tail 100
```

```bash
# CPU 版本
docker compose -f compose.cpu.yaml pull
docker compose -f compose.cpu.yaml run --rm ai-rvc prepare
docker compose -f compose.cpu.yaml up -d
docker compose -f compose.cpu.yaml logs -f --tail 100
```

服务就绪后浏览器访问 `http://127.0.0.1:7860`。如需基于本地源码自行构建，将 `pull` 替换为 `build` 即可：

```bash
export AI_RVC_IMAGE=ai-rvc:local-cuda
docker compose build
docker compose up -d --pull never
```

- `prepare` 子命令用于预先下载并校验基础模型及默认分离模型，方便单独观察下载进度；角色声线模型则在界面按需下载或导入。
- 多卡服务器默认分配 1 张显卡；如需指定显卡编号，可将 `compose.yaml` 中的 `count: 1` 改为 `device_ids: ['1']`。

## 数据卷、权限与备份

Compose 默认使用命名卷 `ai-rvc_ai-rvc-data` 挂载至 `/data`，重建或升级容器时配置、模型与作品均完整保留。

| 容器路径 | 用途 |
|---|---|
| `/data/config.json` | 界面语言与翻唱默认参数 |
| `/data/assets` | 基础模型、分离模型、角色权重与索引 |
| `/data/outputs` | 翻唱成品与七类分轨输出 |
| `/data/cache` | Hugging Face / Torch 模型缓存 |
| `/data/temp` | 上传与处理临时文件 |
| `/data/logs` | 应用运行日志 |

- **宿主机目录绑定**：如需改用宿主机或 NAS 目录，先创建目录并赋予 UID/GID `1000:1000` 读写权限，再将卷配置改为 `/你的路径/ai-rvc-data:/data`。
- **计算设备策略**：容器通过环境变量 `AI_RVC_DEVICE` 固定设备类型，界面中的设备切换为只读；切换 CPU / CUDA 请改用对应的 Compose 文件与镜像重建容器。
- **备份与恢复**（确认无任务运行后操作，Windows 用户建议在 WSL 或 PowerShell 7.4+ 中执行）：

```bash
# 备份数据卷
docker compose stop
docker compose run --rm --no-deps -T --entrypoint tar ai-rvc -C /data -czf - . > ai-rvc-data.tar.gz
docker compose start

# 恢复到新卷
docker compose run --rm --no-deps -T --entrypoint tar ai-rvc -C /data -xzf - < ai-rvc-data.tar.gz
```

> 升级或停止服务时使用 `docker compose down` 即可保留数据；`docker compose down -v` 会删除数据卷，请谨慎使用。

## 登录保护、局域网与反向代理

默认仅监听 `127.0.0.1:7860`。需要修改端口、开放局域网或配置登录时，将 `docker/compose.env.example` 复制为根目录 `.env`：

- 修改端口：设置 `AI_RVC_PORT=8080`
- 局域网监听：设置 `AI_RVC_BIND=<内网IP>` 并建议同时开启登录保护
- 开启登录认证：在 `.env` 设置 `AI_RVC_AUTH_USER`，将密码写入 `docker/secrets/password.txt`，然后叠加 `compose.auth.yaml` 启动：

```bash
docker compose -f compose.yaml -f compose.auth.yaml up -d
```

- **HTTPS 反向代理**：将流量代理至 `127.0.0.1:7860`，保持流式响应与 Range 断点传输，并调大单文件上传体积上限。如需子路径部署，可在 `.env` 中配置 `AI_RVC_ROOT_PATH=/ai-rvc`，代理端剥离前缀后转发。

## 升级与离线迁移

```bash
docker compose stop
docker compose pull
docker compose up -d --no-build
docker compose run --rm ai-rvc check
```

完全离线部署时，可通过 `docker save` / `docker load` 迁移镜像，并同步携带已执行过 `prepare` 的 `/data` 备份卷。Windows 下也可使用 `powershell -ExecutionPolicy Bypass -File tools/Start-Docker.ps1`（支持 `-Variant cpu` 与 `-WithAuth`）一键完成准备与启动。

## 常用诊断命令

```bash
docker compose ps
docker compose logs --tail 200
docker compose run --rm ai-rvc check
docker compose run --rm ai-rvc python -m pip check
docker compose run --rm ai-rvc python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
