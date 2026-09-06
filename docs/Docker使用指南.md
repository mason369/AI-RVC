# Docker 使用指南

版本：`1.5.1`，提供 Linux x86-64 的 CPU、NVIDIA CUDA 两种镜像。根目录 Compose 默认指向 `ghcr.io/mason369/ai-rvc:1.5.1-cuda`，CPU 文件指向 `ghcr.io/mason369/ai-rvc:1.5.1-cpu`。镜像发布与验收状态见 [v1.5.1 Release](https://github.com/mason369/AI-RVC/releases/tag/v1.5.1)。

两种正式 GHCR 镜像已经公开，[发布工作流](https://github.com/mason369/AI-RVC/actions/runs/34028914300)中的 CPU/CUDA 作业各通过 **343 项回归**。两种公开成品均已使用空登录配置匿名拉取，核对版本、源码提交和镜像摘要，再以 UID 1000 断网完成默认六模型和官方 UVR5＋RVC 两条完整翻唱路线，共验证 **22 个 Float32 输出**。同标签本地构建另有各 343 项回归和 22 个输出记录，详情见[平台验收](平台适配与验收.md)。网页、登录、NGINX 子路径、下载、持久化、镜像导入和数据恢复另有候选阶段的实际记录。本机使用 WSL2 Engine，这不等于所有 Docker 宿主机都已实测。

首次 CUDA 拉取曾因 WSL 对 `ghcr.io` 的 DNS 查询超时而失败，尚未启动应用；检查 DNS 和 HTTPS 可达性后重新拉取成功。没有修改 DNS、代理、模型或推理参数；该记录不表示所有网络环境均无问题。完整镜像摘要和验收结果随 Release 提供。镜像索引附带 SBOM 与构建来源记录，实际运行平台只有 `linux/amd64`，附加的证明记录不是 ARM64 镜像。

## 推荐部署

有 NVIDIA 显卡时使用根目录 `compose.yaml`；没有 NVIDIA 显卡时使用独立的 `compose.cpu.yaml`。两个文件启动相同的完整 WebUI，CPU 不会减少模型或简化翻唱流程。AMD ROCm、Intel XPU、Apple MPS、DirectML 和 ARM64 目前没有对应的已验收镜像；Apple Docker 虚拟机中的 CPU 镜像也不能使用 MPS。

设计参考 [Applio 官方 Docker 部署](https://docs.applio.org/getting-started/installation/#docker-deployment)和 [audio-separator 容器使用方式](https://github.com/nomadkaraoke/python-audio-separator#docker)。采用单个 WebUI 服务、明确区分 CPU/GPU、模型缓存挂载；本项目进一步把配置、模型、缓存和输出归入一个 `/data` 卷。GPU 声明按照 [Docker Compose 官方文档](https://docs.docker.com/compose/how-tos/gpu-support/)，采用多阶段构建和非 root 运行，参见 [Docker 构建建议](https://docs.docker.com/build/building/best-practices/)。

宿主机需安装 Docker Engine/Desktop 和 Compose 插件。NVIDIA 还需可用的宿主机驱动与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)；Windows 推荐 Docker Desktop 的 WSL2 后端，WSL 中不要重复安装 Linux 显卡驱动。镜像固定 Python 3.10、PyTorch 2.11.0、torchvision 0.26.0、torchaudio 2.11.0、CUDA 12.8 与 audio-separator 0.47.0；CUDA 不可用时停止，不自动换 CPU。

手动在 WSL 发行版中安装 Docker Engine 时，单独的 systemd 后台服务不会保持发行版存活，参见 [Microsoft 说明](https://learn.microsoft.com/en-us/windows/wsl/systemd)。这种环境请保留前台 `docker compose up` 会话；关闭 WSL 会话可能使服务停止。Linux 服务器与 Docker Desktop 按各自的服务生命周期运行。

首先解压本版源码包或检出相同版本的仓库，在包含 Compose 文件的目录运行：

```bash
# NVIDIA（推荐）
docker compose pull
docker compose run --rm ai-rvc prepare
docker compose up -d
docker compose logs -f --tail 100
```

CPU 使用下面这一组命令，**不要把两个基础 Compose 文件叠加**：

```bash
docker compose -f compose.cpu.yaml pull
docker compose -f compose.cpu.yaml run --rm ai-rvc prepare
docker compose -f compose.cpu.yaml up -d
docker compose -f compose.cpu.yaml logs -f --tail 100
```

看到界面启动后访问 <http://127.0.0.1:7860>。自行构建时，将上述 `pull` 替换为 `build`；两种方式使用相同 Dockerfile 和完整模型链。`prepare` 串行下载和校验必需模型及默认分离模型，失败返回非零状态。普通启动也检查并准备缺失模型；预先执行 `prepare` 可以明确区分下载进度与 Web 服务就绪状态。不会自动下载全部 181 个角色，角色在界面按需下载或导入。

首次下载耗时由网络决定。默认六模型链的内存、显存需求与本地版本相同，请同时为宿主机和 Docker/WSL 留出资源。CPU 整曲处理较慢；显存不足会明确报错，镜像不会降低精度、模型数量、重叠参数或关闭 Karaoke。

多卡服务器默认分配一张 NVIDIA 卡。需要指定显卡时，将 `count: 1` 替换为 `device_ids: ['1']` 等实际设备编号；按 Docker 要求，这两项不能同时设置。模型和音频会持续占用 `/data`，请按自己的保存需求备份、整理，镜像不会自动删除用户作品。

## 数据、权限与备份

两个 Compose 文件均固定项目名 `ai-rvc`，默认卷名为 `ai-rvc_ai-rvc-data`；更换版本目录、停止或重建容器会保留数据。需要多实例时显式使用不同的 `-p` 项目名和端口，每个实例各有自己的卷；升级时继续使用原项目名。

| 容器路径 | 内容 |
|---|---|
| `/data/config.json` | 用户语言和翻唱参数；首次创建，后续保留 |
| `/data/assets` | 基础、分离、角色权重与索引 |
| `/data/outputs` | 成品和七类分轨输出 |
| `/data/cache` | Hugging Face、Torch 等模型缓存 |
| `/data/temp` | 上传和处理中间文件 |
| `/data/logs` | 应用写入的日志；标准输出另由 Docker 收集 |

容器以 UID/GID `1000:1000` 运行。命名卷在首次创建时自动继承镜像目录权限。NAS 或宿主机目录绑定时，先建立空目录，并让运行用户具有读写权限，再把 `ai-rvc-data:/data` 改为 `/你的路径/ai-rvc-data:/data`。可以为已有目录授予合适的 ACL；不要盲目递归修改共享目录所有者，不需要特权模式。权限不符会停止并报告路径。

`/app/configs` 内含程序模块，不能用旧目录整体覆盖。镜像只把 `config.json` 链接到数据卷，语言保存和预设应用继续使用经过校验的原子写入。`AI_RVC_DEVICE` 由所选 Compose 文件明确设置并优先于持久化配置中的设备值，因此容器界面的设备选择和保存按钮只读，保存接口也会明确拒绝修改。更换计算设备时使用对应镜像和 Compose 文件重建容器；其他推理参数保持原值。本地未固定启动设备时仍可在界面修改并保存。

先确认队列空闲，再备份；以下示例适用于默认 NVIDIA 文件，CPU 用户在 `compose` 后加 `-f compose.cpu.yaml`：

```bash
docker compose stop
docker compose run --rm --no-deps -T --entrypoint tar ai-rvc -C /data -czf - . > ai-rvc-data.tar.gz
docker compose start
```

大文件备份请在 Bash、WSL 或 PowerShell 7.4+ 执行。Windows PowerShell 5.1 的原生输出重定向会破坏二进制内容，应从 WSL 使用上面的命令。恢复到新的空卷，确认已备份原卷后执行：

```bash
docker compose run --rm --no-deps -T --entrypoint tar ai-rvc -C /data -xzf - < ai-rvc-data.tar.gz
```

普通 `docker compose down` 保留数据；`down -v` 会删除命名卷，含模型和全部结果，不能作为升级步骤。

## 登录、局域网与反向代理

默认仅绑定 `127.0.0.1`。需要调整端口时，将 `docker/compose.env.example` 复制为根目录 `.env`，设置 `AI_RVC_PORT=8080`。局域网使用时把 `AI_RVC_BIND` 设置为指定网卡的内网 IP，并启用登录。

在 `.env` 设置 `AI_RVC_AUTH_USER`，把密码保存到 `docker/secrets/password.txt`，然后启动：

```bash
docker compose -f compose.yaml -f compose.auth.yaml up -d
# CPU 将第一个文件换为 compose.cpu.yaml
```

密码以 Compose secret 文件注入，不写入镜像、仓库或命令行。缺少用户名、密码文件或密码为空时启动失败。不要公开分享 `.env`、密码文件或数据卷。Gradio 登录适合可信用户共用服务，处理队列与文件存储不提供多租户隔离。

通过现有 HTTPS 反向代理访问时，代理到 `127.0.0.1:7860`，保留流式响应和 Range 请求，按歌曲大小设置上传限制。子路径部署可设 `AI_RVC_ROOT_PATH=/ai-rvc`，代理须将该前缀剥离后转发；浏览器访问对应子路径。当前版本的代理适配情况以[平台验收记录](平台适配与验收.md)为准。

## 升级与离线使用

升级时使用明确版本标签，避免在转换过程中升级。GHCR 标签格式为 `ghcr.io/mason369/ai-rvc:<版本>-cuda` 或 `<版本>-cpu`；默认已经绑定本版镜像，需要其他已发布版本时才在 `.env` 设置 `AI_RVC_IMAGE`。执行：

```bash
docker compose stop
# 先按上文备份，再拉取已发布的明确版本
docker compose pull
docker compose up -d --no-build
docker compose run --rm ai-rvc check
```

源码构建用户更新到确定的源码版本后执行 `docker compose build`，再 `up -d`。新镜像只初始化不存在的配置，不覆盖用户参数；旧配置若含已删除的参数会明确报错，应对照默认配置和[参数说明](有效性与模型兼容性.md)修正，不能自动替换用户配置。

本版正式镜像通过 GHCR 分发，Release 不附离线镜像归档。自行导出的镜像归档可用 `docker load -i AI-RVC-版本-cuda.tar.gz` 导入，随后按对应版本的 Compose 文件启动。镜像不含用户角色权重，也不捆绑大体积默认分离权重；完全离线时需同时携带已经运行过 `prepare` 的 `/data` 备份，再导入角色模型。六个默认分离模型均固定提交号和 SHA-256；已校验的本地文件直接使用，损坏文件报错，不查询远端 `main`。两个固定版本官方 RVC 源码已随镜像提供。

Windows 已安装 Docker Desktop/CLI 时，也可运行 `powershell -ExecutionPolicy Bypass -File tools/Start-Docker.ps1`，CPU 使用 `-Variant cpu`。脚本优先校验并导入同目录随包镜像；源码目录没有镜像归档时执行构建，然后准备模型、启动并等待健康检查。需要登录时，先按上文设置用户名和密码文件，再附加 `-WithAuth`。

## 诊断

```bash
docker compose ps
docker compose logs --tail 200
docker compose run --rm ai-rvc check
docker compose run --rm ai-rvc python -m pip check
docker compose run --rm ai-rvc python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

健康检查仅在模型检查完成、WebUI 返回 HTTP 200 后通过；启用登录时登录页也能接受健康检查。健康不表示某首歌推理一定成功。服务退出最多自动重试 3 次，失败终态由 `compose ps` 和日志显示；修正原因后手动 `up -d`。宿主机重启后需再次启动 Compose，默认没有配置无限重启。

依赖清单保存在镜像 `/opt/ai-rvc/python-packages.txt`，版本和源码提交记录在 OCI 标签。本地未提交候选的 revision 为 `local`，对应文件内容以随包 `SOURCE-MANIFEST.json` 为准；正式工作流使用实际提交号。当前本地实测、云端及其他 GPU 的边界见[平台适配与验收](平台适配与验收.md)。

当前 fairseq 0.12.2 的旧依赖元数据需要 pip 24.0 / setuptools 69.5.1，构建已固定这组工具。`pip check` 通过，但 pip 和 PyTorch 仍可能显示第三方弃用提示；日志没有屏蔽这些提示，也不应自行升级安装工具后继续沿用本轮验收结论。离线使用前应完整准备数据卷；DNS、网络或权限失败会保留原始错误。
