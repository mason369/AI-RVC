"""Container lifecycle only; inference and quality policy use the shared application."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def initialize_data(data: Path, default_config: Path, device: str) -> Path:
    """Seed a new volume without replacing existing user settings or recordings."""
    from configs.schema import validate_config

    for name in ("assets/weights/characters", "outputs", "temp/gradio", "logs", "cache"):
        (data / name).mkdir(parents=True, exist_ok=True)
    # Check write access even when a pre-existing read-only directory can be listed.
    with tempfile.TemporaryFile(dir=data):
        pass
    config_path = data / "config.json"
    if not config_path.exists():
        config = json.loads(default_config.read_text(encoding="utf-8"))
        config["device"] = device
        validate_config(config)
        with config_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=4)
            stream.write("\n")
    validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    return config_path


def authentication(environ) -> tuple[str, str] | None:
    user = environ.get("AI_RVC_AUTH_USER", "")
    password_file = environ.get("AI_RVC_AUTH_PASSWORD_FILE", "")
    if bool(user) != bool(password_file):
        raise ValueError("登录保护必须同时设置 AI_RVC_AUTH_USER 和 AI_RVC_AUTH_PASSWORD_FILE")
    if not user:
        return None
    password = Path(password_file).read_text(encoding="utf-8").rstrip("\r\n")
    if not password:
        raise ValueError("登录密码文件不能为空")
    return user, password


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args.pop(0) if args else "serve"
    variant = os.environ.get("AI_RVC_VARIANT", "cuda")
    device = os.environ.get("AI_RVC_DEVICE", variant)
    if variant not in {"cpu", "cuda"} or device not in {"cpu", "cuda"}:
        raise ValueError("Docker 镜像只支持显式 cpu / cuda；请使用对应镜像和 Compose 文件")
    if variant == "cpu" and device != "cpu":
        raise ValueError("CPU 镜像不能选择 CUDA，请改用 CUDA 镜像")
    os.environ["AI_RVC_DEVICE"] = device
    os.chdir(ROOT)
    initialize_data(Path("/data"), Path("/opt/ai-rvc/default-config.json"), device)

    if command not in {"serve", "prepare", "check"}:
        # Useful for diagnostics, backups and model import; preserves real exit codes.
        os.execvp(command, [command, *args])
    if args:
        raise ValueError(f"{command} 不接受额外参数；服务参数通过 docker/compose.env.example 中的环境变量设置")

    from run import check_environment, check_models
    if command == "prepare":
        return 0 if check_models() else 1
    from lib.device import get_device
    selected = get_device(device)
    # A real allocation catches a detected but unusable CUDA runtime before downloads/UI.
    import torch
    probe = torch.ones(1, device=selected)
    if probe.cpu().item() != 1:
        raise RuntimeError("设备计算自检失败")
    del probe
    if command == "check":
        # The existing checker exposes its own required/default model result and exit status.
        os.execv(sys.executable, [sys.executable, "tools/download_models.py", "--check"])

    auth = authentication(os.environ)
    root_path = os.environ.get("AI_RVC_ROOT_PATH", "")
    if root_path and (not root_path.startswith("/") or root_path.endswith("/")):
        raise ValueError("AI_RVC_ROOT_PATH 应为 /ai-rvc 这样的路径，不能以 / 结尾")
    if not check_environment() or not check_models():
        return 1
    from ui.app import launch
    launch(host="0.0.0.0", port=7860, share=False, open_browser=False,
           auth=auth, root_path=root_path or None)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Docker 启动失败：{exc}", file=sys.stderr, flush=True)
        raise
