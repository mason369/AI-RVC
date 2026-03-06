# -*- coding: utf-8 -*-
"""
RVC AI 翻唱 - 主入口
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from lib.logger import log


def check_environment():
    """检查运行环境"""
    log.header("RVC AI 翻唱系统")

    # 检查 Python 版本
    py_version = sys.version_info
    log.info(f"Python 版本: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        log.warning("建议使用 Python 3.8 或更高版本")

    # 检查 PyTorch
    try:
        import torch
        log.info(f"PyTorch 版本: {torch.__version__}")

        from lib.device import get_device_info, _is_rocm, _has_xpu, _has_directml, _has_mps
        info = get_device_info()
        log.info(f"可用加速后端: {', '.join(info['backends'])}")

        if torch.cuda.is_available():
            backend = "ROCm" if _is_rocm() else "CUDA"
            log.info(f"{backend} 版本: {torch.version.hip if _is_rocm() else torch.version.cuda}")
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        elif _has_xpu():
            log.info(f"Intel GPU: {torch.xpu.get_device_name(0)}")
        elif _has_directml():
            import torch_directml
            log.info(f"DirectML 设备: {torch_directml.device_name(0)}")
        elif _has_mps():
            log.info("Apple MPS 加速可用")
        else:
            log.warning("未检测到 GPU 加速，将使用 CPU")
    except ImportError:
        log.error("未安装 PyTorch")
        return False

    return True


def check_models():
    """检查必需模型"""
    from tools.download_models import check_model, REQUIRED_MODELS

    missing = []
    for name in REQUIRED_MODELS:
        if not check_model(name):
            missing.append(name)

    if missing:
        log.warning(f"缺少必需模型: {', '.join(missing)}")
        log.info("正在下载...")
        from tools.download_models import download_required_models
        if not download_required_models():
            log.error("模型下载失败，请检查网络连接")
            return False

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RVC AI 翻唱系统")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务器地址 (默认: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务器端口 (默认: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公共链接"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过环境检查"
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="仅下载模型"
    )

    args = parser.parse_args()

    # 仅下载模型
    if args.download_models:
        from tools.download_models import download_all_models
        download_all_models()
        return

    # 环境检查
    if not args.skip_check:
        if not check_environment():
            sys.exit(1)

    # 模型检查
    if not check_models():
        log.info("提示: 可以使用 --skip-check 跳过检查")
        sys.exit(1)

    # 启动界面
    log.info(f"启动 Gradio 界面: http://{args.host}:{args.port}")
    from ui.app import launch
    launch(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
