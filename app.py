#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hugging Face Spaces 入口文件
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from lib.ffmpeg_runtime import configure_ffmpeg_runtime

configure_ffmpeg_runtime()

# 设置环境变量
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"

# 导入并启动应用
if __name__ == "__main__":
    from ui.app import launch

    # 启动 Gradio 界面
    launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False
    )
