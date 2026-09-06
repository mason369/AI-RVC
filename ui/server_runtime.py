"""Select a supported socket event loop for the Windows browser server only.

Python 3.10 Proactor socket shutdown raises WinError 10054 when media clients
cancel range/metadata reads. The ASGI server needs sockets, not async pipes;
RVC subprocesses run synchronously in worker threads. Use Uvicorn's documented
custom loop factory, without changing the process-wide policy or error handler.
"""
import asyncio
import os

import gradio.http_server


def selector_loop_factory(use_subprocess: bool = False):
    return asyncio.SelectorEventLoop()


class BrowserServer(gradio.http_server.Server):
    def __init__(self, config, **kwargs):
        if os.name == "nt":
            config.loop = "ui.server_runtime:selector_loop_factory"
        super().__init__(config=config, **kwargs)


def configure_browser_server():
    if os.name == "nt":
        gradio.http_server.Server = BrowserServer
