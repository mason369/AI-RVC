# -*- coding: utf-8 -*-
"""
MCP 服务器 - 为 Claude Code 提供 AI 翻唱工具
"""
import asyncio
import json
import os
import sys
from contextlib import redirect_stdout
import jsonschema
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult
from lib.worker_launch import build_worker_command

from rvc_mcp.tools import (
    list_models,
    convert_voice,
    download_model,
    get_model_status
)

# 创建 MCP 服务器
server = Server("rvc-voice-conversion")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="list_voice_models",
            description="列出所有可用的 RVC 语音模型",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        ),
        Tool(
            name="convert_voice",
            description="使用 RVC 模型转换输入人声的音色；不包含人声分离或伴奏混音。输出为 WAV。",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "输入音频文件的绝对路径"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "输出音频文件的绝对路径"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "list_voice_models 返回的唯一 id；仅当文件名唯一时也可使用 name"
                    },
                    "pitch_shift": {
                        "type": "integer",
                        "description": "音调偏移 (半音)，正数升调，负数降调",
                        "minimum": -24,
                        "maximum": 24,
                        "default": 0
                    },
                    "index_ratio": {
                        "type": "number",
                        "description": "索引混合比率 (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5
                    },
                    "rms_mix_rate": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.25,
                                     "description": "原人声响度包络匹配量，0 为关闭、1 为完全匹配"},
                    "protect": {"type": "number", "minimum": 0, "maximum": 0.5, "default": 0.33,
                                "description": "F0 模型的清辅音保护系数；0.5 关闭，仅使用索引时生效"},
                    "speaker_id": {"type": "integer", "minimum": 0, "default": 0,
                                   "description": "模型内的说话人 ID，必须小于模型真实说话人数"},
                    "f0_method": {"type": "string", "enum": ["rmvpe", "pm", "fcpe"], "default": "rmvpe",
                                  "description": "F0 提取方法，仅 F0 模型使用，不会自动切换算法"}
                },
                "additionalProperties": False,
                "required": ["input_path", "output_path", "model_name"]
            }
        ),
        Tool(
            name="download_base_models",
            description="下载 RVC 所需的基础模型 (HuBERT, RMVPE)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_model_status",
            description="获取基础模型的下载状态",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent] | CallToolResult:
    """执行工具调用"""
    available = {tool.name: tool for tool in await list_tools()}
    if name not in available:
        raise ValueError(f"未知工具: {name}")
    schema = {**available[name].inputSchema, "additionalProperties": False}
    jsonschema.validate(arguments, schema)

    if name == "list_voice_models":
        models = list_models()
        result = {
            "models": models,
            "count": len(models)
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "convert_voice":
        # Native inference DLL initialization must not share a process with
        # Windows stdio reader threads. The worker owns its numerical runtime.
        process = await asyncio.create_subprocess_exec(
            *build_worker_command("mcp-convert"), cwd=Path(__file__).resolve().parents[1],
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=sys.stderr,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        stdout, _ = await process.communicate(json.dumps(arguments, ensure_ascii=False).encode("utf-8"))
        if process.returncode != 0:
            raise RuntimeError(f"MCP 转换子进程失败，退出码 {process.returncode}")
        result = json.loads(stdout.decode("utf-8"))
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))],
            isError=result.get("success") is not True,
        )

    elif name == "download_base_models":
        result = download_model()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))],
            isError=result.get("success") is not True,
        )

    elif name == "get_model_status":
        status = get_model_status()
        return [TextContent(type="text", text=json.dumps(status, ensure_ascii=False, indent=2))]

    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


async def main():
    """启动 MCP 服务器"""
    # Windows: initialize native numerical DLLs on the main thread before
    # stdio starts its blocking reader threads. Lazy worker-thread import of
    # NumPy/OpenBLAS can deadlock with those readers during DLL initialization.
    with redirect_stdout(sys.stderr):
        from infer import official_adapter, pipeline  # noqa: F401
    # Capture the protocol stream first; redirect only application log output.
    async with stdio_server() as (read_stream, write_stream):
        with redirect_stdout(sys.stderr):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )


if __name__ == "__main__":
    asyncio.run(main())
