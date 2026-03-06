# -*- coding: utf-8 -*-
"""
MCP 服务器 - 为 Claude Code 提供 AI 翻唱工具
"""
import asyncio
import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp.tools import (
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
                "required": []
            }
        ),
        Tool(
            name="convert_voice",
            description="使用 RVC 模型进行 AI 翻唱",
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
                        "description": "要使用的语音模型名称"
                    },
                    "pitch_shift": {
                        "type": "number",
                        "description": "音调偏移 (半音)，正数升调，负数降调",
                        "default": 0
                    },
                    "index_ratio": {
                        "type": "number",
                        "description": "索引混合比率 (0-1)",
                        "default": 0.5
                    }
                },
                "required": ["input_path", "output_path", "model_name"]
            }
        ),
        Tool(
            name="download_base_models",
            description="下载 RVC 所需的基础模型 (HuBERT, RMVPE)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_model_status",
            description="获取基础模型的下载状态",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """执行工具调用"""

    if name == "list_voice_models":
        models = list_models()
        result = {
            "models": models,
            "count": len(models)
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "convert_voice":
        result = convert_voice(
            input_path=arguments["input_path"],
            output_path=arguments["output_path"],
            model_name=arguments["model_name"],
            pitch_shift=arguments.get("pitch_shift", 0),
            index_ratio=arguments.get("index_ratio", 0.5),
            filter_radius=arguments.get("filter_radius", 3),
            rms_mix_rate=arguments.get("rms_mix_rate", 0.25),
            protect=arguments.get("protect", 0.33)
        )
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "download_base_models":
        result = download_model()
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    elif name == "get_model_status":
        status = get_model_status()
        return [TextContent(type="text", text=json.dumps(status, ensure_ascii=False, indent=2))]

    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


async def main():
    """启动 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
