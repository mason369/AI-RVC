"""Local real-weight acceptance; results are evidence, not audio-quality scores."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

from infer.contracts import inspect_checkpoint
from infer.official_adapter import convert_vocals_official_upstream
from lib.console_i18n import console_print as print


def audio_evidence(path: Path) -> dict:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    info = sf.info(path)
    if data.size == 0 or not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-6:
        raise RuntimeError(f"验收输出为空、非有限值或无有效信号：{path}")
    return {"path": str(path), "sample_rate": sr, "frames": len(data), "channels": data.shape[1],
            "subtype": info.subtype, "peak": float(np.max(np.abs(data))),
            "rms": float(np.sqrt(np.mean(np.square(data), dtype=np.float64))),
            "not_pcm16_grid_ratio": float(np.mean(np.abs(data * 32768 - np.rint(data * 32768)) > 1e-4))}


async def verify_mcp(output: Path, input_path: Path) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable, args=["-m", "rvc_mcp.server"],
                                   cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print("MCP: 连接客户端", flush=True)
    async with stdio_client(params) as (read, write), ClientSession(
        read, write, read_timeout_seconds=timedelta(seconds=30)
    ) as client:
        await client.initialize()
        print("MCP: 握手通过", flush=True)
        tools = await client.list_tools()
        print("MCP: 工具清单通过", flush=True)
        invalid = await client.call_tool("convert_voice", {"input_path": str(input_path),
            "output_path": str(output / "invalid.wav"), "model_name": "Dia2", "filter_radius": 3})
        if not invalid.isError or (output / "invalid.wav").exists():
            raise RuntimeError("MCP 未拒绝未知参数")
        print("MCP: 未知参数已拒绝", flush=True)
        result = await client.call_tool("convert_voice", {"input_path": str(input_path),
            "output_path": str(output / "mcp_v2.wav"), "model_name": "characters/dia_v2/Dia2.pth", "index_ratio": 0})
        payload = json.loads(result.content[0].text)
        if result.isError or not payload.get("success"):
            raise RuntimeError(f"MCP 真实转换失败：{payload}")
        print("MCP: 真实转换通过", flush=True)
        return {"tools": [tool.name for tool in tools.tools], "unknown_parameter_rejected": True,
                "conversion": audio_evidence(output / "mcp_v2.wav")}


def main() -> int:
    parser = argparse.ArgumentParser(description="使用真实权重验证模型维度与 MCP")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"cases": [], "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    for role, filename, device in [("yoshiko_v2", "Yoshiko2", "cpu"), ("dia_v2", "Dia2", "cuda")]:
        model = ROOT / "assets/weights/characters" / role / f"{filename}.pth"
        contract = inspect_checkpoint(torch.load(model, map_location="cpu", weights_only=False), str(model))
        out = args.output / f"official_{contract.version}_{device}.wav"
        started = time.perf_counter()
        convert_vocals_official_upstream(str(args.input), str(out), str(model), None, "rmvpe", 0, 0, 0, .33, device=device)
        report["cases"].append({"backend": "official", "version": contract.version,
            "feature_dim": contract.feature_dim, "device": device, "seconds": time.perf_counter() - started,
            "audio": audio_evidence(out)})
        if args.local:
            from infer.pipeline import VoiceConversionPipeline
            pipeline = VoiceConversionPipeline(device=device)
            try:
                pipeline.load_voice_model(str(model))
                pipeline.load_hubert(str(ROOT / "assets/hubert/hubert_base.pt"))
                pipeline.load_f0_extractor("rmvpe", str(ROOT / "assets/rmvpe/rmvpe.pt"))
                local_out = args.output / f"local_{contract.version}_{device}.wav"
                pipeline.convert(str(args.input), str(local_out), index_ratio=0, silence_gate=False)
                report["cases"].append({"backend": "local", "version": contract.version,
                    "feature_dim": contract.feature_dim, "device": device, "audio": audio_evidence(local_out)})
            finally:
                pipeline.unload_all()
        (args.output / "runtime_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.mcp:
        report["mcp"] = asyncio.run(verify_mcp(args.output, args.input))
    (args.output / "runtime_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
