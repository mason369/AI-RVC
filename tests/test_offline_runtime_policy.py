"""Offline execution policy and MCP failure protocol regressions."""
import asyncio
import inspect
import json
import unittest
from unittest import mock


class OfflineRuntimePolicyTests(unittest.TestCase):
    def test_offline_runner_selects_eager_before_upstream_configuration(self):
        from infer.official_upstream_runner import main
        source = inspect.getsource(main)
        policy = 'os.environ["RVC_CUDA_GRAPH"] = "0"'
        self.assertLess(source.index(policy), source.index("import configs.config"))

    def test_mcp_conversion_failure_sets_protocol_error(self):
        from rvc_mcp.server import call_tool
        process = mock.Mock(returncode=0)
        process.communicate = mock.AsyncMock(return_value=(
            json.dumps({"success": False, "output_path": None, "error": "missing model"}).encode(), None))
        with mock.patch("rvc_mcp.server.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)):
            result = asyncio.run(call_tool("convert_voice", {
                "input_path": "missing.wav", "output_path": "output.wav", "model_name": "missing"}))
        self.assertTrue(result.isError)
        self.assertFalse(json.loads(result.content[0].text)["success"])

    def test_mcp_download_failure_sets_protocol_error(self):
        from rvc_mcp.server import call_tool
        with mock.patch("rvc_mcp.server.download_model", return_value={"success": False, "error": "download failed"}):
            self.assertTrue(asyncio.run(call_tool("download_base_models", {})).isError)


if __name__ == "__main__":
    unittest.main()
