"""Do not launch a frozen executable as if it were a Python interpreter."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from lib.worker_launch import build_worker_command, dispatch_worker


class WorkerLaunchTests(unittest.TestCase):
    def test_source_vc_and_mcp_use_the_existing_python_entries(self):
        with mock.patch.object(sys, "frozen", False, create=True):
            self.assertTrue(build_worker_command("vc")[1].endswith("official_upstream_runner.py"))
            self.assertEqual(build_worker_command("mcp-convert")[1:], ["-m", "rvc_mcp.worker"])

    def test_frozen_worker_uses_explicit_dispatch(self):
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertEqual(build_worker_command("vc", "--help"), [sys.executable, "--internal-worker", "vc", "--help"])

    def test_unknown_worker_is_rejected(self):
        with self.assertRaises(ValueError):
            build_worker_command("arbitrary.py")
        with self.assertRaises(ValueError):
            dispatch_worker("arbitrary.py", [])

    def test_worker_help_dispatch_does_not_start_the_web_server(self):
        for worker, argument in (("vc", "--sid"), ("uvr5", "--model-name")):
            result = subprocess.run([sys.executable, "run.py", "--internal-worker", worker, "--help"],
                                    capture_output=True, text=True, encoding="utf-8", check=True)
            self.assertIn(argument, result.stdout)
            self.assertNotIn("--share", result.stdout)

    def test_mcp_worker_dispatch_reports_real_failure(self):
        result = subprocess.run([sys.executable, "run.py", "--internal-worker", "mcp-convert"],
            input=json.dumps({"input_path": "missing.wav", "output_path": "out.wav", "model_name": "missing"}),
            capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertFalse(json.loads(result.stdout)["success"])

    def test_default_audit_forwards_silence_configuration(self):
        from tools.default_quality_audit import build_default_cover_kwargs
        config = json.loads(Path("configs/config.json").read_text(encoding="utf-8"))
        config["cover"].update(silence_gate=True, silence_threshold_db=-44, silence_smoothing_ms=37, silence_min_duration_ms=123)
        kwargs = build_default_cover_kwargs({"name": "case", "role_id": "role", "tags": ["test"],
            "input_audio": "input.wav", "model_path": "model.pth"}, config, Path("output"))
        for key in ("silence_gate", "silence_threshold_db", "silence_smoothing_ms", "silence_min_duration_ms"):
            self.assertEqual(kwargs[key], config["cover"][key])


if __name__ == "__main__":
    unittest.main()
