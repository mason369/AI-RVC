"""GPU compute capability alone must not enable unsafe half inference."""
import ast
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from infer.rvc_checkpoint import select_rvc_runtime_precision


class RvcRuntimePrecisionTests(unittest.TestCase):
    def select_cuda(self, name, capability=(7, 5), source=("float16",)):
        with mock.patch.object(torch.version, "hip", None):
            return select_rvc_runtime_precision(
                source, "cuda:0", cuda_capability=capability, cuda_device_name=name,
            )

    def test_gtx_16_series_uses_fp32_even_though_sm_75_supports_half(self):
        for name in (
            "NVIDIA GeForce GTX 1630", "NVIDIA GeForce GTX 1650",
            "NVIDIA GeForce GTX 1650 SUPER", "NVIDIA GeForce GTX 1650 Ti",
            "NVIDIA GeForce GTX 1660", "NVIDIA GeForce GTX 1660 Ti",
            "NVIDIA GeForce GTX 1660 SUPER", "GeForce GTX 1660 Ti with Max-Q Design",
        ):
            with self.subTest(name=name):
                plan = self.select_cuda(name)
                self.assertEqual(plan.torch_dtype, torch.float32)
                self.assertFalse(plan.is_half)
                self.assertIn("GTX 16", plan.reason)

    def test_tensor_core_devices_keep_native_fp16(self):
        for name, capability in (
            ("NVIDIA Tesla V100", (7, 0)), ("NVIDIA GeForce RTX 2080 Ti", (7, 5)),
            ("NVIDIA T4", (7, 5)), ("NVIDIA GeForce RTX 3060", (8, 6)),
            ("NVIDIA GeForce RTX 4070 Ti SUPER", (8, 9)),
        ):
            with self.subTest(name=name):
                plan = self.select_cuda(name, capability)
                self.assertEqual(plan.torch_dtype, torch.float16)
                self.assertTrue(plan.is_half)

    def test_selected_cuda_index_controls_both_name_and_capability(self):
        with mock.patch.object(torch.version, "hip", None), \
             mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)) as capability, \
             mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1660 SUPER") as name:
            plan = select_rvc_runtime_precision(["float16"], "cuda:1")
        self.assertEqual(plan.torch_dtype, torch.float32)
        capability.assert_called_once_with(1)
        name.assert_called_once_with(1)

    def test_implicit_cuda_device_uses_the_current_device(self):
        with mock.patch.object(torch.version, "hip", None), \
             mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)) as capability, \
             mock.patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1660 SUPER") as name:
            self.assertFalse(select_rvc_runtime_precision(["float16"], "cuda").is_half)
        capability.assert_called_once_with(None)
        name.assert_called_once_with(None)

    def test_unknown_device_name_does_not_enable_half_from_sm_alone(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            plan = self.select_cuda(None)
        self.assertEqual(plan.torch_dtype, torch.float32)

    def test_pre_tensor_core_and_non_cuda_devices_widen_half(self):
        self.assertFalse(self.select_cuda("NVIDIA GeForce GTX 1080 Ti", (6, 1)).is_half)
        for device in ("cpu", "mps", "xpu", "privateuseone:0"):
            with self.subTest(device=device), mock.patch("torch.cuda.get_device_name") as name:
                self.assertEqual(select_rvc_runtime_precision(["float16"], device).torch_dtype, torch.float32)
                name.assert_not_called()

    def test_rocm_does_not_use_nvidia_sm_eligibility(self):
        with mock.patch.object(torch.version, "hip", "6.2"), \
             mock.patch("torch.cuda.get_device_name") as name:
            plan = select_rvc_runtime_precision(
                ["float16"], "cuda:0", cuda_capability=(9, 0), cuda_device_name="AMD GPU",
            )
        self.assertEqual(plan.torch_dtype, torch.float32)
        name.assert_not_called()

    def test_hardware_policy_never_narrows_fp32_fp64_or_mixed_weights(self):
        for name in ("NVIDIA GeForce GTX 1660 SUPER", "NVIDIA GeForce RTX 2080 Ti"):
            for source, expected in (
                (("float32",), torch.float32), (("float64",), torch.float64),
                (("bfloat16",), torch.float32), (("float16", "float32"), torch.float32),
                (("float16", "float64"), torch.float64),
            ):
                with self.subTest(name=name, source=source):
                    self.assertEqual(self.select_cuda(name, source=source).torch_dtype, expected)

    def test_gtx_widening_preserves_every_finite_half_value(self):
        plan = self.select_cuda("NVIDIA GeForce GTX 1660 SUPER")
        values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        values = values[torch.isfinite(values)]
        torch.testing.assert_close(values.to(plan.torch_dtype).half(), values, rtol=0, atol=0)

    def test_half_eligibility_agrees_with_the_actual_pinned_upstream_gpu_rule(self):
        from tools.upstream_runtime import runtime_root

        source = runtime_root(Path(__file__).resolve().parents[1], "vc") / "configs/config.py"
        if not source.is_file():
            self.skipTest("Pinned upstream checkout is not available")
        tree = ast.parse(source.read_text(encoding="utf8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_device_dtype_sm")
        namespace = {"torch": torch, "re": re, "logger": logging.getLogger(__name__)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        for name, capability in (
            ("NVIDIA GeForce GTX 1660 SUPER", (7, 5)), ("NVIDIA GeForce GTX 1650", (7, 5)),
            ("NVIDIA GeForce GTX 1080 Ti", (6, 1)), ("NVIDIA GeForce RTX 2080 Ti", (7, 5)),
            ("NVIDIA T4", (7, 5)), ("NVIDIA GeForce RTX 4070 Ti SUPER", (8, 9)),
        ):
            with self.subTest(name=name), \
                 mock.patch.object(torch.version, "hip", None), \
                 mock.patch("torch.cuda.is_available", return_value=True), \
                 mock.patch("torch.cuda.device_count", return_value=2), \
                 mock.patch("torch.cuda.get_device_capability", return_value=capability), \
                 mock.patch("torch.cuda.get_device_name", return_value=name), \
                 mock.patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=6 * 1024**3)):
                upstream = namespace["get_device_dtype_sm"](1)
                plan = select_rvc_runtime_precision(["float16"], "cuda:1")
                self.assertEqual(upstream[0], torch.device("cuda:1"))
                self.assertEqual(plan.torch_dtype, upstream[1])


if __name__ == "__main__":
    unittest.main()
