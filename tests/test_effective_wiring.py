"""Public parameter, GUI and encoding wiring regression tests."""
import asyncio
import inspect
import json
import unittest
from pathlib import Path
from unittest import mock

import torch

from infer.contracts import validate_state_dict


class EffectiveWiringTests(unittest.TestCase):
    def test_nonfinite_or_quantized_weights_are_rejected(self):
        model = torch.nn.Linear(2, 2)
        for value in (torch.full((2, 2), float("nan")), torch.ones((2, 2), dtype=torch.int8)):
            with self.subTest(dtype=value.dtype), self.assertRaises(ValueError):
                validate_state_dict(model, {"weight": value, "bias": model.bias.detach()})

    def test_default_audit_only_passes_existing_cover_arguments(self):
        from infer.cover_pipeline import CoverPipeline
        from tools.default_quality_audit import build_default_cover_kwargs
        config = json.loads(Path("configs/config.json").read_text(encoding="utf-8"))
        kwargs = build_default_cover_kwargs({"name": "test", "role_id": "test", "tags": ["test"],
            "input_audio": "input.wav", "model_path": "model.pth"}, config, Path("outputs"))
        inspect.signature(CoverPipeline.process).bind(object(), **kwargs)

    def test_official_route_disables_inapplicable_controls(self):
        from ui.app import update_route_controls, get_source_constraint_option_maps
        karaoke, merge, source = update_route_controls("official", True)
        self.assertFalse(karaoke["value"])
        self.assertFalse(karaoke["interactive"])
        self.assertFalse(merge["interactive"])
        self.assertFalse(source["interactive"])
        label = get_source_constraint_option_maps()[1]["on"]
        self.assertEqual(update_route_controls("current", True, label)[2]["value"], label)

    def test_route_status_does_not_claim_karaoke_when_disabled(self):
        from ui.app import get_cover_vc_route_status
        self.assertNotIn("MVSep 9205", get_cover_vc_route_status("current", True, False))

    def test_demucs_does_not_retry_after_dropping_invalid_arguments(self):
        from infer.separator import VocalSeparator
        source = inspect.getsource(VocalSeparator.separate)
        self.assertNotIn("except TypeError", source)
        self.assertEqual(source.count("apply_model("), 1)
        for argument in ("shifts=self.shifts", "overlap=self.overlap", "split=self.split"):
            self.assertIn(argument, source)
        self.assertIn('subtype="FLOAT"', source)

    def test_mcp_rejects_unknown_fields_for_every_tool(self):
        from rvc_mcp.server import list_tools
        for tool in asyncio.run(list_tools()):
            with self.subTest(tool=tool.name):
                self.assertIs(tool.inputSchema.get("additionalProperties"), False)

    def test_initial_page_wires_model_capabilities_without_running_cover(self):
        from ui.app import create_ui
        source = inspect.getsource(create_ui)
        self.assertIn("fn=update_model_controls, inputs=[character_dropdown]", source)
        self.assertNotIn("app.load(fn=process_cover", source)


if __name__ == "__main__":
    unittest.main()
