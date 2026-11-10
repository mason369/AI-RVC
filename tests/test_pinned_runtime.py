"""Contract tests for pinned assets and the current upstream API (no network)."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from infer.official_upstream_runner import configure_selected_device, save_vc_result, validate_runtime_precision
from tools import model_assets, upstream_runtime as runtime


class PinnedRuntimeTests(unittest.TestCase):
    def source_fixture(self, root, signature=None):
        # Deliberately independent of REQUIRED_FILES: a legacy-shaped tree must fail.
        files = ("configs/config.py", "infer/vc/pipeline.py", "infer/hubert.py", "infer/rmvpe.py")
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        signature = signature or "self, sid, input_audio_path, f0_up_key, f0_method, file_index, index_rate, resample_sr, rms_mix_rate, protect"
        (root / "infer/vc/modules.py").write_text(
            f"class VC:\n    def vc_single({signature}):\n        pass\n", encoding="utf-8"
        )
        (root / ".git").mkdir(exist_ok=True)

    def test_current_nine_argument_api_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source_fixture(root)
            with patch.object(runtime, "_git", side_effect=[runtime.REVISIONS["vc"], ""]):
                self.assertEqual(runtime.source_problems(root), [])

    def test_legacy_api_is_rejected_even_with_expected_file_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source_fixture(root, "self, sid, input_audio_path, f0_up_key, f0_file, f0_method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect")
            with patch.object(runtime, "_git", side_effect=[runtime.REVISIONS["vc"], ""]):
                self.assertIn("接口不兼容", " ".join(runtime.source_problems(root)))

    def test_wrong_commit_or_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.source_fixture(root)
            for results, message in ((["0" * 40], "版本不符"), ([runtime.REVISIONS["vc"], "infer/hubert.py"], "含修改")):
                with self.subTest(message=message), patch.object(runtime, "_git", side_effect=results):
                    self.assertIn(message, " ".join(runtime.source_problems(root)))

    def test_old_tree_does_not_satisfy_new_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "infer/modules/vc/modules.py"
            old.parent.mkdir(parents=True)
            old.write_text("# old tree", encoding="utf-8")
            self.assertIn("infer/vc/modules.py", runtime.source_problems(root))

    def test_fetch_uses_exact_commit_and_preserves_user_legacy_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            old = project / "_official_rvc/configs/config.py"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"user changes must survive")

            def git(root, *args, **kwargs):
                if args[0] == "checkout":
                    self.source_fixture(root)
                if args[0] == "rev-parse":
                    return runtime.REVISIONS["vc"]
                return ""

            with patch.object(runtime, "_git", side_effect=git) as calls:
                result = runtime.ensure_source(project)
            self.assertEqual(result, project / "_official_rvc_runtime/81eed5e8f68b6bed1789f682fe78cdd324495afc")
            fetch = [call for call in calls.call_args_list if call.args[1] == "fetch"]
            self.assertEqual(len(fetch), 1)
            self.assertEqual(fetch[0].args[2:], ("--depth", "1", "origin", "81eed5e8f68b6bed1789f682fe78cdd324495afc"))
            self.assertEqual(old.read_bytes(), b"user changes must survive")

    def test_failed_fetch_is_explicit_and_preserves_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            with patch.object(runtime, "_git", side_effect=RuntimeError("network failure")):
                with self.assertRaisesRegex(RuntimeError, "network failure"):
                    runtime.ensure_source(project)
            root = runtime.runtime_root(project)
            self.assertFalse(root.exists())
            self.assertTrue(root.with_name(root.name + ".fetching").is_dir())

    def test_uvr5_has_a_separate_immutable_runtime(self):
        self.assertNotEqual(runtime.runtime_root(Path("test")), runtime.runtime_root(Path("test"), "uvr5"))
        self.assertEqual(runtime.REVISIONS["uvr5"], "7ef19867780cf703841ebafb565a4e47d1ea86ff")

    def test_corrupt_local_asset_is_not_overwritten_or_redownloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"corrupt")
            digest = hashlib.sha256(b"correct").hexdigest()
            with patch("huggingface_hub.hf_hub_download") as download:
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    model_assets.download_pinned_asset("owner/model", path.name, path.parent, "1" * 40, digest)
                download.assert_not_called()
            self.assertEqual(path.read_bytes(), b"corrupt")

    def test_valid_local_asset_requires_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"correct")
            digest = hashlib.sha256(b"correct").hexdigest()
            with patch("huggingface_hub.hf_hub_download") as download:
                self.assertEqual(model_assets.download_pinned_asset("owner/model", path.name, path.parent, "1" * 40, digest), path)
                download.assert_not_called()

    def test_download_passes_commit_and_verifies_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "downloaded.bin"
            payload.write_bytes(b"weights")
            with patch("huggingface_hub.hf_hub_download", return_value=str(payload)) as download:
                result = model_assets.download_pinned_asset("owner/model", "requested.bin", root, "2" * 40, hashlib.sha256(b"weights").hexdigest())
            self.assertEqual(result, payload)
            self.assertEqual(download.call_args.kwargs["revision"], "2" * 40)

    def test_missing_transformers_files_are_not_satisfied_by_fairseq_pt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "assets/hubert/hubert_base.pt"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"legacy")
            self.assertEqual(set(runtime.hubert_problems(root)), {
                "assets/hubert_base/config.json", "assets/hubert_base/preprocessor_config.json",
                "assets/hubert_base/pytorch_model.bin",
            })

    def test_invalid_vc_result_cannot_replace_existing_output(self):
        invalid = [None, ("failed", None), ("failed", (None, None)),
                   ("bad", (0, np.ones(3))), ("bad", (16000, np.array([]))),
                   ("bad", (16000, np.array([np.nan])))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.wav"
            path.write_bytes(b"preserve me")
            for result in invalid:
                with self.subTest(result=str(result)), self.assertRaises(RuntimeError):
                    save_vc_result(result, path)
                self.assertEqual(path.read_bytes(), b"preserve me")

    def test_valid_vc_result_is_saved_as_real_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.wav"
            save_vc_result(("Success.", (16000, np.ones(1600, dtype=np.float32) * .1)), path)
            info = sf.info(path)
            self.assertEqual((info.frames, info.samplerate, info.channels), (1600, 16000, 1))
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_cpu_selection_overrides_a_cuda_default_before_config_creation(self):
        module = SimpleNamespace(infer_device=torch.device("cuda"), infer_dtype=torch.float16,
                                 configure_cuda_graph=lambda device: False)
        module.Config = lambda: SimpleNamespace(dtype=module.infer_dtype, is_half=module.infer_dtype == torch.float16)
        config = configure_selected_device(module, torch.device("cpu"))
        self.assertEqual(config.device, torch.device("cpu"))
        self.assertEqual(config.dtype, torch.float32)
        self.assertFalse(config.is_half)

    def test_neither_rvc_nor_rmvpe_can_override_upstream_fp32_policy(self):
        full = SimpleNamespace(is_half=False)
        half = SimpleNamespace(is_half=True)
        for rvc, rmvpe in ((half, full), (full, half), (half, half), (half, None)):
            with self.subTest(rvc=rvc, rmvpe=rmvpe), self.assertRaisesRegex(RuntimeError, "精度配置不一致"):
                validate_runtime_precision(full, rvc, rmvpe)
        validate_runtime_precision(full, full, full)
        validate_runtime_precision(full, full, None)
        validate_runtime_precision(half, full, half)
        validate_runtime_precision(half, half, full)

    def test_runner_imports_upstream_namespace_and_uses_checkpoint_speaker_count(self):
        from test_rvc_checkpoint_compatibility import _official_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("configs", "i18n", "tools", "infer/vc", "assets/weights"):
                (root / name).mkdir(parents=True)
            _, checkpoint = _official_checkpoint("v1", False)
            checkpoint["weight"] = {k: v.double() for k, v in checkpoint["weight"].items()}
            torch.save(checkpoint, root / "assets/weights/fixture.pth")
            (root / "configs/config.py").write_text("""import torch
infer_dtype = torch.float32
def configure_cuda_graph(device): return False
class Config:
    def __init__(self):
        self.dtype = infer_dtype
        self.is_half = infer_dtype == torch.float16
""", encoding="utf8")
            (root / "infer/hubert.py").write_text("""def load_hubert_model(device, is_half=False):
    assert str(device) == 'cpu' and not is_half
    return object()
""", encoding="utf8")
            (root / "infer/vc/pipeline.py").write_text("""import numpy as np
import torch
class Pipeline:
    def __init__(self, tgt_sr, config):
        assert not config.is_half and config.dtype == torch.float32
        self.tgt_sr = tgt_sr
        self.window = 160
        self.t_pad = 160
        self.t_pad2 = 320
        self.t_pad_tgt = 320
    def vc(self, model, net_g, sid, audio, pitch, pitchf, times, index, vectors, index_rate, version, protect):
        feats = torch.ones(1)
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            assert feats.dtype == torch.float64
            synthesized = torch.ones(1, 1, len(audio) * 2, dtype=feats.dtype) * .1
        return synthesized[0, 0].data.cpu().float().numpy()
    def pipeline(self, model, net_g, sid):
        audio = np.ones(1600)
        audio_pad = np.pad(audio, (160, 160))
        pitch = pitchf = index = index_vectors = None
        times, index_rate, version, protect, tgt_sr = [0,0,0], 0, 'v1', .33, self.tgt_sr
        opt_ts = [800]
        audio_opt = []
        for t in opt_ts:
            audio_opt.append(audio)
        audio_opt = np.concatenate(audio_opt)
        max_int16 = 32768
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        return audio_opt
""", encoding="utf8")
            (root / "infer/vc/modules.py").write_text("""class VC:
    def __init__(self, config):
        assert str(config.device) == 'cpu'
    def vc_single(self, sid, input_audio_path, f0_up_key, f0_method, file_index, index_rate, resample_sr, rms_mix_rate, protect):
        assert self.n_spk == 2 and sid == 1
        return 'Success.', (self.tgt_sr, self.pipeline.pipeline(self.hubert_model, self.net_g, sid))
""", encoding="utf8")
            project = Path(__file__).resolve().parents[1]
            output = root / "converted.wav"
            env = dict(os.environ, PYTHONPATH=str(project), PYTHONUTF8="1", weight_root=str(root / "assets/weights"))
            command = [sys.executable, str(project / "infer/official_upstream_runner.py"),
                       "--official-root", str(root), "--sid", "fixture.pth", "--vocals-path", "fixture.wav",
                       "--output-path", str(output), "--f0-method", "rmvpe", "--pitch-shift", "0",
                       "--index-rate", "0", "--rms-mix-rate", "1", "--protect", ".33",
                       "--speaker-id", "1", "--device", "cpu"]
            result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True,
                                    encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(sf.info(output).frames, 3200)
            self.assertEqual(sf.info(output).subtype, 'DOUBLE')
            command[command.index('--speaker-id') + 1] = '2'
            invalid = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True,
                                     encoding='utf8', timeout=60)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('id=2, count=2', invalid.stderr)


if __name__ == "__main__":
    unittest.main()
