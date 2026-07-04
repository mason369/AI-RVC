import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from infer import official_adapter
from infer.pipeline import VoiceConversionPipeline
from infer.rvc_version import infer_rvc_model_version, inspect_rvc_model_version
from tools import download_models


class OfficialAdapterTests(unittest.TestCase):
    def _write_minimal_checkpoint(self, path: Path) -> None:
        checkpoint = {
            "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 1, 256, 40000],
            "weight": {
                "enc_p.emb_phone.weight": torch.zeros(192, 256),
                "emb_g.weight": torch.zeros(1, 256),
            },
            "f0": 1,
        }
        torch.save(checkpoint, path)

    def test_isolated_argv_restores_cli_args(self):
        original = ["runner.py", "--input", "song.wav"]
        with mock.patch.object(sys, "argv", original[:]):
            with official_adapter._IsolatedArgv():
                self.assertEqual(sys.argv, ["runner.py"])
            self.assertEqual(sys.argv, original)

    def test_audio_activity_stats_detects_silent_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "silent.wav"
            sf.write(path, np.zeros((3200, 2), dtype=np.float32), 16000)

            rms, peak, nonzero = official_adapter._get_audio_activity_stats(path)

        self.assertEqual(rms, 0.0)
        self.assertEqual(peak, 0.0)
        self.assertEqual(nonzero, 0)

    def test_detects_v1_from_weight_shape_when_version_is_none(self):
        checkpoint = {
            "version": None,
            "weight": {"enc_p.emb_phone.weight": torch.zeros(192, 256)},
        }

        info = inspect_rvc_model_version(checkpoint, "test model")

        self.assertEqual(info.version, "v1")
        self.assertEqual(info.source, "weight_shape")
        self.assertEqual(info.feature_dim, 256)

    def test_detects_v2_from_weight_shape_when_version_is_missing(self):
        checkpoint = {
            "weight": {"enc_p.emb_phone.weight": torch.zeros(192, 768)},
        }

        self.assertEqual(infer_rvc_model_version(checkpoint, "test model"), "v2")

    def test_official_export_writes_normalized_version_to_model_copy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            source = tmp_root / "BadMeta.pth"
            official_models = tmp_root / "official_models"
            official_indexes = tmp_root / "official_indexes"
            checkpoint = {
                "version": None,
                "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 1, 256, 40000],
                "weight": {
                    "enc_p.emb_phone.weight": torch.zeros(192, 256),
                    "emb_g.weight": torch.zeros(1, 256),
                },
                "f0": 1,
            }
            torch.save(checkpoint, source)

            sid, index_path = official_adapter.export_model_to_official(
                official_models,
                official_indexes,
                str(source),
                None,
            )

            exported = torch.load(official_models / sid, map_location="cpu", weights_only=False)
            self.assertEqual(index_path, None)
            self.assertEqual(exported["version"], "v1")

    def test_official_export_refreshes_same_size_changed_index_copy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            source = tmp_root / "V1Model.pth"
            index_path = tmp_root / "V1Model.index"
            official_models = tmp_root / "official_models"
            official_indexes = tmp_root / "official_indexes"
            official_indexes.mkdir()
            target_index = official_indexes / "V1Model.index"
            index_path.write_bytes(b"new!")
            target_index.write_bytes(b"old?")
            self._write_minimal_checkpoint(source)

            fake_index = mock.Mock()
            fake_index.d = 256
            with mock.patch("infer.official_adapter.faiss.read_index", return_value=fake_index):
                _, copied_index = official_adapter.export_model_to_official(
                    official_models,
                    official_indexes,
                    str(source),
                    str(index_path),
                )

            self.assertEqual(Path(copied_index).read_bytes(), b"new!")

    def test_official_export_rejects_mismatched_index_dim(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            source = tmp_root / "V1Model.pth"
            index_path = tmp_root / "bad.index"
            index_path.write_bytes(b"not a real faiss index")
            checkpoint = {
                "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 1, 256, 40000],
                "weight": {
                    "enc_p.emb_phone.weight": torch.zeros(192, 256),
                    "emb_g.weight": torch.zeros(1, 256),
                },
                "f0": 1,
            }
            torch.save(checkpoint, source)

            fake_index = mock.Mock()
            fake_index.d = 768
            with mock.patch("infer.official_adapter.faiss.read_index", return_value=fake_index):
                with self.assertRaisesRegex(ValueError, "索引维度与模型不匹配"):
                    official_adapter.export_model_to_official(
                        tmp_root / "official_models",
                        tmp_root / "official_indexes",
                        str(source),
                        str(index_path),
                    )

    def test_current_pipeline_rejects_mismatched_index_dim(self):
        pipe = VoiceConversionPipeline.__new__(VoiceConversionPipeline)
        pipe.model_feature_dim = 256
        fake_index = mock.Mock()
        fake_index.d = 768

        with mock.patch("infer.pipeline.faiss.read_index", return_value=fake_index):
            with self.assertRaisesRegex(ValueError, "索引维度与模型不匹配"):
                pipe.load_index("bad.index")

    def test_auto_index_resolution_requires_name_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            model = tmp_root / "Chika2.pth"
            wrong_index = tmp_root / "Riko2.index"
            model.write_bytes(b"model")
            wrong_index.write_bytes(b"index")

            self.assertIsNone(official_adapter._resolve_index_path(model, None))

    def test_auto_index_resolution_accepts_matching_single_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            model = tmp_root / "Hanamaru2.pth"
            index = tmp_root / "hanamaru2.index"
            model.write_bytes(b"model")
            index.write_bytes(b"index")

            self.assertEqual(official_adapter._resolve_index_path(model, None), index)

    def test_ensure_upstream_rvc_tree_clones_missing_tree(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)

            def fake_run(cmd, cwd, check, timeout):
                official_root = Path(cmd[-1])
                for rel_path in download_models.UPSTREAM_RVC_REQUIRED_FILES:
                    target = official_root / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("# ok\n", encoding="utf-8")

            with mock.patch("tools.download_models.shutil.which", return_value="git"):
                with mock.patch("tools.download_models.subprocess.run", side_effect=fake_run) as run_mock:
                    official_root = download_models.ensure_upstream_rvc_tree(tmp_root)

            self.assertTrue(download_models.check_upstream_rvc_tree(tmp_root))
            self.assertEqual(official_root, tmp_root / download_models.UPSTREAM_RVC_DIR)
            run_mock.assert_called_once()

    def test_ensure_upstream_rvc_tree_rejects_incomplete_existing_tree(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            (tmp_root / download_models.UPSTREAM_RVC_DIR).mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "不完整"):
                download_models.ensure_upstream_rvc_tree(tmp_root)


if __name__ == "__main__":
    unittest.main()
