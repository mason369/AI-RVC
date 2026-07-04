import tempfile
import unittest
import zipfile
from pathlib import Path

from tools import character_models


class CustomCharacterModelTests(unittest.TestCase):
    def test_import_custom_pth_and_index_is_listed_as_downloaded(self):
        original_get_project_root = character_models.get_project_root
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            uploads = root / "uploads"
            uploads.mkdir()
            weight = uploads / "voice.pth"
            index = uploads / "voice.index"
            weight.write_bytes(b"not-a-real-checkpoint")
            index.write_bytes(b"not-a-real-index")

            character_models.get_project_root = lambda: root
            try:
                record = character_models.import_custom_character_model(
                    model_file=str(weight),
                    index_file=str(index),
                    display_name="测试角色",
                    source="测试来源",
                    category="测试分类",
                )
                downloaded = character_models.list_downloaded_characters()
            finally:
                character_models.get_project_root = original_get_project_root

        self.assertEqual(record["name"], "测试角色")
        self.assertEqual(record["series"], "测试分类")
        self.assertEqual(record["source"], "测试来源")
        self.assertTrue(Path(record["model_path"]).name.endswith(".pth"))
        self.assertTrue(Path(record["index_path"]).name.endswith(".index"))
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(downloaded[0]["name"], "测试角色")
        self.assertEqual(downloaded[0]["series"], "测试分类")

    def test_import_zip_rejects_multiple_pth_files_and_cleans_created_dir(self):
        original_get_project_root = character_models.get_project_root
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            uploads = root / "uploads"
            uploads.mkdir()
            archive = uploads / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zip_ref:
                zip_ref.writestr("a.pth", b"a")
                zip_ref.writestr("b.pth", b"b")

            character_models.get_project_root = lambda: root
            try:
                with self.assertRaisesRegex(ValueError, "多个 .pth"):
                    character_models.import_custom_character_model(
                        model_file=str(archive),
                        display_name="bad",
                    )
                target_dir = character_models.get_character_models_dir() / "bad"
            finally:
                character_models.get_project_root = original_get_project_root

        self.assertFalse(target_dir.exists())


if __name__ == "__main__":
    unittest.main()
