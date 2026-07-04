# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

import fastapi
from fastapi.testclient import TestClient
from starlette.responses import FileResponse

from ui import app as ui_app


class UiDownloadFilenameTests(unittest.TestCase):
    def test_gradio_temp_prefix_is_removed_before_returning_audio_path(self):
        dirty_name = (
            "C__Users_ADMINI~1_AppData_Local_Temp_gradio_"
            "6ee13f74e75c9ae30f81b76534459a9c5109d913_"
            "小林愛香 - Far far away_国木田花丸-Hanamaru Kunikida-v2_最终翻唱.wav"
        )
        expected_name = "小林愛香 - Far far away_国木田花丸-Hanamaru Kunikida-v2_最终翻唱.wav"

        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / dirty_name
            source.write_bytes(b"RIFF")

            clean_path = Path(ui_app.normalize_download_output_path(str(source)))

            self.assertEqual(clean_path.name, expected_name)
            self.assertTrue(clean_path.exists())
            self.assertEqual(clean_path.read_bytes(), b"RIFF")

    def test_clean_download_name_preserves_mixed_unicode_track_names(self):
        dirty_name = (
            "C__Users_ADMINI~1_AppData_Local_Temp_gradio_hash_"
            "シャイニーカラーズ - Dye the sky. -25 colors_"
            "国木田花丸-Hanamaru Kunikida-Zurakichi v2_伴奏.wav"
        )

        self.assertEqual(
            ui_app.clean_gradio_temp_download_name(dirty_name),
            "シャイニーカラーズ - Dye the sky. -25 colors_"
            "国木田花丸-Hanamaru Kunikida-Zurakichi v2_伴奏.wav",
        )

    def test_cover_download_buttons_follow_available_outputs(self):
        updates = ui_app._cover_download_button_updates(
            "cover.wav",
            None,
            "vocals.wav",
            "",
            None,
            "accompaniment.wav",
        )

        self.assertEqual(updates[0]["value"], "cover.wav")
        self.assertTrue(updates[0]["visible"])
        self.assertIsNone(updates[1]["value"])
        self.assertFalse(updates[1]["visible"])
        self.assertEqual(updates[2]["value"], "vocals.wav")
        self.assertTrue(updates[2]["visible"])
        self.assertFalse(updates[3]["visible"])
        self.assertEqual(updates[5]["value"], "accompaniment.wav")
        self.assertTrue(updates[5]["visible"])

    def test_gradio_file_route_patch_sets_header_on_actual_handler(self):
        app = fastapi.FastAPI()

        class Blocks:
            server_app = app

        dirty_name = (
            "C__Users_ADMINI~1_AppData_Local_Temp_gradio_"
            "8296d3afc17f5d2581a7d3f1627af96ff9cb50fb_"
            "小林愛香 - Far far away_国木田花丸-Hanamaru Kunikida-v2_最终翻唱.wav"
        )
        clean_name = "小林愛香 - Far far away_国木田花丸-Hanamaru Kunikida-v2_最终翻唱.wav"

        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / dirty_name
            source.write_bytes(b"RIFF")

            @app.get("/file={path_or_url:path}")
            async def file(path_or_url: str, request: fastapi.Request):
                return FileResponse(source, headers={"Accept-Ranges": "bytes"})

            ui_app._patch_gradio_file_download(Blocks())

            response = TestClient(app).get(f"/file={quote(str(source), safe='')}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["content-disposition"],
            f"inline; filename*=utf-8''{quote(clean_name, safe='')}",
        )


if __name__ == "__main__":
    unittest.main()
