# -*- coding: utf-8 -*-
import unittest

from infer.cover_pipeline import (
    _clean_input_stem_for_output,
    _clean_output_suffixes,
    _sanitize_filename_component,
)


class CoverOutputFilenameTests(unittest.TestCase):
    def test_gradio_temp_prefix_is_removed_from_output_stem(self):
        cases = [
            (
                r"C:\Users\ADMINI~1\AppData\Local\Temp\gradio\c7a5d706e8079979fa5e539ae7d62f62a711c58c"
                r"\C__Users_ADMINI~1_AppData_Local_Temp_gradio_c7a5d706e8079979fa5e539ae7d62f62a711c58c_小林愛香 - Far far away-0-100.wav",
                "小林愛香 - Far far away",
            ),
            (
                r"/tmp/gradio/hash/C__Users_ADMINI~1_AppData_Local_Temp_gradio_hash_Aqours - 未体験HORIZON (Live)-0-100.wav",
                "Aqours - 未体験HORIZON (Live)",
            ),
        ]

        for path, expected in cases:
            with self.subTest(path=path):
                self.assertEqual(_clean_input_stem_for_output(path), expected)

    def test_localized_download_name_can_match_ui_label(self):
        path = (
            r"C:\Users\ADMINI~1\AppData\Local\Temp\gradio\c7a5d706e8079979fa5e539ae7d62f62a711c58c"
            r"\C__Users_ADMINI~1_AppData_Local_Temp_gradio_c7a5d706e8079979fa5e539ae7d62f62a711c58c_シャイニーカラーズ - Dye the sky. -25 colors-0-100.wav"
        )
        song = _clean_input_stem_for_output(path)
        character = _sanitize_filename_component(
            "国木田花丸-Hanamaru Kunikida-Zurakichi v2",
            "model_display_name",
        )
        suffixes = _clean_output_suffixes({"accompaniment": "伴奏"})

        self.assertEqual(
            f"{song}_{character}_{suffixes['accompaniment']}.wav",
            "シャイニーカラーズ - Dye the sky. -25 colors_国木田花丸-Hanamaru Kunikida-Zurakichi v2_伴奏.wav",
        )


if __name__ == "__main__":
    unittest.main()
