import inspect
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ui import app


class BatchCoverTests(unittest.TestCase):
    def test_batch_and_single_expose_the_same_named_settings(self):
        def settings(handler):
            parameters = list(inspect.signature(handler).parameters)[1:]
            return [name for name in parameters if name not in {"progress", "reuse_state"}]

        self.assertEqual(settings(app.process_cover_batch), settings(app.process_cover_with_downloads))

    def test_empty_decoder_exception_displays_cause_and_logs_traceback(self):
        def fail(_):
            try:
                raise OSError("Unrecognized audio format")
            except OSError as cause:
                raise RuntimeError() from cause

        with patch.object(app, "resolve_character_name", side_effect=fail), patch.object(app.log, "error") as log:
            events = list(app.process_cover_batch(
                ["broken.wav"], "character", 0, 50, 0, True, True,
                "auto", "current", 100, 100, 0, 0, 0, "vocals", {},
            ))
        summary = events[-1][2]
        self.assertIn("Unrecognized audio format", summary)
        self.assertNotIn("Traceback (most recent call last)", summary)
        self.assertIn("Traceback (most recent call last)", log.call_args.args[0])

    def test_item_exception_preserves_prior_and_subsequent_downloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = {}

            def run(path, *settings, run_directory, reuse_state, **kwargs):
                if path == "broken.wav":
                    raise OSError("external decoder failure")
                folder = root / run_directory
                folder.mkdir(parents=True)
                track = folder / "same.wav"
                track.write_bytes(path.encode())
                reuse_state["selected_status"] = {"message": "vocals_complete_status"}
                return None, str(track), None, None, None, None, None, "done"

            with patch.object(app, "process_cover", side_effect=run):
                events = list(app.process_cover_batch(
                    ["a/same.wav", "broken.wav", "b/same.wav"], "character", 0, 50, 0,
                    True, True, "auto", "current", 100, 100, 0, 0, 0, "song", state,
                ))
            rows, picker, summary, download = events[-1]
            self.assertEqual([value for _, value in picker["choices"]], ["0", "2"])
            self.assertIn("external decoder failure", summary)
            self.assertIsNone(rows[1]["result"][1])
            with zipfile.ZipFile(download["value"]) as archive:
                self.assertEqual(archive.namelist(), ["001/same.wav", "003/same.wav"])
                self.assertEqual(archive.read("001/same.wav"), b"a/same.wav")
                self.assertEqual(archive.read("003/same.wav"), b"b/same.wav")
            selected = app.select_batch_result("2", rows, state)
            self.assertEqual(selected[1], rows[2]["result"][1])
            self.assertFalse(Path(download["value"]).with_suffix(".zip.part").exists())
