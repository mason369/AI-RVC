import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from infer import official_adapter


class OfficialAdapterTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
