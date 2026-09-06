import subprocess
import unittest
from pathlib import Path
from unittest import mock

from tools.upstream_runtime import _git


class UpstreamGitDiagnosticsTests(unittest.TestCase):
    def test_git_failure_includes_stderr_and_original_failure(self):
        failure = subprocess.CalledProcessError(128, ["git", "fetch"], stderr="fatal: Could not resolve host: github.com")
        with mock.patch("tools.upstream_runtime.shutil.which", return_value="git"), \
             mock.patch("tools.upstream_runtime.subprocess.run", side_effect=failure), \
             self.assertRaisesRegex(RuntimeError, "Could not resolve host: github.com") as raised:
            _git(Path("runtime"), "fetch", "origin", "revision")
        self.assertIs(raised.exception.__cause__, failure)


if __name__ == "__main__":
    unittest.main()
