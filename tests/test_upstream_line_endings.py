"""Verify immutable upstream source using real cross-platform Git checkouts."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import upstream_runtime as runtime


class UpstreamLineEndingTests(unittest.TestCase):
    def test_crlf_checkout_is_accepted_but_content_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True,
                    text=True, encoding="utf-8",
                ).stdout.strip()

            git("init")
            git("config", "core.autocrlf", "false")
            git("config", "commit.gpgsign", "false")
            for name in runtime.REQUIRED_FILES["uvr5"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"# original source\nvalue = 1\n")
            git("add", ".")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-m", "Original source")
            revision = git("rev-parse", "HEAD")
            for name in runtime.REQUIRED_FILES["uvr5"]:
                path = root / name
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

            self.assertTrue(git("diff", "--name-only", "HEAD", "--"))
            with patch.dict(runtime.REVISIONS, {"uvr5": revision}):
                self.assertEqual(runtime.source_problems(root, "uvr5"), [])
                path.write_bytes(path.read_bytes().replace(b"value = 1", b"value = 2"))
                self.assertIn("含修改", " ".join(runtime.source_problems(root, "uvr5")))


if __name__ == "__main__":
    unittest.main()
