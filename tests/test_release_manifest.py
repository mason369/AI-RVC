"""Use the actual release CLI to catch tampered/missing/additional package files."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ReleaseManifestTests(unittest.TestCase):
    def test_manifest_roundtrip_and_same_size_tampering(self):
        script = Path("tools/release_manifest.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "runtime.bin"
            payload.write_bytes(b"original")
            for arguments in ([], ["--verify"]):
                result = subprocess.run([sys.executable, str(script), str(root), *arguments], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            payload.write_bytes(b"replaced")
            result = subprocess.run([sys.executable, str(script), str(root), "--verify"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            payload.write_bytes(b"original")
            (root / "unexpected.py").write_text("print('unexpected')", encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), str(root), "--verify"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
