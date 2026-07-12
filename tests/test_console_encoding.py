import os
import subprocess
import sys
import unittest


class ConsoleEncodingTests(unittest.TestCase):
    def test_logger_writes_utf8_when_parent_environment_does_not_override_encoding(self):
        env = os.environ.copy()
        env.pop("PYTHONIOENCODING", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from lib.logger import log; log.info('中文 English')",
            ],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            check=True,
        )

        self.assertIn("中文 English".encode("utf-8"), result.stdout)

    def test_installer_initialization_writes_utf8(self):
        env = os.environ.copy()
        env.pop("PYTHONIOENCODING", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import install; print('安装检查')",
            ],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "安装检查".encode("utf-8"))

    def test_console_encoding_does_not_replace_output_errors(self):
        env = os.environ.copy()
        env.pop("PYTHONIOENCODING", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import install, sys; print(sys.stdout.encoding, sys.stdout.errors)",
            ],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip().lower(), "utf-8 strict")


if __name__ == "__main__":
    unittest.main()
