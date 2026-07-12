import ast
import io
import os
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HAN_RE = re.compile(r"[\u3400-\u9fff]")
OUTPUT_CALLS = {
    "print",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "step",
    "detail",
    "progress",
    "model",
    "audio",
    "config",
    "header",
    "format",
}
RELEASE_ROOTS = (
    ROOT / "run.py",
    ROOT / "install.py",
    ROOT / "check_deecho_config.py",
    ROOT / "infer",
    ROOT / "lib",
    ROOT / "tools",
    ROOT / "ui",
    ROOT / "mcp",
)


def _release_python_files():
    for root in RELEASE_ROOTS:
        candidates = (root,) if root.is_file() else root.rglob("*.py")
        for path in candidates:
            if "_official_rvc" not in path.parts and "__pycache__" not in path.parts:
                yield path


class ConsoleI18nTests(unittest.TestCase):
    def tearDown(self):
        from lib.console_i18n import set_console_language

        set_console_language(None)

    def test_language_resolution_uses_explicit_environment(self):
        from lib.console_i18n import get_console_language

        with patch.dict(os.environ, {"AI_RVC_LANGUAGE": "en_US"}, clear=False):
            self.assertEqual(get_console_language(), "en_US")

    def test_invalid_environment_language_fails(self):
        from lib.console_i18n import get_console_language

        with patch.dict(os.environ, {"AI_RVC_LANGUAGE": "fr_FR"}, clear=False):
            with self.assertRaisesRegex(ValueError, "Unsupported console language"):
                get_console_language()

    def test_english_translation_is_offline_and_contains_no_chinese(self):
        from lib.console_i18n import localize_console_message

        translated = localize_console_message(
            "开始翻唱处理: demo.wav；模型未找到",
            "en_US",
        )
        self.assertIn("demo.wav", translated)
        self.assertIsNone(HAN_RE.search(translated))

    def test_missing_english_translation_fails_explicitly(self):
        from lib.console_i18n import ConsoleLocalizationError, localize_console_message

        with self.assertRaises(ConsoleLocalizationError):
            localize_console_message("龘", "en_US")

    def test_console_print_switches_languages(self):
        from lib.console_i18n import console_print, set_console_language

        set_console_language("en_US")
        output = io.StringIO()
        with redirect_stdout(output):
            console_print("设备信息")
        self.assertIsNone(HAN_RE.search(output.getvalue()))

        set_console_language("zh_CN")
        output = io.StringIO()
        with redirect_stdout(output):
            console_print("设备信息")
        self.assertIn("设备信息", output.getvalue())

    def test_logger_prefixes_are_localized(self):
        from lib.console_i18n import set_console_language
        from lib.logger import log

        set_console_language("en_US")
        output = io.StringIO()
        with redirect_stdout(output):
            log.config("RVC模型: demo.pth")
            log.audio("伴奏文件: demo.wav")
            log.model("模型已加载")
        self.assertIsNone(HAN_RE.search(output.getvalue()))

    def test_all_release_console_literals_have_english_catalog_coverage(self):
        from lib.console_i18n import localize_console_message

        missing = []
        for path in _release_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                selected = []
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    elif isinstance(node.func, ast.Name):
                        name = node.func.id
                    else:
                        name = ""
                    if name in OUTPUT_CALLS:
                        selected = node.args
                elif isinstance(node, ast.Raise) and node.exc is not None:
                    selected = [node.exc]

                for argument in selected:
                    expression = ast.unparse(argument)
                    for source in re.findall(r"[\u3400-\u9fff]+", expression):
                        try:
                            translated = localize_console_message(source, "en_US")
                        except Exception as exc:
                            missing.append(f"{path.relative_to(ROOT)}:{node.lineno}: {exc}")
                        else:
                            if HAN_RE.search(translated):
                                missing.append(
                                    f"{path.relative_to(ROOT)}:{node.lineno}: {source!r}"
                                )
        self.assertEqual(missing, [])

    def test_all_release_string_constants_are_safe_for_dynamic_logging(self):
        from lib.console_i18n import localize_console_message

        missing = []
        for path in _release_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and HAN_RE.search(node.value)
                ):
                    continue
                for source in re.findall(r"[\u3400-\u9fff]+", node.value):
                    try:
                        translated = localize_console_message(source, "en_US")
                    except Exception as exc:
                        missing.append(f"{path.relative_to(ROOT)}:{node.lineno}: {exc}")
                    else:
                        if HAN_RE.search(translated):
                            missing.append(
                                f"{path.relative_to(ROOT)}:{node.lineno}: {source!r}"
                            )
        self.assertEqual(missing, [])

    def test_direct_chinese_prints_use_the_localized_print_boundary(self):
        missing = []
        for path in _release_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            aliases_print = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "lib.console_i18n"
                and any(
                    alias.name == "console_print" and alias.asname == "print"
                    for alias in node.names
                )
                for node in tree.body
            )
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    continue
                if HAN_RE.search(ast.unparse(node)) and not aliases_print:
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(missing, [])

    def test_ui_localizes_pipeline_progress_and_error_details(self):
        source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
        self.assertIn("desc=localize_console_message", source)
        self.assertIn("localized_error = localize_console_message", source)


if __name__ == "__main__":
    unittest.main()
