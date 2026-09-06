"""Print untranslated runtime terms without changing any files."""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.console_i18n import localize_console_message

if __name__ == "__main__":
    missing = set()
    for root in [ROOT / name for name in ("infer", "lib", "ui", "tools", "rvc_mcp", "configs/schema.py")]:
        for path in [root] if root.is_file() else root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for term in re.findall(r"[\u3400-\u9fff]+", node.value):
                        try:
                            localize_console_message(term, "en_US")
                        except Exception:
                            missing.add(term)
    print(json.dumps(sorted(missing), ensure_ascii=False))
