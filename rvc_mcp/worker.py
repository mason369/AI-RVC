"""Isolated numerical runtime for one MCP conversion request."""
import json
import sys
from contextlib import redirect_stdout


def main():
    arguments = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        from rvc_mcp.tools import convert_voice
        result = convert_voice(**arguments)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
