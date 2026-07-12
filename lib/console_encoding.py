# -*- coding: utf-8 -*-
"""Console encoding setup shared by application and installer entry points."""

import sys


def configure_console_encoding() -> None:
    """Write project console output as UTF-8 on terminals and redirected streams."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="strict")
