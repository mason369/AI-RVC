# -*- coding: utf-8 -*-
"""Strict console localization for project-owned messages."""

from __future__ import annotations

import builtins
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, TextIO


SUPPORTED_CONSOLE_LANGUAGES = frozenset({"zh_CN", "en_US"})
LANGUAGE_ENV_VAR = "AI_RVC_LANGUAGE"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.json"
_ENGLISH_CATALOG_PATH = _PROJECT_ROOT / "i18n" / "console_en_US.json"
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_ENGLISH_PUNCTUATION = str.maketrans(
    {
        "，": ", ",
        "。": ".",
        "：": ": ",
        "；": "; ",
        "（": " (",
        "）": ")",
        "【": "[",
        "】": "]",
        "！": "!",
        "、": ", ",
    }
)
_language_override: str | None = None


class ConsoleLocalizationError(RuntimeError):
    """Raised when a project console message cannot be localized exactly."""


def _validate_language(language: str) -> str:
    normalized = str(language).strip()
    if normalized not in SUPPORTED_CONSOLE_LANGUAGES:
        raise ValueError(
            f"Unsupported console language {normalized!r}; "
            "expected zh_CN or en_US"
        )
    return normalized


def set_console_language(language: str | None) -> None:
    """Set a process-local language override, or clear it with ``None``."""
    global _language_override
    _language_override = None if language is None else _validate_language(language)


def get_console_language() -> str:
    """Resolve the console locale from override, environment, then config."""
    if _language_override is not None:
        return _language_override

    environment_language = os.environ.get(LANGUAGE_ENV_VAR)
    if environment_language is not None:
        return _validate_language(environment_language)

    try:
        config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsoleLocalizationError(
            f"Cannot read console language from {_CONFIG_PATH}: {exc}"
        ) from exc

    return _validate_language(config.get("language", ""))


@lru_cache(maxsize=1)
def _load_english_catalog() -> tuple[tuple[str, str], ...]:
    try:
        raw_catalog = json.loads(_ENGLISH_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsoleLocalizationError(
            f"Cannot load English console catalog {_ENGLISH_CATALOG_PATH}: {exc}"
        ) from exc

    if not isinstance(raw_catalog, dict) or not raw_catalog:
        raise ConsoleLocalizationError("English console catalog must be a non-empty object")

    catalog: list[tuple[str, str]] = []
    for source, translated in raw_catalog.items():
        if not isinstance(source, str) or not source or not _HAN_RE.search(source):
            raise ConsoleLocalizationError(
                f"Invalid English console catalog source key: {source!r}"
            )
        if not isinstance(translated, str) or not translated.strip():
            raise ConsoleLocalizationError(
                f"Missing English console translation for {source!r}"
            )
        if _HAN_RE.search(translated):
            raise ConsoleLocalizationError(
                f"English console translation still contains Chinese: {source!r}"
            )
        catalog.append((source, translated))

    return tuple(sorted(catalog, key=lambda item: len(item[0]), reverse=True))


def localize_console_message(message: Any, language: str | None = None) -> str:
    """Localize one message and fail if English output remains untranslated."""
    text = str(message)
    selected_language = get_console_language() if language is None else _validate_language(language)
    if selected_language == "zh_CN" or not _HAN_RE.search(text):
        return text

    catalog = _load_english_catalog()
    translated: list[str] = []
    position = 0
    while position < len(text):
        if not _HAN_RE.match(text[position]):
            translated.append(text[position])
            position += 1
            continue

        match = next(
            (
                (source, replacement)
                for source, replacement in catalog
                if text.startswith(source, position)
            ),
            None,
        )
        if match is None:
            remaining = "".join(
                dict.fromkeys(_HAN_RE.findall(text[position:]))
            )
            raise ConsoleLocalizationError(
                "English console catalog has no complete translation for message; "
                f"remaining characters={remaining!r}, source={text!r}"
            )

        source, replacement = match
        if (
            translated
            and translated[-1]
            and translated[-1][-1].isascii()
            and translated[-1][-1].isalnum()
            and replacement
            and replacement[0].isascii()
            and replacement[0].isalnum()
        ):
            translated.append(" ")
        translated.append(replacement)
        position += len(source)

    result = "".join(translated).translate(_ENGLISH_PUNCTUATION)
    if _HAN_RE.search(result):
        raise ConsoleLocalizationError(
            f"English console localization produced Chinese output: {result!r}"
        )
    return result


def console_print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Localized replacement for project-owned direct ``print`` calls."""
    if not isinstance(sep, str) or not isinstance(end, str):
        raise TypeError("sep and end must be strings")
    message = sep.join(str(value) for value in values)
    builtins.print(
        localize_console_message(message),
        end=end,
        file=file,
        flush=flush,
    )
