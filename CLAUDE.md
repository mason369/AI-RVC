# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVC v2 voice conversion & AI cover system. Users upload a song, the pipeline separates vocals from accompaniment (Mel-Band Roformer), converts the vocal timbre via RVC v2 (HuBERT + RMVPE + FAISS), then mixes the result back with the accompaniment.

## Commands

```powershell
# Activate venv (Windows)
.\venv310\Scripts\Activate.ps1

# Install dependencies
python install.py              # full install + launch
python install.py --check      # check only
python install.py --cpu        # CPU variant

# Run
python run.py                          # default: http://127.0.0.1:7860
python run.py --skip-check             # skip env/model validation
python run.py --host 0.0.0.0 --port 8080 --share

# Download base models (HuBERT, RMVPE)
python tools/download_models.py

# Quick CUDA check
python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture

**Entry:** `run.py` → env check → model check → `ui/app.py:launch()`

**Pipeline flow** (`infer/cover_pipeline.py:CoverPipeline.process`):
1. Vocal separation (`infer/separator.py`) — Roformer (default), Demucs, or UVR5
2. RVC voice conversion (`infer/pipeline.py`) — HuBERT features → RMVPE F0 → RVC v2 inference with FAISS retrieval
3. Mixing (`lib/mixer.py`) — volume adjust + reverb via pedalboard

**Character model system** (`tools/character_models.py`):
- 117 downloadable character models from HuggingFace (`trioskosmos/rvc_models`)
- Stored in `assets/weights/characters/`
- Version notes (epochs, sample rate) extracted from .pth metadata and cached in `_version_notes.json`
- Display name assembly: `_get_display_name()` appends `(500 epochs·40k)` style training info

**UI** (`ui/app.py`):
- Gradio 3.50.2, single-file ~1500 lines
- i18n via `i18n/zh_CN.json`, accessed through `t(key, section)` helper
- Two main tabs: voice conversion (direct) and song cover (full pipeline)

**Config:** `configs/config.json` — device, F0 method, index rate, cover separator settings, path mappings

## Key Conventions

- Python 3.10, UTF-8, 4-space indent
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- User-facing text is bilingual Chinese/English
- Commit messages: short imperative subjects, Chinese/English mixed (e.g. `infer: fix CUDA OOM`)
- No automated test suite; verify changes by running one voice conversion + one cover through the UI
- `_official_rvc/` is vendored upstream reference — don't modify unless syncing

## Important Paths

- `configs/config.json` — all runtime settings
- `infer/cover_pipeline.py` — orchestrates the full cover workflow
- `infer/pipeline.py` — RVC v2 inference core
- `infer/separator.py` — Roformer/Demucs vocal separation wrappers
- `tools/character_models.py` — character model registry (117 entries) + download logic
- `tools/download_models.py` — base model (HuBERT/RMVPE) downloader
- `lib/mixer.py` — audio mixing with volume/reverb
- `ui/app.py` — entire Gradio UI
- `mcp/server.py` + `mcp/tools.py` — MCP server integration for Claude Code

## Things to Watch

- `fairseq` is pinned to `0.12.2` — HuBERT loading breaks on other versions
- `audio-separator` must be installed with `[gpu]` extra for CUDA support
- Roformer model auto-downloads on first use to `assets/separator_models/`
- Gradio is pinned to `3.50.2`; the UI code uses v3 API patterns (not v4)
- Model weights (.pt, .pth) and audio files are gitignored — never commit them
