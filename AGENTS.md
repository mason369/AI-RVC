# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

RVC v2 voice conversion & AI cover system. Users upload a song, the pipeline separates vocals from accompaniment (Mel-Band Roformer), converts the vocal timbre via RVC v2 (HuBERT + RMVPE + FAISS), then mixes the result back with the accompaniment.

**Platform Support**: Windows / Linux / WSL2 / Google Colab

**Key Features**:
- AI song covers with automatic vocal separation and mixing
- 117 downloadable character models
- 4 mixing presets (universal, vocal-focused, accompaniment-focused, live)
- Karaoke mode (lead/backing vocal separation)
- 4 VC preprocessing modes (auto, direct, uvr_deecho, legacy)
- Dual VC pipeline (current implementation vs official RVC)
- Multi-backend GPU support (CUDA, ROCm, XPU, DirectML, MPS)

## Commands

```powershell
# Activate venv (Windows)
.\venv310\Scripts\Activate.ps1

# Activate venv (Linux/WSL2)
source venv310/bin/activate

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

# Download character models
python -c "from tools.character_models import download_character_model; download_character_model('rin')"

# Quick CUDA check
python -c "import torch; print(torch.cuda.is_available())"

# Colab
# Open AI_RVC_Colab.ipynb in Google Colab, set runtime to GPU (T4), run cells sequentially
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
- Gradio 3.50.2, single-file ~2000 lines
- i18n via `i18n/zh_CN.json`, accessed through `t(key, section)` helper
- Three main tabs: song cover (full pipeline), model management, settings
- Cover tab features:
  - Character model download/management with series filtering and keyword search
  - 4 mixing presets (universal, vocal-focused, accompaniment-focused, live)
  - Karaoke separation (lead/backing vocals)
  - 4 VC preprocessing modes (auto, direct, uvr_deecho, legacy)
  - Source constraint control (auto/off/on)
  - Dual VC pipeline mode (current/official)
  - Singing repair (official mode only)
  - Real-time VC route status display
- Model management tab:
  - Base model download (HuBERT, RMVPE)
  - Mature DeEcho model download
  - Model list table with refresh
- Settings tab:
  - Device info display
  - Backend selection (CUDA/ROCm/XPU/DirectML/MPS/CPU)
  - Config save

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
- `tools/download_models.py` — base model (HuBERT/RMVPE) + mature DeEcho downloader
- `lib/mixer.py` — audio mixing with volume/reverb
- `ui/app.py` — entire Gradio UI (~2000 lines)
- `mcp/server.py` + `mcp/tools.py` — MCP server integration for Codex
- `AI_RVC_Colab.ipynb` — Google Colab notebook with full feature parity
- `install.py` — cross-platform installation script (Windows/Linux)

## Things to Watch

- `fairseq` is pinned to `0.12.2` — HuBERT loading breaks on other versions
- `audio-separator` must be installed with `[gpu]` extra for CUDA support
- Roformer model auto-downloads on first use to `assets/separator_models/`
- Gradio is pinned to `3.50.2`; the UI code uses v3 API patterns (not v4)
- Model weights (.pt, .pth) and audio files are gitignored — never commit them
- Path handling uses `pathlib.Path` for cross-platform compatibility (Windows/Linux)
- Virtual environment activation differs by platform: `Scripts/Activate.ps1` (Windows) vs `bin/activate` (Linux)
- `install.py` has hardcoded Windows Python paths in `PYTHON310_CANDIDATES` but falls back to `py -3.10` launcher
- Platform detection uses `os.name == "nt"` for Windows-specific logic (venv paths, etc.)
- All core functionality is platform-agnostic; audio libraries work better on Linux
- Colab notebook (`AI_RVC_Colab.ipynb`) provides full feature parity with Web UI
