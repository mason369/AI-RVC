# Repository Guidelines

## Project Structure & Module Organization
- `run.py` is the application entrypoint.
- `ui/` contains the Gradio interface (`ui/app.py`).
- `infer/` holds inference and cover pipelines (`pipeline.py`, `cover_pipeline.py`) plus VC/UVR modules.
- `lib/` contains shared runtime utilities (audio, mixer, logger, device).
- `models/` stores model wrappers and synthesis definitions.
- `configs/` stores runtime settings (`configs/config.json` is the main file).
- `tools/` includes utility scripts (for example `download_models.py`).
- `assets/` holds runtime model assets, `outputs/` stores generated audio, and `temp/` is scratch space.
- `_official_rvc/` is upstream reference code; treat it as vendored unless a sync/update is intended.

## Build, Test, and Development Commands
Use PowerShell from repository root:

```powershell
cd C:\AI-RVC
.\venv310\Scripts\Activate.ps1
pip install -r requirements.txt
python tools/download_models.py
python run.py --host 127.0.0.1 --port 7860
```

- `python run.py --skip-check`: skip environment/model checks for faster iteration.
- `python -c "import torch; print(torch.cuda.is_available())"`: quick CUDA sanity check.

## Coding Style & Naming Conventions
- Language: Python, UTF-8, 4-space indentation.
- Naming: `snake_case` for functions/variables/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer small, focused functions; add type hints for new or changed public helpers.
- Keep config keys explicit and backward-compatible; update defaults near the read path.
- Keep user-facing logs/messages concise and consistent with existing bilingual (Chinese/English) style.

## Testing Guidelines
- No dedicated top-level automated suite is currently maintained; rely on targeted smoke tests.
- For inference/UI changes, run one voice conversion and one cover workflow through the UI.
- For config/path changes, test startup with and without `--skip-check`, then verify outputs in `outputs/`.
- If you add automated tests, place them under `tests/` and use `pytest` conventions (`test_*.py`).

## Commit & Pull Request Guidelines
- Current history uses short, direct subjects (Chinese/English mixed), such as `初始化...`, `添加...`, `update`.
- Write imperative, scoped messages (example: `infer: fix CUDA OOM by enabling chunking`).
- PRs should include: goal, affected paths, config/model impact, manual verification steps, and UI screenshots when relevant.
- Do not commit virtual environments, model weights, generated audio, or `.env` secrets.
