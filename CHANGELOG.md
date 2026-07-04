# Changelog

## v1.3.0 - 2026-07-04

AI-RVC v1.3.0 focuses on the default cover quality route, model compatibility, and release hygiene.

### Highlights

- Changed the default cover route to strict RMVPE with `f0_hybrid_mode=off`; hybrid/fallback F0 paths now fail loudly instead of silently changing behavior.
- Enabled the official-compatible RVC inference path by default while keeping the project cover preprocessing, cleanup, source constraint, and mixing chain.
- Added RVC checkpoint version detection from weight shape, so v1/v2 models with missing or wrong `version` metadata are handled explicitly.
- Validates FAISS index dimensions against the loaded RVC model before conversion.
- Added custom character model import from `.pth`, `.pth + .index`, or a single-model `.zip`.
- Expanded the downloadable character registry to 181 entries and improved series/category grouping.
- Cleaned generated cover download filenames by removing Gradio temp prefixes and adding per-output download buttons.

### Quality Route

- RoFormer De-Reverb is now the strict VC preprocessing path for default covers.
- Source cleanup and transition smoothing now preserve more active vocal body while still suppressing echo tails, quiet artifacts, and transition spikes.
- Added a default-quality audit tool that blocks hidden parameter overrides and requires listening review for final quality verdicts.
- Removed the UI singing-repair toggle from the default flow because it depended on F0 fallback behavior.

### Installation And Runtime

- `run.py` now verifies required base models and prepares the vendored official RVC source tree when the default route needs it.
- `tools/download_models.py` can prepare `_official_rvc/` and reports incomplete trees as hard errors.
- Dependency checks now enforce exact versions for Gradio, fairseq, and audio-separator where the project depends on pinned behavior.
- Hugging Face Hub is constrained below 1.0 to stay aligned with the Space runtime.

### UI And Documentation

- Cover controls now validate values instead of clamping or silently falling back.
- Mix presets update actual mix sliders and still allow manual adjustment.
- README and Hugging Face README now document the current model positioning, SOTA boundaries, strict defaults, and official RVC source preparation.
- Local agent instruction files and ignored docs artifacts are removed from Git tracking.

### Tests

- Added coverage for audio cleanup guards, RVC version detection, official adapter export, index dimension validation, custom model import, clean output filenames, UI download buttons, default quality audit policy, install requirement checks, and strict cover configuration.
- Verified locally with `python -m unittest discover -s tests` on July 4, 2026: 116 tests passed.
