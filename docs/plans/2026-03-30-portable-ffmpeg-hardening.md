# Portable FFmpeg Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make portable builds self-contained with bundled ffmpeg and ensure runtime code prefers the bundled binary instead of requiring a system installation.

**Architecture:** Add a dedicated runtime helper that resolves bundled ffmpeg paths, prepends them to `PATH`, and sets explicit environment variables. Update GitHub build workflows to copy platform ffmpeg binaries into the packaged tree, and add regression tests that lock both the runtime helper and packaging contract.

**Tech Stack:** Python, GitHub Actions, PyInstaller, pytest

---

### Task 1: Add failing tests for portable ffmpeg runtime

**Files:**
- Create: `tests/test_ffmpeg_runtime.py`

**Step 1: Write the failing test**
- Assert a bundled ffmpeg directory is preferred over system lookup.
- Assert runtime setup prepends the bundled directory to `PATH`.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_ffmpeg_runtime.py`

**Step 3: Write minimal implementation**
- Add the helper module and call it from `run.py`.

**Step 4: Run test to verify it passes**
- Run: `pytest -q tests/test_ffmpeg_runtime.py`

**Step 5: Commit**
- Commit helper + test once green.

### Task 2: Remove remaining hard dependency on `ffprobe`

**Files:**
- Modify: `infer/modules/uvr5/modules.py`

**Step 1: Write the failing test**
- Assert the module no longer calls `ffmpeg.probe(..., cmd="ffprobe")`.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_ffmpeg_runtime.py`

**Step 3: Write minimal implementation**
- Replace the probe path with a library-based metadata read that does not shell out to `ffprobe`.

**Step 4: Run tests to verify they pass**
- Run: `pytest -q tests/test_ffmpeg_runtime.py`

**Step 5: Commit**
- Commit metadata-path cleanup.

### Task 3: Bundle ffmpeg in GitHub release builds

**Files:**
- Modify: `.github/workflows/build-executables.yml`
- Modify: `AI-RVC.spec`

**Step 1: Write the failing test**
- Assert the workflow stages copy a bundled ffmpeg directory and package it.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_ffmpeg_runtime.py`

**Step 3: Write minimal implementation**
- Add a workflow step to copy `ffmpeg` into `tools/ffmpeg/bin`.
- Add the ffmpeg directory to packaged data inputs for both workflow and spec.

**Step 4: Run tests to verify they pass**
- Run: `pytest -q tests/test_ffmpeg_runtime.py`

**Step 5: Commit**
- Commit workflow/spec hardening.
