"""Immutable official RVC runtimes; never alter the user's legacy source tree."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tools.model_assets import download_pinned_asset, verify_asset

REPOSITORY = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
REVISIONS = {
    "vc": "81eed5e8f68b6bed1789f682fe78cdd324495afc",
    "uvr5": "7ef19867780cf703841ebafb565a4e47d1ea86ff",
}
REQUIRED_FILES = {
    "vc": ("configs/config.py", "infer/vc/modules.py", "infer/vc/pipeline.py",
           "infer/hubert.py", "infer/rmvpe.py"),
    "uvr5": ("configs/config.py", "infer/modules/uvr5/modules.py"),
}
VC_SINGLE_PARAMETERS = (
    "self", "sid", "input_audio_path", "f0_up_key", "f0_method", "file_index",
    "index_rate", "resample_sr", "rms_mix_rate", "protect",
)
HUBERT_REPOSITORY = "lj1995/VoiceConversionWebUI"
HUBERT_REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
HUBERT_FILES = {
    "config.json": "0346950779dfb7f9316fa74ed846e2b8a22a08eedfdc5387b73f327cb1a4a7cf",
    "preprocessor_config.json": "7c1976a680fb7acc757cd36fb08eef878fa36c70b4c9d2d595df9c608bbbbf0e",
    "pytorch_model.bin": "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5",
}


def runtime_root(project_root: Path, capability: str = "vc") -> Path:
    if capability not in REVISIONS:
        raise ValueError(f"未知官方运行能力: {capability}")
    return Path(project_root) / "_official_rvc_runtime" / REVISIONS[capability]


def _git(root: Path, *args: str, timeout: int = 60) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("准备及核验固定版本官方 RVC 源码需要 Git，请安装 Git")
    try:
        result = subprocess.run(
            [executable, "-C", str(root), *args], check=True, timeout=timeout,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Git {args[0]} failed (exit={exc.returncode}): {details}") from exc
    return result.stdout.strip()


def source_problems(root: Path, capability: str = "vc") -> list[str]:
    missing = [name for name in REQUIRED_FILES[capability] if not (root / name).is_file()]
    if missing:
        return missing
    if getattr(sys, "frozen", False):
        problems = _packaged_source_problems(root, capability)
        if problems:
            return problems
    else:
        if not (root / ".git").exists():
            return ["缺少 .git，无法确认官方源码版本"]
        revision = _git(root, "rev-parse", "HEAD")
        if revision != REVISIONS[capability]:
            return [f"源码版本不符: expected={REVISIONS[capability]}, actual={revision}"]
        # A Windows checkout may be shared with WSL. Compare canonical Git text
        # instead of treating CRLF checkout conversion as an upstream source edit.
        changed = _git(root, "-c", "core.autocrlf=true", "diff", "--name-only", "HEAD", "--")
        if changed:
            return [f"固定版本源码含修改，已保留并停止使用: {changed}"]
    if capability == "vc":
        tree = ast.parse((root / "infer/vc/modules.py").read_text(encoding="utf-8"))
        vc = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VC"), None)
        method = next((node for node in vc.body if isinstance(node, ast.FunctionDef) and node.name == "vc_single"), None) if vc else None
        parameters = tuple(arg.arg for arg in method.args.args) if method else ()
        if parameters != VC_SINGLE_PARAMETERS:
            return [f"官方 VC.vc_single 接口不兼容: {parameters}"]
    return []


def _packaged_source_problems(root: Path, capability: str) -> list[str]:
    """Validate the pinned Git blob contents without Git in a portable package."""
    manifest = Path(__file__).with_name("upstream_source_manifest.json")
    if not manifest.is_file():
        return ["便携包缺少官方源码校验清单"]
    record = json.loads(manifest.read_text(encoding="utf-8"))[capability]
    if record["revision"] != REVISIONS[capability]:
        return ["便携包官方源码校验清单版本不符"]
    files = record["files"]
    if not set(REQUIRED_FILES[capability]).issubset(files):
        return ["便携包官方源码校验清单不完整"]
    problems = []
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            problems.append(relative)
            continue
        data = path.read_bytes()
        if expected["text"]:
            data = data.replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            problems.append(f"官方源码 SHA-256 不匹配: {relative}")
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if ".git" not in relative.parts and relative.as_posix() not in files:
            problems.append(f"官方源码包含额外代码: {relative}")
    return problems


def ensure_source(project_root: Path, capability: str = "vc", timeout: int = 900) -> Path:
    root = runtime_root(project_root, capability)
    if not root.exists():
        if getattr(sys, "frozen", False):
            raise RuntimeError(f"便携包缺少固定版本官方源码: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = root.with_name(root.name + ".fetching")
        if staging.exists():
            raise RuntimeError(f"发现未完成的源码准备目录，已保留，请检查后再继续: {staging}")
        staging.mkdir()
        try:
            _git(staging, "init")
            _git(staging, "remote", "add", "origin", REPOSITORY)
            _git(staging, "fetch", "--depth", "1", "origin", REVISIONS[capability], timeout=timeout)
            _git(staging, "checkout", "--detach", "FETCH_HEAD")
            problems = source_problems(staging, capability)
            if problems:
                raise RuntimeError("；".join(problems))
            staging.rename(root)
        except Exception as exc:
            raise RuntimeError(
                f"准备固定版本官方 {capability} 源码失败: {exc}；下载现场保留在 {staging}"
            ) from exc
    problems = source_problems(root, capability)
    if problems:
        raise RuntimeError(f"官方 {capability} 源码不可用: {root}；" + "；".join(problems))
    return root


def hubert_problems(project_root: Path) -> list[str]:
    problems = []
    for name, digest in HUBERT_FILES.items():
        path = Path(project_root) / "assets/hubert_base" / name
        if not path.is_file():
            problems.append(f"assets/hubert_base/{name}")
        else:
            try:
                verify_asset(path, digest)
            except RuntimeError as exc:
                problems.append(str(exc))
    return problems


def ensure_transformers_hubert(project_root: Path) -> Path:
    assets = Path(project_root) / "assets"
    for name, digest in HUBERT_FILES.items():
        download_pinned_asset(
            HUBERT_REPOSITORY, f"hubert_base/{name}", assets, HUBERT_REVISION, digest,
        )
    return assets / "hubert_base"
