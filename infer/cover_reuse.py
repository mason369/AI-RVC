"""Session-local references to completed audio stages. Audio stays in its original run."""
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tools.model_assets import file_sha256


def asset_version(path):
    if path is None:
        return None
    path = Path(path).resolve()
    try:
        return str(path), file_sha256(path)
    except FileNotFoundError:
        return str(path), None


def separator_versions(engine, model):
    """Fingerprint installed inputs, including configs and the auto-selected UVR5 weight."""
    from infer.separator import (_CUSTOM_AUDIO_SEPARATOR_MODELS, KARAOKE_DEFAULT_MODEL,
                                 KARAOKE_SOTA_MODELS, LEAP_INSTRUMENTAL_MODEL,
                                 LEAP_XE_VOCALS_MODEL, ROFORMER_DEFAULT_MODEL, _separator_model_dir)

    assets = Path(__file__).resolve().parents[1] / "assets"
    if engine == "uvr5":
        directory = assets / "uvr5_weights"
    elif engine == "demucs":
        import torch
        directory = Path(torch.hub.get_dir()) / "checkpoints"
    else:
        directory = assets / "separator_models"
        if isinstance(model, (list, tuple)):
            models = model
        else:
            models = {ROFORMER_DEFAULT_MODEL: (LEAP_XE_VOCALS_MODEL, LEAP_INSTRUMENTAL_MODEL),
                      KARAOKE_DEFAULT_MODEL: KARAOKE_SOTA_MODELS}.get(model, (model,))
        if all(name in _CUSTOM_AUDIO_SEPARATOR_MODELS for name in models):
            paths = []
            for name in models:
                spec = _CUSTOM_AUDIO_SEPARATOR_MODELS[name]
                root = directory / spec.get("local_subdir", "")
                paths.extend(root / spec[key] for key in ("model_filename", "config_filename"))
                window_config = _separator_model_dir() / spec.get("local_subdir", "") / spec["config_filename"]
                if window_config.resolve() != paths[-1].resolve():
                    paths.append(window_config)
            return tuple(map(asset_version, paths))
    return tuple(asset_version(p) for p in sorted(directory.rglob("*"))
                 if p.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx", ".th", ".yaml", ".json"})


@dataclass
class Stage:
    key: tuple
    directory: Path
    files: tuple
    preprocess_mode: str | None


class CoverReuse:
    def __init__(self, session_state, source, directory):
        self.directory = Path(directory).resolve()
        self.stages = None
        self.hits = []
        if session_state is not None:
            song = (Path(source).suffix.lower(), file_sha256(Path(source)))
            self.stages = session_state.setdefault(song, {})

    def origin(self, name):
        return self.stages[name].directory if self.stages is not None else None

    def restore(self, name, key):
        stage = self.stages.get(name) if self.stages is not None else None
        if stage is None or stage.key != key():
            return None
        # Verify the actual copies. A missing or edited stem invalidates the entire stage.
        paths = []
        with tempfile.TemporaryDirectory(dir=self.directory) as staging:
            for index, (relative, digest) in enumerate(stage.files):
                copy = Path(staging) / str(index)
                try:
                    shutil.copyfile(stage.directory / relative, copy)
                except FileNotFoundError:
                    return None
                if file_sha256(copy) != digest:
                    return None
                paths.append(str(self.directory / relative))
            for index, target in enumerate(map(Path, paths)):
                target.parent.mkdir(parents=True, exist_ok=True)
                (Path(staging) / str(index)).replace(target)
        self.hits.append(name)
        return paths

    def save(self, name, key, paths, preprocess_mode=None):
        if self.stages is not None:
            # A separator may return the physical volume path for a symlinked run directory.
            resolved = (Path(path).resolve(strict=True) for path in paths)
            files = tuple((path.relative_to(self.directory), file_sha256(path)) for path in resolved)
            self.stages[name] = Stage(key(), self.directory, files, preprocess_mode)
