# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata
from pathlib import Path
import sys

sys.path.insert(0, str(Path(SPEC).resolve().parent))
from tools.package_runtime import validate_upstream_sources
from tools.package_runtime import validate_cuda_runtime

import torch

validate_upstream_sources(Path(SPEC).resolve().parent)

# Retired assets may remain in an existing installation; never redistribute them.
separator_datas = [
    (str(path), str(path.parent))
    for path in Path('assets/separator_models').rglob('*')
    if path.is_file() and not any('polarformer' in part.lower() for part in path.parts)
]

bundle_datas, bundle_binaries, bundle_imports = [], [], []
for package in ('torch', 'torchaudio', 'gradio', 'gradio_client', 'torchfcpe', 'transformers', 'rvc_mcp', 'mcp'):
    package_datas, package_binaries, package_imports = collect_all(package)
    bundle_datas.extend(package_datas)
    bundle_binaries.extend(package_binaries)
    bundle_imports.extend(package_imports)

if sys.platform.startswith('linux') and torch.version.cuda:
    bundle_imports.append('nvidia')

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=bundle_binaries,
    datas=[
        ('ui/*.py', 'ui'),
        ('ui/multitrack/dist/player.html', 'ui/multitrack/dist'),
        ('infer', 'infer'),
        ('lib', 'lib'),
        ('models', 'models'),
        ('tools', 'tools'),
        ('tools/ffmpeg', 'tools/ffmpeg'),
        ('rvc_mcp', 'rvc_mcp'),
        ('i18n', 'i18n'),
        ('configs', 'configs'),
        ('_official_rvc_runtime', '_official_rvc_runtime'),
        ('assets/hubert', 'assets/hubert'),
        ('assets/hubert_base', 'assets/hubert_base'),
        ('assets/rmvpe', 'assets/rmvpe'),
        ('assets/uvr5_weights', 'assets/uvr5_weights'),
        ('assets/pretrained_v2', 'assets/pretrained_v2'),
    ] + separator_datas + bundle_datas + collect_data_files('torchfcpe') + collect_data_files('transformers')
      + collect_data_files('audio_separator') + collect_data_files('safehttpx')
      + collect_data_files('groovy')
      + copy_metadata('audio-separator'),
    hiddenimports=[
        'torch',
        'torchaudio',
        'gradio',
        'librosa',
        'soundfile',
        'scipy',
        'numpy',
        'onnxruntime',
        'einops',
        'yaml',
        'fairseq',
        'transformers',
        'transformers.models.hubert.modeling_hubert',
        'audio_separator',
        'demucs',
        'pedalboard',
        'parselmouth',
        'pyworld',
        'torchcrepe',
        'torchfcpe',
        'faiss',
        'huggingface_hub',
        'gdown',
        'ffmpeg',
    ] + bundle_imports + collect_submodules('rvc_mcp') + collect_submodules('mcp')
      + collect_submodules('audio_separator.separator')
      + collect_submodules('audio_separator.separator.uvr_lib_v5.roformer'),
    hookspath=[str(Path(SPEC).resolve().parent / 'tools' / 'pyinstaller_hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI-RVC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI-RVC',
)

validate_upstream_sources(Path(DISTPATH) / 'AI-RVC' / '_internal')
if torch.version.cuda:
    validate_cuda_runtime(Path(DISTPATH) / 'AI-RVC' / '_internal', torch.version.cuda)
