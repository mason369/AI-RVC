"""角色资产校验：文件存在不等于有效模型，不以文件名推断 RVC 架构。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def model_files(directory: Path) -> tuple[Path, Path | None]:
    directory = Path(directory)
    weights = sorted(directory.rglob('*.pth')) if directory.is_dir() else []
    if not weights:
        raise FileNotFoundError(f'角色目录没有 .pth 权重：{directory}')
    if len(weights) != 1:
        raise ValueError(f'角色目录包含多个 .pth，必须明确拆分，不能任选第一个：{directory}')
    indices = sorted(directory.rglob('*.index'))
    matching = [path for path in indices if path.stem.casefold() == weights[0].stem.casefold()]
    if len(matching) == 1 and len(indices) == 1:
        index = matching[0]
    elif len(indices) == 1:
        index = indices[0]
    elif not indices:
        index = None
    else:
        raise ValueError(f'角色目录包含多个索引，不能猜测对应关系：{directory}')
    for path in [weights[0]] + ([index] if index else []):
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f'角色资产路径越界：{path}')
    return weights[0], index


def _stamp(path: Path | None):
    if path is None:
        return None
    from tools.model_assets import file_sha256
    return str(path.resolve()), file_sha256(path)


@lru_cache(maxsize=64)
def _inspect_cached(weight_stamp: tuple, index_stamp: tuple | None) -> dict:
    import torch
    from infer.contracts import inspect_checkpoint, validate_state_dict, read_index
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid_nono,
    )

    path = Path(weight_stamp[0])
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    contract = inspect_checkpoint(checkpoint, str(path))
    classes = {('v1', True): SynthesizerTrnMs256NSFsid,
               ('v2', True): SynthesizerTrnMs768NSFsid,
               ('v1', False): SynthesizerTrnMs256NSFsid_nono,
               ('v2', False): SynthesizerTrnMs768NSFsid_nono}
    # Shape validation does not need a second allocated copy of the network.
    with torch.device('meta'):
        model = classes[(contract.version, contract.uses_f0)](*contract.config, is_half=False)
        del model.enc_q
    validate_state_dict(model, checkpoint['weight'])
    result = {'version': contract.version, 'feature_dim': contract.feature_dim,
              'sample_rate': contract.sample_rate, 'speakers': contract.speaker_count,
              'uses_f0': contract.uses_f0,
              'dtypes': sorted({str(v.dtype) for v in checkpoint['weight'].values()}),
              'model_path': str(path), 'index_path': index_stamp[0] if index_stamp else None,
              'index_vectors': 0, 'runtime_verified': False}
    if index_stamp:
        index, _ = read_index(index_stamp[0], contract.feature_dim, min_vectors=8)
        result['index_vectors'] = int(index.ntotal)
    return result


def inspect_model_assets(weight_path: Path, index_path: Path | None = None, *, require_index: bool = False) -> dict:
    if require_index and index_path is None:
        raise FileNotFoundError('该角色下载配置包含索引，但本地缺少 .index 文件')
    return dict(_inspect_cached(_stamp(Path(weight_path)), _stamp(Path(index_path)) if index_path else None))


def inspect_character_directory(directory: Path, *, require_index: bool = False) -> dict:
    weight, index = model_files(directory)
    return inspect_model_assets(weight, index, require_index=require_index)
