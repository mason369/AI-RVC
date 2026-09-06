"""Checked floating-output and full-vector retrieval adapters for pinned RVC.

The output adapter keeps network, normalization, and feature dimensions unchanged.
Its encoding changes use unit gain instead of 32768 and float32 instead of int16.
When retrieval is enabled, use the shared exact FP32-vector search, including
zero-distance handling and explicit failures instead of swallowed index errors.
No source checkout or model weights are modified. A changed API fails closed.
The separate F0 adapter restores native unvoiced zeros so protection is effective.
"""
from __future__ import annotations

import inspect
import textwrap
from types import MethodType
from types import SimpleNamespace


def preserve_unvoiced_f0(pipeline) -> None:
    """Keep extractor voicing decisions, as in the standard RVC protection path."""
    original = type(pipeline).get_f0
    source = textwrap.dedent(inspect.getsource(original))
    interpolation = '''    try:
        uv = f0 == 0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    except Exception:
        traceback.print_exc()'''
    if source.count(interpolation) != 1:
        raise RuntimeError("官方 RVC F0 接口发生变化；已停止转换")
    validation = '''    if not np.isfinite(f0).all() or np.any(f0 < 0):
        raise ValueError("F0 输出必须是有限的非负频率")'''
    source = source.replace(interpolation, validation, 1)
    namespace = dict(original.__globals__)
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    pipeline.get_f0 = MethodType(namespace[original.__name__], pipeline)


def preserve_float_vc_output(pipeline, *, index_loader=None, index_retriever=None) -> None:
    original = type(pipeline).pipeline
    source = textwrap.dedent(inspect.getsource(original))
    changes = {
        "max_int16 = 32768": "max_int16 = 1.0",
        "audio_opt = (audio_opt * max_int16).astype(np.int16)":
            "audio_opt = (audio_opt * max_int16).astype(np.float32)",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise RuntimeError("官方 RVC 音频编码接口发生变化，无法保证浮点输出；已停止转换")
        source = source.replace(before, after, 1)
    namespace = dict(original.__globals__)
    if index_loader is not None:
        before = '''    if (
        file_index != ""
        and os.path.exists(file_index)
        and index_rate != 0
    ):
        try:
            index = faiss.read_index(file_index)
            index_vectors = index.reconstruct_n(0, index.ntotal)
        except:
            traceback.print_exc()
            index = index_vectors = None'''
        if source.count(before) != 1 or index_retriever is None:
            raise RuntimeError("官方 RVC 索引接口发生变化，无法保证完整精度检索；已停止转换")
        source = source.replace(before, '    if index_rate != 0:\n        index, index_vectors = _rvc_load_index(file_index)', 1)
        namespace['_rvc_load_index'] = index_loader
        vc_original = type(pipeline).vc
        vc_source = textwrap.dedent(inspect.getsource(vc_original))
        before_retrieval = '''        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)'''
        if vc_source.count(before_retrieval) != 1:
            raise RuntimeError("官方 RVC 特征检索接口发生变化；已停止转换")
        vc_source = vc_source.replace(before_retrieval, '        npy = _rvc_retrieve(index, index_vectors, npy, 8)', 1)
        vc_namespace = dict(vc_original.__globals__)
        vc_namespace['_rvc_retrieve'] = index_retriever
        exec(compile(vc_source, inspect.getfile(vc_original), "exec"), vc_namespace)
        pipeline.vc = MethodType(vc_namespace[vc_original.__name__], pipeline)
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    pipeline.pipeline = MethodType(namespace[original.__name__], pipeline)


def preserve_float_uvr_output(vr_module) -> None:
    """Replace only the checked integer encoders in pinned UVR5 VR classes."""
    import numpy as np

    writer = vr_module.sf.write

    def write_float_wav(path, data, sample_rate, **kwargs):
        if str(path).lower().endswith(".wav"):
            kwargs["subtype"] = "FLOAT"
        return writer(path, data, sample_rate, **kwargs)

    for class_name in ("AudioPre", "AudioPreDeEcho"):
        model_class = getattr(vr_module, class_name)
        original = model_class._path_audio_
        source = textwrap.dedent(inspect.getsource(original))
        for stem in ("wav_instrument", "wav_vocals"):
            before = f'(np.array({stem}) * 32768).astype("int16")'
            if source.count(before) != 2:
                raise RuntimeError("官方 UVR5 音频编码接口发生变化，无法保证浮点输出；已停止分离")
            source = source.replace(before, f"np.asarray({stem}, dtype=np.float32)")
        namespace = dict(original.__globals__)
        namespace["sf"] = SimpleNamespace(write=write_float_wav)
        exec(compile(source, inspect.getfile(original), "exec"), namespace)
        model_class._path_audio_ = namespace[original.__name__]
