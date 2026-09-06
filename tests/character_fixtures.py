"""生成小型未训练权重，仅用于合同测试，不作为角色质量证据。"""
import faiss
import numpy as np
import torch


def write_character_fixture(weight, index=None):
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid
    config = [17, 4, 8, 8, 16, 2, 1, 3, 0, '1', [3], [[1, 3, 5]],
              [10, 10, 2, 2], 32, [20, 20, 4, 4], 2, 8, 40000]
    model = SynthesizerTrnMs256NSFsid(*config, is_half=False)
    del model.enc_q
    torch.save({'config': config, 'weight': model.state_dict(), 'f0': 1, 'version': 'v1'}, weight)
    if index is not None:
        artifact = faiss.IndexFlatL2(256)
        artifact.add(np.random.default_rng(17).normal(size=(12, 256)).astype(np.float32))
        faiss.write_index(artifact, str(index))
