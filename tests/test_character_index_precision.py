import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from infer.contracts import read_index, retrieve_features


class CharacterIndexPrecisionTests(unittest.TestCase):
    def test_unicode_index_path_uses_original_index_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '角色索引.index'
            vectors = np.ones((8, 256), dtype=np.float32)
            index = faiss.IndexFlatL2(256)
            index.add(vectors)
            path.write_bytes(faiss.serialize_index(index).tobytes())
            _, restored = read_index(path, 256)
            np.testing.assert_array_equal(restored, vectors)

    def test_sparse_ivf_search_is_exhaustive_and_does_not_rewrite_artifact(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(160, 256)).astype(np.float32)
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(256), 256, 2)
        index.train(vectors)
        index.add(vectors[:1])
        # Stored IVF can return -1 when one partition has fewer than k vectors.
        index.add(vectors[1:9])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.index'
            faiss.write_index(index, str(path))
            before = path.read_bytes()
            exact, loaded = read_index(path, 256, min_vectors=8)
            self.assertIsInstance(exact, faiss.IndexFlatL2)
            np.testing.assert_array_equal(loaded, vectors[:9])
            scores, neighbors = exact.search(vectors[:9], 8)
            self.assertTrue((neighbors >= 0).all())
            self.assertTrue(np.isfinite(scores).all())
            self.assertEqual(before, path.read_bytes())

    def test_exact_duplicate_vectors_do_not_produce_nan_or_borrow_another_vector(self):
        vectors = np.array([[1, 2], [1, 2], [4, 8]], dtype=np.float32)
        index = faiss.IndexFlatL2(2)
        index.add(vectors)
        result = retrieve_features(index, vectors, vectors[:1], 3)
        np.testing.assert_array_equal(result, vectors[:1])

    def test_non_l2_and_quantized_indices_are_rejected(self):
        vectors = np.ones((12, 256), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for index in (faiss.IndexFlatIP(256), faiss.IndexScalarQuantizer(256, faiss.ScalarQuantizer.QT_fp16)):
                index.train(vectors)
                index.add(vectors)
                path = Path(tmp) / 'unsupported.index'
                faiss.write_index(index, str(path))
                with self.assertRaisesRegex(ValueError, '索引类型'):
                    read_index(path, 256)


if __name__ == '__main__':
    unittest.main()
