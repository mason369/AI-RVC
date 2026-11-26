import os
import tempfile
import unittest
from pathlib import Path

from infer.cover_reuse import CoverReuse, asset_version


class CoverReuseTests(unittest.TestCase):
    def test_volume_alias_and_resolved_stem_restore_the_same_float_wav(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = root / "数据卷"
            volume.mkdir()
            alias = root / "temp"
            try:
                alias.symlink_to(volume, target_is_directory=True)
            except OSError as exc:
                if getattr(exc, "winerror", None) == 1314:
                    self.skipTest("Directory symlinks require Windows Developer Mode or privilege")
                raise
            first = alias / "first"
            first.mkdir()
            second = alias / "second"
            second.mkdir()
            physical_stem = volume / "first" / "人声.wav"
            samples = np.array([[0.125, -0.25], [0.33333334, -0.5]], dtype=np.float32)
            sf.write(physical_stem, samples, 44100, subtype="FLOAT")
            state = {}
            key = lambda: ("model",)
            CoverReuse(state, physical_stem, first).save("primary", key, [physical_stem])
            restored, = CoverReuse(state, physical_stem, second).restore("primary", key)
            self.assertTrue(Path(restored).samefile(second / physical_stem.name))
            self.assertEqual(Path(restored).read_bytes(), physical_stem.read_bytes())
            decoded, rate = sf.read(restored, dtype="float32")
            self.assertEqual(rate, 44100)
            np.testing.assert_array_equal(decoded, samples)

    def test_content_identity_and_session_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.wav"
            source.write_bytes(b"first upload")
            original = root / "first"
            original.mkdir()
            stem = original / "vocals.wav"
            stem.write_bytes(b"separated stem")
            state = {}
            CoverReuse(state, source, original).save("primary", lambda: ("model",), [stem])
            next_run = root / "next"
            next_run.mkdir()
            renamed = root / "new-name.wav"
            renamed.write_bytes(source.read_bytes())
            reuse = CoverReuse(state, renamed, next_run)
            self.assertIsNone(reuse.restore("primary", lambda: ("other model",)))
            restored, = reuse.restore("primary", lambda: ("model",))
            self.assertEqual(Path(restored).read_bytes(), stem.read_bytes())
            self.assertNotEqual(Path(restored), stem)
            self.assertIsNone(CoverReuse({}, source, next_run).restore("primary", lambda: ("model",)))
            source.write_bytes(b"other upload")
            self.assertIsNone(CoverReuse(state, source, next_run).restore("primary", lambda: ("model",)))

    def test_missing_or_modified_stem_rebuilds_the_whole_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            first.mkdir()
            next_run = root / "new"
            next_run.mkdir()
            source = root / "song.wav"
            source.write_bytes(b"upload")
            stems = [first / name for name in ("vocals.wav", "accompaniment.wav")]
            for stem in stems:
                stem.write_bytes(b"original")
            state = {}
            key = lambda: ("separator",)
            CoverReuse(state, source, first).save("primary", key, stems)
            asset = asset_version(stems[1])
            stamp = stems[1].stat()
            stems[1].write_bytes(b"modified")
            os.utime(stems[1], ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertNotEqual(asset_version(stems[1]), asset)
            reuse = CoverReuse(state, source, next_run)
            self.assertIsNone(reuse.restore("primary", key))
            stems[1].unlink()
            self.assertIsNone(reuse.restore("primary", key))
            self.assertEqual(list(next_run.iterdir()), [])
            self.assertEqual(reuse.hits, [])
