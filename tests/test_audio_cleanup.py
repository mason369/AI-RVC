import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from infer.cover_pipeline import CoverPipeline


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _preemphasis_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    residual = np.empty_like(audio)
    residual[0] = audio[0]
    residual[1:] = audio[1:] - 0.97 * audio[:-1]
    return _rms(residual)


def _fade(length: int) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    ramp = max(1, min(length // 4, 512))
    env[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
    return env


def _transition_spike_count(candidate: np.ndarray, reference: np.ndarray, sr: int) -> int:
    import librosa

    frame_length = 2048
    hop_length = 512
    cand_rms = librosa.feature.rms(
        y=np.asarray(candidate, dtype=np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    ref_rms = librosa.feature.rms(
        y=np.asarray(reference, dtype=np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    frame_count = min(cand_rms.size, ref_rms.size)
    if frame_count <= 2:
        return 0
    cand_rms = cand_rms[:frame_count]
    ref_rms = ref_rms[:frame_count]
    cand_delta = np.abs(np.diff(cand_rms))
    ref_delta = np.abs(np.diff(ref_rms))
    spike_mask = (
        (cand_delta > (0.010 + 1.8 * ref_delta))
        & (np.maximum(cand_rms[:-1], cand_rms[1:]) > float(np.percentile(cand_rms, 60)))
    )
    return int(np.sum(spike_mask))


class EchoTailGateTests(unittest.TestCase):
    def test_loud_echo_tail_is_suppressed_when_dereverb_removes_it(self):
        sr = 48000
        t = np.arange(int(2.4 * sr), dtype=np.float32) / sr
        original = np.zeros_like(t)
        dereverbed = np.zeros_like(t)

        direct_mask = (t >= 0.20) & (t < 0.72)
        echo_mask = (t >= 1.16) & (t < 1.68)
        direct = 0.24 * np.sin(2.0 * np.pi * 330.0 * t[direct_mask])
        echo = 0.12 * np.sin(2.0 * np.pi * 330.0 * (t[echo_mask] - 0.96))
        direct *= _fade(direct.size)
        echo *= _fade(echo.size)
        original[direct_mask] = direct
        original[echo_mask] = echo
        dereverbed[direct_mask] = direct

        gain, gated_frames, total_frames = CoverPipeline._compute_echo_tail_sample_gain(
            original=original,
            dereverbed=dereverbed,
            sr=sr,
        )

        direct_gain = gain[int(0.34 * sr): int(0.58 * sr)]
        echo_gain = gain[int(1.28 * sr): int(1.54 * sr)]
        self.assertGreater(total_frames, 0)
        self.assertGreater(gated_frames, 0)
        self.assertGreater(float(np.percentile(direct_gain, 5)), 0.85)
        self.assertLess(
            float(np.mean(echo_gain)),
            0.55,
            "loud echo tails should be reduced even when they are not quiet",
        )


class BreathCleanupTests(unittest.TestCase):
    def test_low_energy_recovery_breath_loses_tonal_hf_without_body_loss(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        converted = np.zeros_like(t)

        body = (t >= 0.25) & (t < 0.90)
        breath = (t >= 1.15) & (t < 1.45)
        source[body] = 0.15 * np.sin(2.0 * np.pi * 260.0 * t[body])
        converted[body] = source[body]

        breath_body = 0.012 * np.sin(2.0 * np.pi * 620.0 * t[breath])
        breath_body *= _fade(breath_body.size)
        source[breath] = breath_body
        converted[breath] = breath_body + 0.009 * np.sin(2.0 * np.pi * 5200.0 * t[breath])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            converted_path = tmp / "converted.wav"
            sf.write(source_path, source, sr)
            sf.write(converted_path, converted, sr)

            pipeline = CoverPipeline(device="cpu")
            before, _ = sf.read(converted_path)
            pipeline._apply_source_breath_cleanup(str(source_path), str(converted_path))
            pipeline._apply_source_transition_cleanup(str(source_path), str(converted_path))
            after, out_sr = sf.read(converted_path)

        self.assertEqual(out_sr, sr)
        before_breath = before[int(1.20 * sr): int(1.40 * sr)]
        after_breath = after[int(1.20 * sr): int(1.40 * sr)]
        before_body = before[int(0.36 * sr): int(0.76 * sr)]
        after_body = after[int(0.36 * sr): int(0.76 * sr)]

        self.assertLess(_preemphasis_rms(after_breath), _preemphasis_rms(before_breath) * 0.80)
        self.assertGreater(_rms(after_body), _rms(before_body) * 0.96)


class SourceTextureGuardTests(unittest.TestCase):
    def test_quality_analysis_reports_highest_severity_transition_times(self):
        sr = 48000
        t = np.arange(int(4.0 * sr), dtype=np.float32) / sr
        reference = 0.08 * np.sin(2.0 * np.pi * 260.0 * t)
        candidate = reference.copy()

        mild_times = [0.42 + 0.10 * i for i in range(14)]
        for time_sec in mild_times:
            mask = (t >= time_sec) & (t < time_sec + 0.035)
            candidate[mask] *= 1.80

        severe_time = 3.10
        severe_mask = (t >= severe_time) & (t < severe_time + 0.040)
        candidate[severe_mask] *= 3.00

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            candidate_path = tmp / "candidate.wav"
            reference_path = tmp / "reference.wav"
            sf.write(candidate_path, candidate, sr)
            sf.write(reference_path, reference, sr)

            analysis = CoverPipeline._analyze_quality_stage(
                candidate_path=str(candidate_path),
                reference_path=str(reference_path),
            )

        times = analysis.get("transition_spike_times_sec", [])
        self.assertTrue(
            any(abs(float(time_sec) - severe_time) <= 0.08 for time_sec in times[:4]),
            f"expected severe late transition near {severe_time}s in top suspect times, got {times}",
        )

    def test_transition_cleanup_keeps_stable_target_timbre_body(self):
        sr = 48000
        t = np.arange(int(1.8 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        converted = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.45)
        burst = (t >= 0.92) & (t < 1.02)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.13 * np.sin(2.0 * np.pi * 520.0 * t[body])
        source[body] = source_body * _fade(source_body.size)
        converted[body] = target_body * _fade(target_body.size)

        burst_noise = 0.11 * np.sin(2.0 * np.pi * 520.0 * t[burst])
        converted[burst] += burst_noise * _fade(burst_noise.size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            converted_path = tmp / "converted.wav"
            sf.write(source_path, source, sr)
            sf.write(converted_path, converted, sr)

            pipeline = CoverPipeline(device="cpu")
            before, _ = sf.read(converted_path)
            pipeline._apply_source_transition_cleanup(str(source_path), str(converted_path))
            after, out_sr = sf.read(converted_path)

        self.assertEqual(out_sr, sr)
        stable_slice = slice(int(0.42 * sr), int(0.78 * sr))
        before_stable = before[stable_slice]
        after_stable = after[stable_slice]
        self.assertLess(_rms(after_stable - before_stable), _rms(before_stable) * 0.04)
        source_stable = source[stable_slice]
        self.assertGreater(
            _rms(after_stable - source_stable),
            _rms(before_stable - source_stable) * 0.94,
        )

    def test_quality_selection_keeps_target_body_but_uses_clean_quiet_segment(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        raw = np.zeros_like(t)
        processed = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.25)
        quiet_artifact = (t >= 1.45) & (t < 1.75)

        source_body = 0.14 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.14 * np.sin(2.0 * np.pi * 520.0 * t[body])
        source[body] = source_body * _fade(source_body.size)
        raw[body] = target_body * _fade(target_body.size)

        processed_body = (
            0.82 * source[body]
            + 0.18 * raw[body]
        )
        processed[body] = processed_body

        raw[quiet_artifact] = (
            0.030
            * np.sin(2.0 * np.pi * 5200.0 * t[quiet_artifact])
            * _fade(int(np.sum(quiet_artifact)))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            raw_path = tmp / "raw.wav"
            processed_path = tmp / "processed.wav"
            output_path = tmp / "selected.wav"
            sf.write(source_path, source, sr)
            sf.write(raw_path, raw, sr)
            sf.write(processed_path, processed, sr)

            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_quality_candidate_selection(
                source_vocals_path=str(source_path),
                raw_candidate_path=str(raw_path),
                processed_candidate_path=str(processed_path),
                output_path=str(output_path),
            )
            selected, out_sr = sf.read(output_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(report["raw_guard_frames"], 0)
        self.assertGreater(report["processed_guard_frames"], 0)

        body_slice = slice(int(0.45 * sr), int(1.05 * sr))
        quiet_slice = slice(int(1.50 * sr), int(1.70 * sr))

        self.assertLess(
            _rms(selected[body_slice] - raw[body_slice]),
            _rms(processed[body_slice] - raw[body_slice]) * 0.35,
        )
        self.assertGreater(
            _rms(selected[body_slice] - source[body_slice]),
            _rms(raw[body_slice] - source[body_slice]) * 0.80,
        )
        self.assertLess(
            _preemphasis_rms(selected[quiet_slice]),
            _preemphasis_rms(raw[quiet_slice]) * 0.45,
        )

    def test_quality_selection_does_not_keep_raw_active_hf_whine(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        raw = np.zeros_like(t)
        processed = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.45)
        source_body = 0.14 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.14 * np.sin(2.0 * np.pi * 520.0 * t[body])
        whine = 0.026 * np.sin(2.0 * np.pi * 5400.0 * t[body])
        env = _fade(source_body.size)

        source[body] = source_body * env
        raw[body] = (target_body + whine) * env
        processed[body] = (0.74 * source_body + 0.26 * target_body) * env

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            raw_path = tmp / "raw.wav"
            processed_path = tmp / "processed.wav"
            output_path = tmp / "selected.wav"
            sf.write(source_path, source, sr)
            sf.write(raw_path, raw, sr)
            sf.write(processed_path, processed, sr)

            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_quality_candidate_selection(
                source_vocals_path=str(source_path),
                raw_candidate_path=str(raw_path),
                processed_candidate_path=str(processed_path),
                output_path=str(output_path),
            )
            selected, out_sr = sf.read(output_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(report["raw_active_artifact_frames"], 0)

        body_slice = slice(int(0.45 * sr), int(1.20 * sr))
        self.assertLess(
            _preemphasis_rms(selected[body_slice]),
            _preemphasis_rms(raw[body_slice]) * 0.72,
        )

    def test_quality_selection_keeps_moderate_bright_target_harmonic(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        raw = np.zeros_like(t)
        processed = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.45)
        source_body = 0.14 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.14 * np.sin(2.0 * np.pi * 520.0 * t[body])
        bright_harmonic = 0.010 * np.sin(2.0 * np.pi * 4160.0 * t[body])
        env = _fade(source_body.size)

        source[body] = source_body * env
        raw[body] = (target_body + bright_harmonic) * env
        processed[body] = (0.78 * source_body + 0.22 * target_body) * env

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            raw_path = tmp / "raw.wav"
            processed_path = tmp / "processed.wav"
            output_path = tmp / "selected.wav"
            sf.write(source_path, source, sr)
            sf.write(raw_path, raw, sr)
            sf.write(processed_path, processed, sr)

            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_quality_candidate_selection(
                source_vocals_path=str(source_path),
                raw_candidate_path=str(raw_path),
                processed_candidate_path=str(processed_path),
                output_path=str(output_path),
            )
            selected, out_sr = sf.read(output_path)

        self.assertEqual(out_sr, sr)
        self.assertEqual(report["raw_active_artifact_frames"], 0)

        body_slice = slice(int(0.45 * sr), int(1.20 * sr))
        self.assertLess(
            _rms(selected[body_slice] - raw[body_slice]),
            _rms(processed[body_slice] - raw[body_slice]) * 0.45,
        )

    def test_quality_selection_rejects_raw_when_it_reintroduces_active_transition_spikes(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        raw = np.zeros_like(t)
        processed = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.55)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.13 * np.sin(2.0 * np.pi * 520.0 * t[body])
        env = _fade(source_body.size)
        source[body] = source_body * env
        raw[body] = target_body * env

        step = (t >= 0.86) & (t < 0.96)
        raw[step] *= 1.58
        processed[body] = (0.60 * source_body + 0.40 * target_body) * env

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            raw_path = tmp / "raw.wav"
            processed_path = tmp / "processed.wav"
            output_path = tmp / "selected.wav"
            sf.write(source_path, source, sr)
            sf.write(raw_path, raw, sr)
            sf.write(processed_path, processed, sr)

            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_quality_candidate_selection(
                source_vocals_path=str(source_path),
                raw_candidate_path=str(raw_path),
                processed_candidate_path=str(processed_path),
                output_path=str(output_path),
            )
            selected, out_sr = sf.read(output_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(report["raw_transition_guard_frames"], 0)
        self.assertLessEqual(
            _transition_spike_count(selected, source, sr),
            _transition_spike_count(processed, source, sr),
        )

    def test_quality_selection_smooths_processed_transition_spikes_when_raw_is_not_usable(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        raw = np.zeros_like(t)
        processed = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.55)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.13 * np.sin(2.0 * np.pi * 520.0 * t[body])
        env = _fade(source_body.size)
        source[body] = source_body * env
        processed[body] = target_body * env

        step = (t >= 0.84) & (t < 0.94)
        processed[step] *= 1.62
        raw[:] = processed

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            raw_path = tmp / "raw.wav"
            processed_path = tmp / "processed.wav"
            output_path = tmp / "selected.wav"
            sf.write(source_path, source, sr)
            sf.write(raw_path, raw, sr)
            sf.write(processed_path, processed, sr)

            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_quality_candidate_selection(
                source_vocals_path=str(source_path),
                raw_candidate_path=str(raw_path),
                processed_candidate_path=str(processed_path),
                output_path=str(output_path),
            )
            selected, out_sr = sf.read(output_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(_transition_spike_count(processed, source, sr), 0)
        self.assertEqual(_transition_spike_count(selected, source, sr), 0)
        self.assertGreater(report["post_selection_residual_transition_frames"], 0)

    def test_active_body_presence_recovery_lifts_dropped_syllable_without_boosting_spike(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        converted = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.55)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.13 * np.sin(2.0 * np.pi * 520.0 * t[body])
        env = _fade(source_body.size)
        source[body] = source_body * env
        converted[body] = target_body * env

        dropped = (t >= 0.82) & (t < 0.94)
        spike = (t >= 1.15) & (t < 1.22)
        converted[dropped] *= 0.42
        converted[spike] *= 1.75

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            converted_path = tmp / "converted.wav"
            sf.write(source_path, source, sr)
            sf.write(converted_path, converted, sr)

            before, _ = sf.read(converted_path)
            pipeline = CoverPipeline(device="cpu")
            report = pipeline._restore_active_body_presence(
                source_vocals_path=str(source_path),
                converted_vocals_path=str(converted_path),
            )
            after, out_sr = sf.read(converted_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(report["active_body_recovery_frames"], 0)
        dropped_slice = slice(int(0.84 * sr), int(0.92 * sr))
        spike_slice = slice(int(1.16 * sr), int(1.20 * sr))
        stable_slice = slice(int(0.42 * sr), int(0.70 * sr))

        self.assertGreater(_rms(after[dropped_slice]), _rms(before[dropped_slice]) * 1.18)
        self.assertLessEqual(_rms(after[spike_slice]), _rms(before[spike_slice]) * 1.02)
        self.assertLess(_rms(after[stable_slice] - before[stable_slice]), _rms(before[stable_slice]) * 0.04)

    def test_quality_selection_recovers_active_body_presence_after_smoothing(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        raw = np.zeros_like(t)
        processed = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.55)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.13 * np.sin(2.0 * np.pi * 520.0 * t[body])
        env = _fade(source_body.size)
        source[body] = source_body * env
        processed[body] = target_body * env

        dropped = (t >= 0.82) & (t < 0.94)
        processed[dropped] *= 0.42
        raw[:] = processed

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            raw_path = tmp / "raw.wav"
            processed_path = tmp / "processed.wav"
            output_path = tmp / "selected.wav"
            sf.write(source_path, source, sr)
            sf.write(raw_path, raw, sr)
            sf.write(processed_path, processed, sr)

            before, _ = sf.read(processed_path)
            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_quality_candidate_selection(
                source_vocals_path=str(source_path),
                raw_candidate_path=str(raw_path),
                processed_candidate_path=str(processed_path),
                output_path=str(output_path),
            )
            selected, out_sr = sf.read(output_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(report["post_selection_active_body_recovery_frames"], 0)
        dropped_slice = slice(int(0.84 * sr), int(0.92 * sr))
        self.assertGreater(_rms(selected[dropped_slice]), _rms(before[dropped_slice]) * 1.18)

    def test_mix_foreground_preparation_lifts_vocal_balance_safely(self):
        import librosa

        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        vocals = np.zeros_like(t)
        accompaniment = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.55)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.09 * np.sin(2.0 * np.pi * 520.0 * t[body])
        accompaniment_body = 0.14 * np.sin(2.0 * np.pi * 180.0 * t[body])
        env = _fade(source_body.size)
        source[body] = source_body * env
        vocals[body] = target_body * env
        accompaniment[body] = accompaniment_body * env

        def _active_balance_db(vocal_audio: np.ndarray, accompaniment_audio: np.ndarray) -> float:
            weights = CoverPipeline._compute_activity_sample_weights(source, sr)
            vocal_rms = CoverPipeline._weighted_rms(vocal_audio, weights)
            accompaniment_rms = CoverPipeline._weighted_rms(accompaniment_audio, weights)
            return float(20.0 * np.log10((vocal_rms + 1e-8) / (accompaniment_rms + 1e-8)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            vocals_path = tmp / "vocals.wav"
            accompaniment_path = tmp / "accompaniment.wav"
            output_path = tmp / "foreground.wav"
            sf.write(source_path, source, sr)
            sf.write(vocals_path, vocals, sr)
            sf.write(accompaniment_path, accompaniment, sr)

            before_db = _active_balance_db(vocals, accompaniment)
            pipeline = CoverPipeline(device="cpu")
            report = pipeline._prepare_mix_vocal_foreground(
                source_vocals_path=str(source_path),
                vocals_path=str(vocals_path),
                accompaniment_path=str(accompaniment_path),
                output_path=str(output_path),
            )
            foreground, out_sr = sf.read(report["vocals_path"])
            after_db = _active_balance_db(foreground, accompaniment)

        self.assertEqual(out_sr, sr)
        self.assertLess(before_db, -2.0)
        self.assertGreater(report["foreground_gain"], 1.0)
        self.assertGreater(after_db, before_db + 0.5)
        self.assertLessEqual(float(np.max(np.abs(foreground))), 0.88)

    def test_residual_transition_smoothing_reduces_active_envelope_spikes_without_body_loss(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        converted = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.55)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.13 * np.sin(2.0 * np.pi * 520.0 * t[body])
        env = _fade(source_body.size)
        source[body] = source_body * env
        converted[body] = target_body * env

        spike = (t >= 0.84) & (t < 0.94)
        converted[spike] *= 1.62

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            converted_path = tmp / "converted.wav"
            sf.write(source_path, source, sr)
            sf.write(converted_path, converted, sr)

            before, _ = sf.read(converted_path)
            before_spikes = _transition_spike_count(before, source, sr)
            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_residual_transition_smoothing(
                source_vocals_path=str(source_path),
                converted_vocals_path=str(converted_path),
            )
            after, out_sr = sf.read(converted_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(before_spikes, 0)
        self.assertGreater(report["residual_transition_frames"], 0)
        self.assertLess(
            _transition_spike_count(after, source, sr),
            before_spikes,
        )
        stable_slice = slice(int(0.42 * sr), int(0.72 * sr))
        self.assertGreater(_rms(after[stable_slice]), _rms(before[stable_slice]) * 0.96)

    def test_residual_transition_smoothing_converges_on_hard_active_step(self):
        sr = 48000
        t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
        source = np.zeros_like(t)
        converted = np.zeros_like(t)

        body = (t >= 0.25) & (t < 1.55)
        source_body = 0.13 * np.sin(2.0 * np.pi * 260.0 * t[body])
        target_body = 0.13 * np.sin(2.0 * np.pi * 520.0 * t[body])
        env = _fade(source_body.size)
        source[body] = source_body * env
        converted[body] = target_body * env

        hard_step = (t >= 0.84) & (t < 0.94)
        converted[hard_step] *= 2.50

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / "source.wav"
            converted_path = tmp / "converted.wav"
            sf.write(source_path, source, sr)
            sf.write(converted_path, converted, sr)

            before, _ = sf.read(converted_path)
            pipeline = CoverPipeline(device="cpu")
            report = pipeline._apply_residual_transition_smoothing(
                source_vocals_path=str(source_path),
                converted_vocals_path=str(converted_path),
            )
            after, out_sr = sf.read(converted_path)

        self.assertEqual(out_sr, sr)
        self.assertGreater(_transition_spike_count(before, source, sr), 0)
        self.assertEqual(_transition_spike_count(after, source, sr), 0)
        self.assertGreaterEqual(report["passes"], 2)
        stable_slice = slice(int(0.42 * sr), int(0.72 * sr))
        self.assertGreater(_rms(after[stable_slice]), _rms(before[stable_slice]) * 0.96)


if __name__ == "__main__":
    unittest.main()
