# -*- coding: utf-8 -*-
"""
翻唱流水线 - 整合人声分离、RVC转换、混音的完整流程
"""
import os
import gc
import re
import json
import uuid
import shutil
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, List

from infer.separator import (
    VocalSeparator,
    RoformerSeparator,
    RoformerDereverbSeparator,
    KaraokeSeparator,
    ROFORMER_DEFAULT_MODEL,
    ROFORMER_DEREVERB_DEFAULT_MODEL,
    KARAOKE_DEFAULT_MODEL,
    check_demucs_available,
    check_roformer_available,
    get_available_models,
)
from infer.official_adapter import (
    setup_official_env,
    separate_uvr5,
    separate_uvr5_official_upstream,
    convert_vocals_official,
    convert_vocals_official_upstream,
)
from infer.advanced_dereverb import advanced_dereverb, apply_reverb_to_converted
from infer.quality_policy import (
    compute_active_source_replace,
    compute_source_cleanup_budget,
    resolve_cover_f0_policy,
)
from lib.audio import soft_clip
from lib.mixer import mix_vocals_and_accompaniment
from lib.logger import log
from lib.device import get_device, empty_device_cache


def _format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def _get_audio_duration(file_path: str) -> float:
    """获取音频时长（秒）"""
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    except:
        return 0.0


def _format_duration(seconds: float) -> str:
    """格式化时长"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


class CoverPipeline:
    """AI 翻唱流水线"""

    def __init__(self, device: str = "cuda"):
        """
        初始化流水线

        Args:
            device: 计算设备
        """
        self.device = str(get_device(device))
        self.separator = None
        self.karaoke_separator = None
        self.rvc_pipeline = None
        self.temp_dir = Path(__file__).parent.parent / "temp" / "cover"
        self._last_vc_preprocess_mode = "direct"

    def _get_session_dir(self, session_id: str = None) -> Path:
        """获取会话临时目录"""
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        session_dir = self.temp_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def _get_available_uvr_deecho_model() -> Optional[str]:
        """优先使用学习型 DeEcho / DeReverb，而不是手工频谱去回声。"""
        root = Path(__file__).parent.parent / "assets" / "uvr5_weights"
        candidates = [
            ("VR-DeEchoDeReverb", root / "VR-DeEchoDeReverb.pth"),
            ("onnx_dereverb_By_FoxJoy", root / "onnx_dereverb_By_FoxJoy" / "vocals.onnx"),
            ("VR-DeEchoNormal", root / "VR-DeEchoNormal.pth"),
            ("VR-DeEchoAggressive", root / "VR-DeEchoAggressive.pth"),
        ]
        for model_name, model_path in candidates:
            if model_path.exists():
                return model_name
        return None

    @staticmethod
    def _get_preferred_deecho_model_label() -> Optional[str]:
        """Return the highest-quality learned deecho route available to this install."""
        if check_roformer_available():
            return f"RoFormer:{ROFORMER_DEREVERB_DEFAULT_MODEL}"
        uvr_model = CoverPipeline._get_available_uvr_deecho_model()
        if uvr_model:
            return f"UVR:{uvr_model}"
        return None

    def _apply_roformer_deecho_for_vc(self, vocals_path: str, session_dir: Path) -> Optional[str]:
        """Use the public RoFormer dereverb/deecho model before legacy UVR DeEcho."""
        if not check_roformer_available():
            return None

        deecho_dir = session_dir / "vc_roformer_deecho"
        separator = RoformerDereverbSeparator(device=self.device)

        try:
            log.model(
                "VC预处理使用 RoFormer De-Reverb SOTA 模型: "
                f"{ROFORMER_DEREVERB_DEFAULT_MODEL}"
            )
            dry_path = separator.separate_dry(vocals_path, str(deecho_dir))
            scored = self._score_uvr_deecho_candidate(vocals_path, Path(dry_path))
            if scored is not None:
                _, metrics = scored
                self._uvr_deecho_metrics = metrics
                log.detail(
                    "RoFormer De-Reverb candidate: "
                    f"{Path(dry_path).name}, score={metrics['score']:.2f}, "
                    f"sep={metrics['separation_db']:.2f}dB, corr={metrics['corr']:.3f}, "
                    f"ratio={metrics['active_ratio']:.3f}, "
                    f"reduction={metrics['reduction_ratio']:.3f}"
                )
            log.audio(f"RoFormer De-Reverb selected dry vocal output: {Path(dry_path).name}")
            return dry_path
        except Exception as exc:
            log.warning(f"RoFormer De-Reverb 预处理失败，回退 UVR DeEcho: {exc}")
            return None
        finally:
            separator.unload_model()

    def _apply_uvr_deecho_for_vc(self, vocals_path: str, session_dir: Path) -> Optional[str]:
        """如果本地已有 UVR DeEcho 模型，则优先用学习型方法清理回声。"""
        model_name = self._get_available_uvr_deecho_model()
        if not model_name:
            return None

        from infer.modules.uvr5.modules import uvr

        root = Path(__file__).parent.parent
        os.environ["weight_uvr5_root"] = str(root / "assets" / "uvr5_weights")

        input_dir = session_dir / "vc_deecho_input"
        vocal_dir = session_dir / "vc_deecho_vocal"
        ins_dir = session_dir / "vc_deecho_ins"
        input_dir.mkdir(parents=True, exist_ok=True)
        vocal_dir.mkdir(parents=True, exist_ok=True)
        ins_dir.mkdir(parents=True, exist_ok=True)

        input_file = input_dir / Path(vocals_path).name
        shutil.copy2(vocals_path, input_file)

        log.model(f"VC预处理使用UVR DeEcho模型: {model_name}")
        for _ in uvr(model_name, str(input_dir), str(vocal_dir), [], str(ins_dir), 10, "wav"):
            pass

        candidate_files = sorted(
            list(vocal_dir.glob("*.wav")) + list(ins_dir.glob("*.wav")),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidate_files:
            log.warning("UVR DeEcho produced no usable vocal output; falling back to direct lead input")
            return None

        selected_file = self._select_best_uvr_deecho_output(vocals_path, candidate_files)
        if selected_file is None:
            selected_file = candidate_files[-1]
        log.audio(f"UVR DeEcho selected vocal output: {selected_file.name}")
        return str(selected_file)

    @staticmethod
    def _score_uvr_deecho_candidate(reference_path: str, candidate_path: Path) -> Optional[Tuple[float, Dict[str, float]]]:
        """Score UVR DeEcho candidate for VC: keep direct lead, minimize quiet residuals."""
        import librosa

        try:
            reference_audio, reference_sr = librosa.load(reference_path, sr=None, mono=True)
            candidate_audio, candidate_sr = librosa.load(str(candidate_path), sr=None, mono=True)
        except Exception:
            return None

        reference_audio = np.asarray(reference_audio, dtype=np.float32)
        candidate_audio = np.asarray(candidate_audio, dtype=np.float32)
        if reference_audio.size == 0 or candidate_audio.size == 0:
            return None

        if candidate_sr != reference_sr:
            candidate_audio = librosa.resample(
                candidate_audio,
                orig_sr=candidate_sr,
                target_sr=reference_sr,
            ).astype(np.float32)

        aligned_len = min(reference_audio.size, candidate_audio.size)
        if aligned_len <= 2048:
            return None

        reference_audio = reference_audio[:aligned_len]
        candidate_audio = candidate_audio[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8
        frame_rms = librosa.feature.rms(
            y=reference_audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        if frame_rms.size == 0:
            return None

        frame_db = 20.0 * np.log10(frame_rms + eps)
        ref_db = float(np.percentile(frame_db, 95))
        active_frames = frame_db > (ref_db - 24.0)
        quiet_frames = frame_db < (ref_db - 36.0)

        active_mask = np.repeat(active_frames.astype(np.float32), hop_length)
        quiet_mask = np.repeat(quiet_frames.astype(np.float32), hop_length)
        if active_mask.size < aligned_len:
            active_mask = np.pad(active_mask, (0, aligned_len - active_mask.size), mode="edge")
        if quiet_mask.size < aligned_len:
            quiet_mask = np.pad(quiet_mask, (0, aligned_len - quiet_mask.size), mode="edge")
        active_mask = active_mask[:aligned_len] > 0.5
        quiet_mask = quiet_mask[:aligned_len] > 0.5

        if not np.any(active_mask):
            return None

        active_rms = float(np.sqrt(np.mean(np.square(candidate_audio[active_mask])) + 1e-12))
        quiet_rms = float(np.sqrt(np.mean(np.square(candidate_audio[quiet_mask])) + 1e-12)) if np.any(quiet_mask) else 1e-6
        ref_active_rms = float(np.sqrt(np.mean(np.square(reference_audio[active_mask])) + 1e-12))
        corr = 0.0
        if np.sum(active_mask) > 32:
            corr_val = np.corrcoef(reference_audio[active_mask], candidate_audio[active_mask])[0, 1]
            if np.isfinite(corr_val):
                corr = float(np.clip(corr_val, -1.0, 1.0))

        separation_db = float(20.0 * np.log10((active_rms + 1e-12) / (quiet_rms + 1e-12)))
        active_ratio = float(active_rms / (ref_active_rms + 1e-12))
        ratio_penalty = abs(float(np.log2(max(active_ratio, 1e-4))))
        active_diff_rms = float(
            np.sqrt(np.mean(np.square(reference_audio[active_mask] - candidate_audio[active_mask])) + 1e-12)
        )
        reduction_ratio = float(active_diff_rms / (ref_active_rms + 1e-12))
        score = separation_db + 18.0 * corr - 6.0 * ratio_penalty

        return score, {
            "score": score,
            "separation_db": separation_db,
            "corr": corr,
            "active_ratio": active_ratio,
            "reduction_ratio": reduction_ratio,
        }

    def _select_best_uvr_deecho_output(self, reference_path: str, candidate_files: List[Path]) -> Optional[Path]:
        """Pick the UVR DeEcho branch best suited for VC input."""
        best_path = None
        best_score = None
        best_metrics = None

        for candidate_path in candidate_files:
            scored = self._score_uvr_deecho_candidate(reference_path, candidate_path)
            if scored is None:
                continue

            score, metrics = scored
            log.detail(
                "UVR DeEcho candidate: "
                f"{candidate_path.name}, score={metrics['score']:.2f}, "
                f"sep={metrics['separation_db']:.2f}dB, corr={metrics['corr']:.3f}, "
                f"ratio={metrics['active_ratio']:.3f}, "
                f"reduction={metrics['reduction_ratio']:.3f}"
            )
            if best_score is None or score > best_score:
                best_score = score
                best_path = candidate_path
                best_metrics = metrics

        # 保存最佳候选的质量指标，供 blend 决策使用
        self._uvr_deecho_metrics = best_metrics
        return best_path

    @staticmethod
    def _save_debug_audio_snapshot(
        session_dir: Path,
        audio_path: str,
        label: str,
    ) -> Optional[str]:
        source_path = Path(audio_path)
        if not source_path.exists():
            return None
        suffix = source_path.suffix or ".wav"
        snapshot_path = session_dir / f"{label}{suffix}"
        shutil.copy2(source_path, snapshot_path)
        return str(snapshot_path)

    @staticmethod
    def _append_quality_debug_entry(session_dir: Path, entry: Dict[str, object]) -> None:
        report_path = session_dir / "quality_debug.json"
        payload: Dict[str, object] = {"stages": []}
        if report_path.exists():
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"stages": []}
        stages = payload.get("stages")
        if not isinstance(stages, list):
            stages = []
        stages.append(entry)
        payload["stages"] = stages
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _export_suspect_transition_clips(
        session_dir: Path,
        stage: str,
        candidate_path: str,
        reference_path: Optional[str],
        suspect_times: List[float],
        max_clips: int = 6,
        clip_duration_sec: float = 1.2,
    ) -> List[str]:
        import soundfile as sf

        if not suspect_times:
            return []

        clip_dir = session_dir / "debug_clips" / stage
        clip_dir.mkdir(parents=True, exist_ok=True)

        exported: List[str] = []

        def _export_one(prefix: str, path: str, time_sec: float) -> Optional[str]:
            if not path or not Path(path).exists():
                return None
            audio, sr = sf.read(path, always_2d=True)
            audio = np.asarray(audio, dtype=np.float32)
            clip_samples = int(max(1, round(float(clip_duration_sec) * sr)))
            center = int(round(float(time_sec) * sr))
            start = max(0, center - clip_samples // 2)
            end = min(audio.shape[0], start + clip_samples)
            start = max(0, end - clip_samples)
            clip = audio[start:end]
            out_path = clip_dir / f"{prefix}_{float(time_sec):07.3f}s.wav"
            sf.write(str(out_path), clip, sr)
            return str(out_path)

        for time_sec in suspect_times[: max(1, int(max_clips))]:
            candidate_clip = _export_one("candidate", candidate_path, float(time_sec))
            reference_clip = _export_one("reference", reference_path, float(time_sec)) if reference_path else None
            if candidate_clip:
                exported.append(candidate_clip)
            if reference_clip:
                exported.append(reference_clip)

        return exported

    @staticmethod
    def _analyze_quality_stage(
        candidate_path: str,
        reference_path: Optional[str] = None,
    ) -> Dict[str, object]:
        import librosa
        import soundfile as sf

        def _load_mono(path: str) -> Tuple[np.ndarray, int]:
            audio, sr = sf.read(path, always_2d=True)
            audio = np.asarray(audio, dtype=np.float32)
            mono = audio.mean(axis=1)
            return mono.astype(np.float32), int(sr)

        def _safe_rms(values: np.ndarray, mask: np.ndarray) -> float:
            values = np.asarray(values, dtype=np.float32)
            mask = np.asarray(mask, dtype=bool)
            if values.size == 0 or mask.size == 0 or not np.any(mask):
                return 0.0
            return float(np.sqrt(np.mean(np.square(values[mask])) + 1e-12))

        def _preemphasis_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            residual = np.empty_like(audio)
            residual[0] = audio[0]
            residual[1:] = audio[1:] - 0.97 * audio[:-1]
            return librosa.feature.rms(
                y=residual,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        candidate_audio, candidate_sr = _load_mono(candidate_path)
        aligned_len = int(candidate_audio.size)
        peak = float(np.max(np.abs(candidate_audio))) if aligned_len > 0 else 0.0
        clip_samples = int(np.sum(np.abs(candidate_audio) >= 0.995)) if aligned_len > 0 else 0
        analysis: Dict[str, object] = {
            "sample_rate": int(candidate_sr),
            "duration_sec": float(aligned_len / max(candidate_sr, 1)),
            "global_rms": float(np.sqrt(np.mean(np.square(candidate_audio)) + 1e-12)) if aligned_len > 0 else 0.0,
            "peak": peak,
            "clip_samples": clip_samples,
            "clip_ratio": float(clip_samples / max(aligned_len, 1)),
        }

        if not reference_path:
            return analysis

        reference_audio, reference_sr = _load_mono(reference_path)
        if reference_sr != candidate_sr:
            reference_audio = librosa.resample(
                reference_audio,
                orig_sr=reference_sr,
                target_sr=candidate_sr,
            ).astype(np.float32)

        aligned_len = min(reference_audio.size, candidate_audio.size)
        if aligned_len <= 2048:
            return analysis

        reference_audio = reference_audio[:aligned_len]
        candidate_audio = candidate_audio[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8

        ref_rms = librosa.feature.rms(
            y=reference_audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        cand_rms = librosa.feature.rms(
            y=candidate_audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        ref_hf = _preemphasis_rms(reference_audio, frame_length, hop_length)
        cand_hf = _preemphasis_rms(candidate_audio, frame_length, hop_length)
        frame_count = min(ref_rms.size, cand_rms.size, ref_hf.size, cand_hf.size)
        if frame_count <= 4:
            return analysis

        ref_rms = ref_rms[:frame_count]
        cand_rms = cand_rms[:frame_count]
        ref_hf = ref_hf[:frame_count]
        cand_hf = cand_hf[:frame_count]

        ref_db = 20.0 * np.log10(ref_rms + eps)
        ref_db_peak = float(np.percentile(ref_db, 95))
        active_mask = ref_db >= (ref_db_peak - 24.0)
        quiet_mask = ref_db <= (ref_db_peak - 38.0)
        midquiet_mask = (ref_db > (ref_db_peak - 38.0)) & (ref_db <= (ref_db_peak - 28.0))

        corr_active = 0.0
        if np.sum(active_mask) > 8:
            corr_val = np.corrcoef(ref_rms[active_mask], cand_rms[active_mask])[0, 1]
            if np.isfinite(corr_val):
                corr_active = float(np.clip(corr_val, -1.0, 1.0))

        cand_delta = np.abs(np.diff(cand_rms))
        ref_delta = np.abs(np.diff(ref_rms))
        spike_threshold = 0.010 + 1.8 * ref_delta
        spike_mask = (
            (cand_delta > spike_threshold)
            & (np.maximum(cand_rms[:-1], cand_rms[1:]) > float(np.percentile(cand_rms, 60)))
        )
        suspect_frames = np.where(spike_mask)[0]
        suspect_times = [
            round(float(idx * hop_length / candidate_sr), 3)
            for idx in suspect_frames[:12]
        ]

        low_mid_energy = np.clip((ref_db_peak - 18.0 - ref_db) / 22.0, 0.0, 1.0)
        not_sustained_gap = 1.0 - np.clip((ref_db_peak - 42.0 - ref_db) / 8.0, 0.0, 1.0)
        breath_excess = low_mid_energy * not_sustained_gap * np.maximum(
            np.clip((cand_rms - 1.15 * ref_rms) / (cand_rms + eps), 0.0, 1.0),
            np.clip((cand_hf - 1.10 * ref_hf) / (cand_hf + eps), 0.0, 1.0),
        )

        analysis.update(
            {
                "reference_duration_sec": float(reference_audio.size / max(candidate_sr, 1)),
                "active_rms_ratio": float(_safe_rms(cand_rms, active_mask) / (_safe_rms(ref_rms, active_mask) + 1e-12)),
                "active_hf_ratio": float(_safe_rms(cand_hf, active_mask) / (_safe_rms(ref_hf, active_mask) + 1e-12)),
                "quiet_rms_ratio": float(_safe_rms(cand_rms, quiet_mask) / (_safe_rms(ref_rms, quiet_mask) + 1e-12)),
                "midquiet_rms_ratio": float(_safe_rms(cand_rms, midquiet_mask) / (_safe_rms(ref_rms, midquiet_mask) + 1e-12)),
                "quiet_hf_ratio": float(_safe_rms(cand_hf, quiet_mask) / (_safe_rms(ref_hf, quiet_mask) + 1e-12)),
                "midquiet_hf_ratio": float(_safe_rms(cand_hf, midquiet_mask) / (_safe_rms(ref_hf, midquiet_mask) + 1e-12)),
                "corr_active": corr_active,
                "transition_spike_ratio": float(np.percentile(cand_delta, 95) / (np.percentile(ref_delta, 95) + 1e-12)),
                "transition_spike_frames": int(suspect_frames.size),
                "transition_spike_times_sec": suspect_times,
                "synthetic_breath_frames": int(np.sum(breath_excess > 0.20)),
            }
        )
        return analysis

    def _record_quality_debug(
        self,
        session_dir: Path,
        stage: str,
        candidate_path: str,
        reference_path: Optional[str] = None,
        snapshot_label: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        try:
            snapshot_path = None
            if snapshot_label:
                snapshot_path = self._save_debug_audio_snapshot(
                    session_dir=session_dir,
                    audio_path=candidate_path,
                    label=snapshot_label,
                )
            analysis = self._analyze_quality_stage(
                candidate_path=snapshot_path or candidate_path,
                reference_path=reference_path,
            )
            entry: Dict[str, object] = {
                "stage": stage,
                "candidate_path": str(snapshot_path or candidate_path),
                "reference_path": str(reference_path) if reference_path else None,
                "preprocess_mode": self._last_vc_preprocess_mode,
                "analysis": analysis,
            }
            if extra:
                entry["extra"] = extra
            suspect_times = analysis.get("transition_spike_times_sec", [])
            if isinstance(suspect_times, list) and suspect_times:
                exported_clips = self._export_suspect_transition_clips(
                    session_dir=session_dir,
                    stage=stage,
                    candidate_path=str(snapshot_path or candidate_path),
                    reference_path=reference_path,
                    suspect_times=[float(t) for t in suspect_times],
                )
                if exported_clips:
                    entry["suspect_clips"] = exported_clips
            self._append_quality_debug_entry(session_dir, entry)

            if reference_path:
                log.detail(
                    f"Quality[{stage}]: "
                    f"active={analysis.get('active_rms_ratio', 0.0):.3f}, "
                    f"active_hf={analysis.get('active_hf_ratio', 0.0):.3f}, "
                    f"quiet={analysis.get('quiet_rms_ratio', 0.0):.3f}, "
                    f"midquiet={analysis.get('midquiet_rms_ratio', 0.0):.3f}, "
                    f"quiet_hf={analysis.get('quiet_hf_ratio', 0.0):.3f}, "
                    f"spike={analysis.get('transition_spike_ratio', 0.0):.3f}, "
                    f"breath_frames={analysis.get('synthetic_breath_frames', 0)}"
                )
                suspect_times = analysis.get("transition_spike_times_sec", [])
                if isinstance(suspect_times, list) and suspect_times:
                    times_text = ", ".join(f"{float(t):.2f}s" for t in suspect_times[:8])
                    log.detail(f"Quality[{stage}] suspect transitions: {times_text}")
            else:
                log.detail(
                    f"Quality[{stage}]: "
                    f"peak={analysis.get('peak', 0.0):.3f}, "
                    f"clip_ratio={analysis.get('clip_ratio', 0.0):.6f}, "
                    f"rms={analysis.get('global_rms', 0.0):.4f}"
                )
        except Exception as e:
            log.warning(f"Quality debug capture failed at {stage}: {e}")

    @staticmethod
    def _load_quality_stage_analysis(
        session_dir: Path,
        stage: str,
    ) -> Optional[Dict[str, object]]:
        report_path = session_dir / "quality_debug.json"
        if not report_path.exists():
            return None
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None

        stages = payload.get("stages", [])
        if not isinstance(stages, list):
            return None

        for entry in reversed(stages):
            if not isinstance(entry, dict):
                continue
            if entry.get("stage") != stage:
                continue
            analysis = entry.get("analysis")
            if isinstance(analysis, dict):
                return analysis
        return None

    def _maybe_log_diagnostic_hint(
        self,
        session_dir: Path,
        model_path: str,
        index_path: Optional[str],
    ) -> None:
        final_analysis = self._load_quality_stage_analysis(session_dir, "vc_final_state")
        raw_analysis = self._load_quality_stage_analysis(session_dir, "vc_raw")
        analysis = final_analysis or raw_analysis
        if not analysis:
            return

        quiet_rms = float(analysis.get("quiet_rms_ratio", 0.0) or 0.0)
        quiet_hf = float(analysis.get("quiet_hf_ratio", 0.0) or 0.0)
        spike = float(analysis.get("transition_spike_ratio", 0.0) or 0.0)
        breath_frames = int(analysis.get("synthetic_breath_frames", 0) or 0)
        should_hint = (
            quiet_rms >= 3.0
            or quiet_hf >= 2.5
            or spike >= 1.6
            or breath_frames >= 350
        )
        if not should_hint:
            return

        command = (
            f'python tools/diagnose_vc_session.py --session-dir "{session_dir}" '
            f'--model-path "{model_path}"'
        )
        if index_path:
            command += f' --index-path "{index_path}"'

        log.warning(
            "Persistent VC artifacts detected after final cleanup; "
            "run the diagnostic matrix to compare short A/B clips."
        )
        log.warning(
            "Diagnostic summary: "
            f"quiet={quiet_rms:.3f}, quiet_hf={quiet_hf:.3f}, "
            f"spike={spike:.3f}, breath_frames={breath_frames}"
        )
        log.detail(f"Diagnostic command: {command}")

    def _init_separator(
        self,
        model_name: str = "htdemucs",
        shifts: int = 2,
        overlap: float = 0.25,
        split: bool = True,
        roformer_model: str = ROFORMER_DEFAULT_MODEL,
    ):
        """初始化人声分离器 (Demucs 或 Roformer)"""
        # Roformer 模式
        if model_name == "roformer":
            if not check_roformer_available():
                raise ImportError(
                    "请安装 audio-separator: pip install audio-separator[gpu]"
                )
            if (
                self.separator is not None
                and isinstance(self.separator, RoformerSeparator)
                and getattr(self.separator, "model_filename", None) == roformer_model
            ):
                return
            if self.separator is not None:
                self.separator.unload_model()
                self.separator = None
            self.separator = RoformerSeparator(
                model_filename=roformer_model,
                device=self.device,
            )
            return

        # Demucs 模式
        if not check_demucs_available():
            raise ImportError("请安装 demucs: pip install demucs")

        available = {m["name"] for m in get_available_models() if m["name"] != "roformer"}
        if model_name not in available:
            log.warning(
                f"未知的 Demucs 模型 '{model_name}'，回退到 'htdemucs'"
            )
            model_name = "htdemucs"

        if (
            self.separator is not None
            and isinstance(self.separator, VocalSeparator)
            and getattr(self.separator, "model_name", None) == model_name
            and getattr(self.separator, "shifts", None) == shifts
            and getattr(self.separator, "overlap", None) == overlap
            and getattr(self.separator, "split", None) == split
        ):
            return

        if self.separator is not None:
            self.separator.unload_model()
            self.separator = None

        self.separator = VocalSeparator(
            model_name=model_name,
            device=self.device,
            shifts=shifts,
            overlap=overlap,
            split=split
        )

    def _init_karaoke_separator(self, model_name: str = KARAOKE_DEFAULT_MODEL):
        """初始化主唱/和声分离器"""
        if not check_roformer_available():
            raise ImportError("请安装 audio-separator: pip install audio-separator[gpu]")

        if (
            self.karaoke_separator is not None
            and isinstance(self.karaoke_separator, KaraokeSeparator)
            and getattr(self.karaoke_separator, "model_candidates", [None])[0] == model_name
        ):
            return

        if self.karaoke_separator is not None:
            self.karaoke_separator.unload_model()
            self.karaoke_separator = None

        self.karaoke_separator = KaraokeSeparator(
            model_filename=model_name,
            device=self.device,
        )

    def _separate_karaoke(
        self,
        vocals_path: str,
        session_dir: Path,
        karaoke_model: str = KARAOKE_DEFAULT_MODEL,
    ) -> Tuple[str, str]:
        """分离主唱与和声，并在分离后立即释放显存"""
        karaoke_dir = session_dir / "karaoke"
        karaoke_dir.mkdir(parents=True, exist_ok=True)

        self._init_karaoke_separator(karaoke_model)
        lead_vocals_path, backing_vocals_path = self.karaoke_separator.separate(
            vocals_path,
            str(karaoke_dir),
        )

        if self.karaoke_separator is not None:
            self.karaoke_separator.unload_model()
            self.karaoke_separator = None
        gc.collect()
        empty_device_cache()

        return lead_vocals_path, backing_vocals_path

    @staticmethod
    def _ensure_2d(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio[np.newaxis, :]
        return audio

    @staticmethod
    def _match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
        if audio.shape[0] == channels:
            return audio
        if audio.shape[0] == 1 and channels == 2:
            return np.repeat(audio, 2, axis=0)
        if audio.shape[0] == 2 and channels == 1:
            return np.mean(audio, axis=0, keepdims=True)
        if audio.shape[0] > channels:
            return audio[:channels]
        repeats = channels - audio.shape[0]
        if repeats <= 0:
            return audio
        return np.concatenate([audio, np.repeat(audio[-1:, :], repeats, axis=0)], axis=0)

    @staticmethod
    def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        import librosa

        if audio.ndim == 1:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return np.stack(
            [librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr) for ch in audio],
            axis=0,
        )

    @staticmethod
    def _estimate_echo_metric(audio: np.ndarray, sr: int) -> float:
        """Estimate echo/reverb amount from RMS-envelope autocorrelation."""
        import librosa

        if audio.size == 0:
            return 1.0
        rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256, center=True)[0]
        if rms.size < 8:
            return 1.0
        rms = rms - float(np.mean(rms))
        denom = float(np.dot(rms, rms) + 1e-8)
        if denom <= 0:
            return 1.0
        ac = np.correlate(rms, rms, mode="full")[len(rms) - 1 :] / denom
        lag_min = max(1, int(0.03 * sr / 256))   # 30ms
        lag_max = max(lag_min + 1, int(0.12 * sr / 256))  # 120ms
        lag_max = min(lag_max, len(ac))
        if lag_min >= lag_max:
            return 1.0
        return float(np.max(ac[lag_min:lag_max]))

    def _select_mono_for_vc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Pick the least-echo mono candidate from {L, R, Mid} to avoid phase-mix artifacts.
        """
        audio = self._ensure_2d(audio).astype(np.float32)
        if audio.shape[0] == 1:
            return audio[0]

        left = audio[0]
        right = audio[1] if audio.shape[0] > 1 else audio[0]
        mid = 0.5 * (left + right)
        candidates = {
            "left": left,
            "right": right,
            "mid": mid,
        }
        best_name = None
        best_score = None
        for name, cand in candidates.items():
            score = self._estimate_echo_metric(cand, sr)
            if best_score is None or score < best_score:
                best_name = name
                best_score = score

        if log:
            log.detail(
                f"VC输入单声道选择: {best_name}, 回声指标={best_score:.4f}"
            )
        return candidates[best_name]

    @staticmethod
    def _dereverb_for_vc(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        智能去混响：区分自然混响和真实回声，动态调整抑制强度
        """
        import librosa

        if audio.size == 0:
            return audio
        x = audio.astype(np.float32)
        n_fft = 2048
        hop = 512
        win = 2048
        eps = 1e-8

        spec = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=win)
        mag = np.abs(spec).astype(np.float32)
        phase = np.exp(1j * np.angle(spec))

        if mag.shape[1] < 4:
            return x

        # 计算RMS能量曲线，用于区分高能量段和低能量段
        rms = librosa.feature.rms(y=x, frame_length=win, hop_length=hop, center=True)[0]
        rms_db = 20.0 * np.log10(rms + eps)
        ref_db = float(np.percentile(rms_db, 90))

        # 高能量段（主唱强的地方）：vocal_strength接近1
        # 低能量段（回声尾巴）：vocal_strength接近0
        vocal_strength = np.clip((rms_db - (ref_db - 35.0)) / 25.0, 0.0, 1.0)
        vocal_strength = np.pad(vocal_strength, (0, mag.shape[1] - len(vocal_strength)), mode='edge')

        late = np.zeros_like(mag, dtype=np.float32)
        # Recursive late-reverb estimate: decayed history + delayed observation.
        for t in range(2, mag.shape[1]):
            late[:, t] = np.maximum(
                late[:, t - 1] * 0.94,
                mag[:, t - 2] * 0.86,
            )

        # 动态抑制系数：高能量段保守（0.65），低能量段激进（0.82）
        suppress_coef = 0.65 + 0.17 * (1.0 - vocal_strength)
        direct = np.maximum(mag - suppress_coef[np.newaxis, :] * late, 0.0)

        # Dynamic floor: pure-echo frames get floor≈0, direct-voice frames keep more
        echo_ratio = np.clip(late / (mag + eps), 0.0, 1.0)
        # 高能量段保留更多原始信号（floor系数0.22），低能量段少保留（0.12）
        floor_coef = 0.12 + 0.10 * vocal_strength
        floor = (1.0 - echo_ratio) * floor_coef[np.newaxis, :] * mag
        direct = np.maximum(direct, floor)

        # Smooth in time to avoid musical noise.
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        kernel /= np.sum(kernel)
        direct = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"),
            axis=1,
            arr=direct,
        )
        direct = np.clip(direct, 0.0, mag + eps)

        # Dynamic dry blend: 高能量段混合更多原始信号（0.30），低能量段少混合（0.10）
        frame_echo = np.mean(echo_ratio, axis=0, keepdims=True)  # [1, T]
        blend = (1.0 - frame_echo) * (0.10 + 0.20 * vocal_strength[np.newaxis, :])
        out_spec = direct * phase
        dry_spec = mag * phase
        blended_spec = (1.0 - blend) * out_spec + blend * dry_spec
        out = librosa.istft(blended_spec, hop_length=hop, win_length=win, length=len(x)).astype(np.float32)

        out = soft_clip(out, threshold=0.9, ceiling=0.99)
        return out.astype(np.float32)

    @staticmethod
    def _compute_echo_tail_sample_gain(
        original: np.ndarray,
        dereverbed: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, int, int]:
        """根据 original 与 dereverbed 的差异估计回声尾段抑制增益。"""
        import librosa

        if original.size == 0 or dereverbed.size == 0:
            return np.ones_like(dereverbed, dtype=np.float32), 0, 0

        frame_length = 2048
        hop_length = 512
        orig_rms = librosa.feature.rms(
            y=original, frame_length=frame_length, hop_length=hop_length, center=True
        )[0]
        derev_rms = librosa.feature.rms(
            y=dereverbed, frame_length=frame_length, hop_length=hop_length, center=True
        )[0]

        eps = 1e-8
        orig_rms_db = 20.0 * np.log10(orig_rms + eps)
        derev_rms_db = 20.0 * np.log10(derev_rms + eps)
        ref_db = float(np.percentile(orig_rms_db, 95))
        derev_ref_db = float(np.percentile(derev_rms_db, 95))

        attenuation_ratio = derev_rms / (orig_rms + eps)

        direct_activity = np.clip((derev_rms_db - (derev_ref_db - 30.0)) / 18.0, 0.0, 1.0)
        direct_activity = CoverPipeline._hold_activity_curve(
            direct_activity,
            max(1, int(0.20 * sr / hop_length)),
        )

        removed_strength = np.clip((0.62 - attenuation_ratio) / 0.48, 0.0, 1.0)
        direct_absence = np.clip((0.30 - direct_activity) / 0.30, 0.0, 1.0)
        quiet_tail_strength = (
            np.clip(((ref_db - 36.0) - orig_rms_db) / 12.0, 0.0, 1.0)
            * removed_strength
            * direct_absence
        )
        loud_echo_strength = (
            np.clip((orig_rms_db - (ref_db - 30.0)) / 18.0, 0.0, 1.0)
            * removed_strength
            * direct_absence
        )
        echo_strength = np.maximum(quiet_tail_strength, loud_echo_strength)

        # Enforce minimum duration of 100ms
        min_frames = max(1, int(0.1 * sr / hop_length))
        # Dilate: only keep runs >= min_frames
        gate = (echo_strength > 0.20).astype(np.float32)
        # Simple run-length filter
        filtered = np.zeros_like(gate)
        run_start = 0
        in_run = False
        for i in range(len(gate)):
            if gate[i] > 0.5:
                if not in_run:
                    run_start = i
                    in_run = True
            else:
                if in_run:
                    if (i - run_start) >= min_frames:
                        filtered[run_start:i] = 1.0
                    in_run = False
        if in_run and (len(gate) - run_start) >= min_frames:
            filtered[run_start:len(gate)] = 1.0

        # 50ms sigmoid transition
        transition_frames = max(1, int(0.05 * sr / hop_length))
        kernel = np.ones(transition_frames, dtype=np.float32) / transition_frames
        filtered = np.convolve(filtered * echo_strength, kernel, mode="same")
        filtered = np.clip(filtered, 0.0, 1.0)

        # Apply: gated frames attenuated to 0.18x，保留更多尾音避免不自然断裂
        gain_curve = 1.0 - filtered * 0.82  # 1.0 for normal, 0.18 for gated

        # Expand frame-level gain to sample-level
        sample_gain = CoverPipeline._frame_curve_to_sample_gain(
            gain_curve,
            len(dereverbed),
            hop_length,
        )

        gated_count = int(np.sum(filtered > 0.5))
        return sample_gain.astype(np.float32), gated_count, len(filtered)

    @staticmethod
    def _fit_frame_curve(curve: np.ndarray, target_len: int) -> np.ndarray:
        """Pad/truncate frame curves to the target frame count."""
        curve = np.asarray(curve, dtype=np.float32).reshape(-1)
        if target_len <= 0:
            return np.zeros(0, dtype=np.float32)
        if curve.size == target_len:
            return curve
        if curve.size == 0:
            return np.zeros(target_len, dtype=np.float32)
        if curve.size > target_len:
            return curve[:target_len].astype(np.float32)
        pad_width = target_len - curve.size
        return np.pad(curve, (0, pad_width), mode="edge").astype(np.float32)

    @staticmethod
    def _hold_activity_curve(curve: np.ndarray, hold_frames: int) -> np.ndarray:
        """Keep recent vocal activity for a short trailing window."""
        curve = np.asarray(curve, dtype=np.float32).reshape(-1)
        if curve.size == 0:
            return curve

        hold_frames = max(1, int(hold_frames))
        if hold_frames <= 1:
            return curve.astype(np.float32)

        held = np.empty_like(curve, dtype=np.float32)
        window = []
        for index, value in enumerate(curve):
            while window and window[-1][1] <= value:
                window.pop()
            window.append((index, float(value)))
            min_index = index - hold_frames + 1
            while window and window[0][0] < min_index:
                window.pop(0)
            held[index] = window[0][1] if window else float(value)
        return held.astype(np.float32)

    @staticmethod
    def _frame_curve_to_sample_gain(
        frame_curve: np.ndarray,
        n_samples: int,
        hop_length: int,
    ) -> np.ndarray:
        """Interpolate frame-domain gains to sample-domain gains."""
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        frame_curve = np.asarray(frame_curve, dtype=np.float32).reshape(-1)
        if frame_curve.size == 0:
            return np.ones(n_samples, dtype=np.float32)

        sample_indices = np.arange(n_samples, dtype=np.float32)
        frame_indices = np.clip(sample_indices / float(hop_length), 0, frame_curve.size - 1)
        return np.interp(
            frame_indices,
            np.arange(frame_curve.size, dtype=np.float32),
            frame_curve,
        ).astype(np.float32)


    @staticmethod
    def _compute_activity_sample_weights(
        reference_audio: np.ndarray,
        sr: int,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Build sample-domain weights from active vocal regions only."""
        import librosa

        reference_audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
        if reference_audio.size == 0:
            return np.zeros(0, dtype=np.float32)

        eps = 1e-8
        frame_rms = librosa.feature.rms(
            y=reference_audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        frame_rms = np.asarray(frame_rms, dtype=np.float32)
        frame_db = 20.0 * np.log10(frame_rms + eps)
        ref_db = float(np.percentile(frame_db, 95))

        activity = np.clip((frame_db - (ref_db - 30.0)) / 18.0, 0.0, 1.0)
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        kernel /= np.sum(kernel)
        activity = np.convolve(activity, kernel, mode="same")
        activity = CoverPipeline._hold_activity_curve(
            activity,
            max(1, int(0.24 * sr / hop_length)),
        )
        frame_weights = np.clip(activity * activity, 0.0, 1.0)
        return CoverPipeline._frame_curve_to_sample_gain(
            frame_weights,
            len(reference_audio),
            hop_length,
        )

    @staticmethod
    def _weighted_rms(audio: np.ndarray, weights: np.ndarray) -> float:
        """Compute RMS under sample-domain weights."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if audio.size == 0 or weights.size == 0:
            return 0.0

        aligned_len = min(audio.size, weights.size)
        if aligned_len <= 0:
            return 0.0

        audio = audio[:aligned_len]
        weights = np.clip(weights[:aligned_len], 0.0, 1.0)
        total = float(np.sum(weights))
        if total <= 1e-6:
            return 0.0
        return float(np.sqrt(np.sum((audio * audio) * weights) / total + 1e-12))

    def _apply_source_gap_suppression(
        self,
        source_vocals_path: str,
        converted_vocals_path: str,
    ) -> None:
        """Suppress hallucinated noise in sustained no-vocal gaps only."""
        import librosa
        import soundfile as sf

        source_audio, source_sr = librosa.load(source_vocals_path, sr=None, mono=True)
        converted_audio, converted_sr = sf.read(converted_vocals_path)
        if converted_audio.ndim > 1:
            converted_audio = converted_audio.mean(axis=1)
        source_audio = np.asarray(source_audio, dtype=np.float32)
        converted_audio = np.asarray(converted_audio, dtype=np.float32)

        if source_sr != converted_sr:
            source_audio = librosa.resample(
                source_audio,
                orig_sr=source_sr,
                target_sr=converted_sr,
            ).astype(np.float32)

        aligned_len = min(len(source_audio), len(converted_audio))
        if aligned_len <= 0:
            return

        source_audio = source_audio[:aligned_len]
        converted_main = converted_audio[:aligned_len]
        gain, gated_frames, total_frames = self._compute_quiet_gap_sample_gain(
            source_audio,
            converted_sr,
        )
        gain = np.clip(gain[:aligned_len], 0.0, 1.0).astype(np.float32)
        suppressed = converted_main * gain

        attenuated_samples = int(np.sum(gain < 0.50))
        if attenuated_samples > 0:
            log.detail(
                f"Source gap suppression: attenuated {attenuated_samples}/{aligned_len} samples in no-vocal regions"
            )
        if gated_frames > 0:
            log.detail(
                f"Source gap suppression: detected {gated_frames}/{total_frames} sustained quiet frames"
            )

        if len(converted_audio) > aligned_len:
            tail = converted_audio[aligned_len:] * 0.0
            converted_audio = np.concatenate([suppressed, tail.astype(np.float32)])
        else:
            converted_audio = suppressed

        sf.write(converted_vocals_path, converted_audio.astype(np.float32), converted_sr)

    def _apply_source_breath_cleanup(
        self,
        source_vocals_path: str,
        converted_vocals_path: str,
    ) -> None:
        """Blend dry source breaths back where converted low-energy regions sound too synthetic."""
        import librosa
        import soundfile as sf

        def _preemphasis_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            residual = np.empty_like(audio)
            residual[0] = audio[0]
            residual[1:] = audio[1:] - 0.97 * audio[:-1]
            return librosa.feature.rms(
                y=residual,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0]

        def _spectral_flatness(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            return librosa.feature.spectral_flatness(
                y=audio + 1e-8,
                n_fft=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        source_audio, source_sr = librosa.load(source_vocals_path, sr=None, mono=True)
        converted_audio, converted_sr = sf.read(converted_vocals_path)
        if converted_audio.ndim > 1:
            converted_audio = converted_audio.mean(axis=1)
        source_audio = np.asarray(source_audio, dtype=np.float32)
        converted_audio = np.asarray(converted_audio, dtype=np.float32)

        if source_sr != converted_sr:
            source_audio = librosa.resample(
                source_audio,
                orig_sr=source_sr,
                target_sr=converted_sr,
            ).astype(np.float32)

        aligned_len = min(len(source_audio), len(converted_audio))
        if aligned_len <= 0:
            return

        source_main = source_audio[:aligned_len]
        converted_main = converted_audio[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8

        source_rms = librosa.feature.rms(
            y=source_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        converted_rms = librosa.feature.rms(
            y=converted_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        source_hf = _preemphasis_rms(source_main, frame_length, hop_length)
        converted_hf = _preemphasis_rms(converted_main, frame_length, hop_length)
        source_flatness = _spectral_flatness(source_main, frame_length, hop_length)
        converted_flatness = _spectral_flatness(converted_main, frame_length, hop_length)

        frame_count = min(
            source_rms.size,
            converted_rms.size,
            source_hf.size,
            converted_hf.size,
            source_flatness.size,
            converted_flatness.size,
        )
        if frame_count <= 0:
            return

        source_rms = source_rms[:frame_count].astype(np.float32)
        converted_rms = converted_rms[:frame_count].astype(np.float32)
        source_hf = source_hf[:frame_count].astype(np.float32)
        converted_hf = converted_hf[:frame_count].astype(np.float32)
        source_flatness = source_flatness[:frame_count].astype(np.float32)
        converted_flatness = converted_flatness[:frame_count].astype(np.float32)

        source_db = 20.0 * np.log10(source_rms + eps)
        ref_db = float(np.percentile(source_db, 95))

        low_mid_energy = np.clip((ref_db - 18.0 - source_db) / 22.0, 0.0, 1.0)
        not_sustained_gap = 1.0 - np.clip((ref_db - 42.0 - source_db) / 8.0, 0.0, 1.0)
        breath_like = (
            not_sustained_gap
            * np.clip((source_flatness - 0.22) / 0.22, 0.0, 1.0)
            * np.clip(source_hf / (float(np.percentile(source_hf, 72)) + eps), 0.0, 1.25)
        )
        excess_rms = np.clip(
            (converted_rms - 1.15 * source_rms) / (converted_rms + eps),
            0.0,
            1.0,
        )
        excess_hf = np.clip(
            (converted_hf - 1.10 * source_hf) / (converted_hf + eps),
            0.0,
            1.0,
        )
        tonalized_breath = breath_like * np.clip(
            (source_flatness - converted_flatness) / 0.18,
            0.0,
            1.0,
        )
        hf_ratio = converted_hf / (source_hf + eps)
        low_energy_tonal_hf = (
            np.sqrt(low_mid_energy)
            * not_sustained_gap
            * excess_hf
            * np.clip((hf_ratio - 1.02) / 1.80, 0.0, 1.0)
        )

        blend_curve = np.maximum(
            np.maximum(
                low_mid_energy * not_sustained_gap * np.maximum(excess_rms, excess_hf),
                1.35 * low_energy_tonal_hf,
            ),
            0.88
            * tonalized_breath
            * np.maximum(
                np.clip((converted_rms - 0.92 * source_rms) / (converted_rms + eps), 0.0, 1.0),
                np.clip((converted_hf - 0.95 * source_hf) / (converted_hf + eps), 0.0, 1.0),
            ),
        )
        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)
        blend_curve = np.convolve(blend_curve, smooth_kernel, mode="same")
        blend_curve = self._hold_activity_curve(
            blend_curve,
            max(1, int(0.05 * converted_sr / hop_length)),
        )
        blend_curve = np.clip(blend_curve, 0.0, 1.0).astype(np.float32)

        blended_frames = int(np.sum(blend_curve > 0.20))
        if blended_frames <= 0:
            return
        mechanical_frames = int(np.sum(tonalized_breath > 0.20))

        sample_blend = 0.82 * self._frame_curve_to_sample_gain(
            blend_curve,
            aligned_len,
            hop_length,
        )
        sample_blend = np.clip(sample_blend[:aligned_len], 0.0, 0.82).astype(np.float32)
        blended_main = (
            converted_main * (1.0 - sample_blend)
            + source_main * sample_blend
        ).astype(np.float32)

        if len(converted_audio) > aligned_len:
            converted_audio = np.concatenate([blended_main, converted_audio[aligned_len:]])
        else:
            converted_audio = blended_main

        sf.write(converted_vocals_path, converted_audio.astype(np.float32), converted_sr)
        log.detail(
            "Source breath cleanup: "
            f"blended {blended_frames}/{frame_count} low-mid-energy frames toward dry source "
            f"(mechanical={mechanical_frames})"
        )

    def _apply_source_transition_cleanup(
        self,
        source_vocals_path: str,
        converted_vocals_path: str,
    ) -> None:
        """Blend dry source into glitch-prone breaths and transition spikes."""
        import librosa
        import soundfile as sf

        def _preemphasis_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            residual = np.empty_like(audio)
            residual[0] = audio[0]
            residual[1:] = audio[1:] - 0.97 * audio[:-1]
            return librosa.feature.rms(
                y=residual,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        source_audio, source_sr = librosa.load(source_vocals_path, sr=None, mono=True)
        converted_audio, converted_sr = sf.read(converted_vocals_path)
        if converted_audio.ndim > 1:
            converted_audio = converted_audio.mean(axis=1)
        source_audio = np.asarray(source_audio, dtype=np.float32)
        converted_audio = np.asarray(converted_audio, dtype=np.float32)

        if source_sr != converted_sr:
            source_audio = librosa.resample(
                source_audio,
                orig_sr=source_sr,
                target_sr=converted_sr,
            ).astype(np.float32)

        aligned_len = min(len(source_audio), len(converted_audio))
        if aligned_len <= 0:
            return

        source_main = source_audio[:aligned_len]
        converted_main = converted_audio[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8

        source_rms = librosa.feature.rms(
            y=source_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        converted_rms = librosa.feature.rms(
            y=converted_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        source_hf = _preemphasis_rms(source_main, frame_length, hop_length)
        converted_hf = _preemphasis_rms(converted_main, frame_length, hop_length)

        frame_count = min(
            source_rms.size,
            converted_rms.size,
            source_hf.size,
            converted_hf.size,
        )
        if frame_count <= 4:
            return

        source_rms = source_rms[:frame_count]
        converted_rms = converted_rms[:frame_count]
        source_hf = source_hf[:frame_count]
        converted_hf = converted_hf[:frame_count]

        source_db = 20.0 * np.log10(source_rms + eps)
        ref_db = float(np.percentile(source_db, 95))
        low_mid_energy = np.clip((ref_db - 18.0 - source_db) / 22.0, 0.0, 1.0)
        not_sustained_gap = 1.0 - np.clip((ref_db - 42.0 - source_db) / 8.0, 0.0, 1.0)
        excess_rms = np.clip(
            (converted_rms - 1.10 * source_rms) / (converted_rms + eps),
            0.0,
            1.0,
        )
        excess_hf = np.clip(
            (converted_hf - 1.07 * source_hf) / (converted_hf + eps),
            0.0,
            1.0,
        )

        breath_curve = low_mid_energy * not_sustained_gap * np.maximum(excess_rms, excess_hf)
        cand_delta = np.abs(np.diff(converted_rms, prepend=converted_rms[:1]))
        ref_delta = np.abs(np.diff(source_rms, prepend=source_rms[:1]))
        spike_excess = np.clip(
            (cand_delta - (0.007 + 1.35 * ref_delta)) / (cand_delta + eps),
            0.0,
            1.0,
        )
        activity = np.clip((source_db - (ref_db - 28.0)) / 14.0, 0.0, 1.0)
        transition_curve = spike_excess * np.maximum(activity, 0.35 * low_mid_energy)
        quiet_curve = np.clip((ref_db - 35.0 - source_db) / 12.0, 0.0, 1.0) * np.maximum(
            excess_rms,
            excess_hf,
        )
        hf_ratio = converted_hf / (source_hf + eps)
        tonal_hf_curve = (
            np.sqrt(low_mid_energy)
            * not_sustained_gap
            * excess_hf
            * np.clip((hf_ratio - 1.02) / 1.80, 0.0, 1.0)
        )
        overshoot_energy = np.maximum(
            np.clip((converted_rms - 1.09 * source_rms) / (converted_rms + eps), 0.0, 1.0),
            np.clip((converted_hf - 1.07 * source_hf) / (converted_hf + eps), 0.0, 1.0),
        )
        overshoot_focus = np.clip(
            np.maximum(
                1.15 * spike_excess,
                0.55 * low_mid_energy * np.maximum(excess_rms, excess_hf),
            ),
            0.0,
            1.0,
        )
        overshoot_curve = activity * overshoot_energy * overshoot_focus

        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)
        for _ in range(2):
            breath_curve = np.convolve(breath_curve, smooth_kernel, mode="same")
            transition_curve = np.convolve(transition_curve, smooth_kernel, mode="same")
            quiet_curve = np.convolve(quiet_curve, smooth_kernel, mode="same")
            tonal_hf_curve = np.convolve(tonal_hf_curve, smooth_kernel, mode="same")
            overshoot_curve = np.convolve(overshoot_curve, smooth_kernel, mode="same")

        breath_curve = self._hold_activity_curve(
            breath_curve,
            max(1, int(0.07 * converted_sr / hop_length)),
        )
        transition_curve = self._hold_activity_curve(
            transition_curve,
            max(1, int(0.07 * converted_sr / hop_length)),
        )
        quiet_curve = self._hold_activity_curve(
            quiet_curve,
            max(1, int(0.08 * converted_sr / hop_length)),
        )
        tonal_hf_curve = self._hold_activity_curve(
            tonal_hf_curve,
            max(1, int(0.06 * converted_sr / hop_length)),
        )
        overshoot_curve = self._hold_activity_curve(
            overshoot_curve,
            max(1, int(0.04 * converted_sr / hop_length)),
        )

        blend_curve = np.maximum.reduce(
            [
                0.72 * np.clip(breath_curve, 0.0, 1.0),
                0.74 * np.clip(transition_curve, 0.0, 1.0),
                0.50 * np.clip(quiet_curve, 0.0, 1.0),
                0.76 * np.clip(tonal_hf_curve, 0.0, 1.0),
                0.40 * np.clip(overshoot_curve, 0.0, 1.0),
            ]
        )
        blend_curve = np.clip(blend_curve, 0.0, 0.88).astype(np.float32)

        blended_frames = int(np.sum(blend_curve > 0.20))
        if blended_frames <= 0:
            return

        sample_blend = self._frame_curve_to_sample_gain(
            blend_curve,
            aligned_len,
            hop_length,
        )[:aligned_len]
        repaired_main = (
            converted_main * (1.0 - sample_blend)
            + source_main * sample_blend
        ).astype(np.float32)
        trim_curve = np.clip(
            np.maximum(
                0.72 * overshoot_curve,
                0.55 * transition_curve,
                0.50 * tonal_hf_curve,
            ),
            0.0,
            1.0,
        ).astype(np.float32)
        trim_weights = self._frame_curve_to_sample_gain(
            trim_curve,
            aligned_len,
            hop_length,
        )[:aligned_len]
        trim_weights = (
            trim_weights
            * self._compute_activity_sample_weights(source_main, converted_sr)[:aligned_len]
        ).astype(np.float32)
        trim_weights = np.clip(trim_weights, 0.0, 1.0)

        ref_active_rms = self._weighted_rms(source_main, trim_weights)
        out_active_rms = self._weighted_rms(repaired_main, trim_weights)
        gain = 1.0
        if (
            float(np.sum(trim_weights)) > 1e-3
            and ref_active_rms > 1e-6
            and out_active_rms > ref_active_rms * 1.05
        ):
            gain = float(
                np.clip(
                    (1.04 * ref_active_rms) / (out_active_rms + eps),
                    0.92,
                    1.0,
                )
            )
            if gain < 0.999:
                repaired_main = self._apply_weighted_gain(
                    repaired_main,
                    trim_weights,
                    gain,
                ).astype(np.float32)

        if len(converted_audio) > aligned_len:
            converted_audio = np.concatenate([repaired_main, converted_audio[aligned_len:]])
        else:
            converted_audio = repaired_main

        sf.write(converted_vocals_path, converted_audio.astype(np.float32), converted_sr)
        log.detail(
            "Source transition cleanup: "
            f"blended {blended_frames}/{frame_count} frames "
            f"(breath={int(np.sum(breath_curve > 0.20))}, "
            f"spike={int(np.sum(transition_curve > 0.20))}, "
            f"quiet={int(np.sum(quiet_curve > 0.20))}, "
            f"overshoot={int(np.sum(overshoot_curve > 0.20))})"
        )
        if gain < 0.999:
            log.detail(
                "Source transition cleanup: "
                f"hotspot RMS trim ref={ref_active_rms:.6f}, "
                f"out={out_active_rms:.6f}, gain={gain:.3f}"
            )

    def _restore_active_vocal_loudness(
        self,
        reference_vocals_path: str,
        converted_vocals_path: str,
        target_ratio: float = 0.86,
        max_gain: float = 1.85,
    ) -> None:
        """Restore vocal body loudness without boosting breaths and unstable frames."""
        import librosa
        import soundfile as sf

        def _preemphasis_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            residual = np.empty_like(audio)
            residual[0] = audio[0]
            residual[1:] = audio[1:] - 0.97 * audio[:-1]
            return librosa.feature.rms(
                y=residual,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        reference_audio, reference_sr = librosa.load(reference_vocals_path, sr=None, mono=True)
        converted_audio, converted_sr = sf.read(converted_vocals_path)
        if converted_audio.ndim > 1:
            converted_audio = converted_audio.mean(axis=1)
        reference_audio = np.asarray(reference_audio, dtype=np.float32)
        converted_audio = np.asarray(converted_audio, dtype=np.float32)

        if reference_sr != converted_sr:
            reference_audio = librosa.resample(
                reference_audio,
                orig_sr=reference_sr,
                target_sr=converted_sr,
            ).astype(np.float32)

        aligned_len = min(len(reference_audio), len(converted_audio))
        if aligned_len <= 0:
            return

        reference_main = reference_audio[:aligned_len]
        converted_main = converted_audio[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8

        reference_rms = librosa.feature.rms(
            y=reference_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        converted_rms = librosa.feature.rms(
            y=converted_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        reference_hf = _preemphasis_rms(reference_main, frame_length, hop_length)
        converted_hf = _preemphasis_rms(converted_main, frame_length, hop_length)

        frame_count = min(
            reference_rms.size,
            converted_rms.size,
            reference_hf.size,
            converted_hf.size,
        )
        if frame_count <= 4:
            return

        reference_rms = reference_rms[:frame_count]
        converted_rms = converted_rms[:frame_count]
        reference_hf = reference_hf[:frame_count]
        converted_hf = converted_hf[:frame_count]

        reference_db = 20.0 * np.log10(reference_rms + eps)
        ref_db = float(np.percentile(reference_db, 95))
        body_curve = np.square(np.clip((reference_db - (ref_db - 18.0)) / 9.0, 0.0, 1.0))
        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)
        body_curve = np.convolve(body_curve, smooth_kernel, mode="same")
        body_curve = self._hold_activity_curve(
            body_curve,
            max(1, int(0.08 * converted_sr / hop_length)),
        )
        under_target = np.clip(
            (0.95 * reference_rms - converted_rms) / (0.95 * reference_rms + eps),
            0.0,
            1.0,
        )
        artifact_guard = 1.0 - np.maximum(
            np.clip((converted_hf - 1.12 * reference_hf) / (converted_hf + eps), 0.0, 1.0),
            np.clip((converted_rms - 1.10 * reference_rms) / (converted_rms + eps), 0.0, 1.0),
        )
        boost_curve = np.clip(body_curve * under_target * artifact_guard, 0.0, 1.0).astype(np.float32)

        boosted_frames = int(np.sum(boost_curve > 0.20))
        if boosted_frames <= 0:
            return

        sample_weights = self._frame_curve_to_sample_gain(
            boost_curve,
            aligned_len,
            hop_length,
        )[:aligned_len]
        ref_body_rms = self._weighted_rms(reference_main, sample_weights)
        out_body_rms = self._weighted_rms(converted_main, sample_weights)
        if ref_body_rms <= 1e-6 or out_body_rms <= 1e-6:
            return

        target_body_rms = target_ratio * ref_body_rms
        if out_body_rms >= target_body_rms * 0.98:
            return

        gain = float(np.clip(target_body_rms / (out_body_rms + eps), 1.0, max_gain))
        if gain <= 1.001:
            return

        restored_main = self._apply_weighted_gain(
            converted_main,
            sample_weights,
            gain,
        ).astype(np.float32)
        restored_main = soft_clip(restored_main, threshold=0.92, ceiling=0.985)

        if len(converted_audio) > aligned_len:
            converted_audio = np.concatenate([restored_main, converted_audio[aligned_len:]])
        else:
            converted_audio = restored_main

        sf.write(converted_vocals_path, converted_audio.astype(np.float32), converted_sr)
        log.detail(
            "Source loudness restore: "
            f"boosted {boosted_frames}/{frame_count} body frames, "
            f"ref={ref_body_rms:.6f}, out={out_body_rms:.6f}, gain={gain:.3f}"
        )

    def _restore_voiced_body_from_raw(
        self,
        raw_vocals_path: str,
        source_vocals_path: str,
        converted_vocals_path: str,
        max_blend: float = 0.18,
    ) -> None:
        """Restore target timbre body from raw VC only on stable voiced phrases."""
        import librosa
        import soundfile as sf

        def _preemphasis_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            residual = np.empty_like(audio)
            residual[0] = audio[0]
            residual[1:] = audio[1:] - 0.97 * audio[:-1]
            return librosa.feature.rms(
                y=residual,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        def _spectral_flatness(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            return librosa.feature.spectral_flatness(
                y=audio + 1e-8,
                n_fft=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        raw_audio, raw_sr = sf.read(raw_vocals_path)
        source_audio, source_sr = librosa.load(source_vocals_path, sr=None, mono=True)
        converted_audio, converted_sr = sf.read(converted_vocals_path)

        if raw_audio.ndim > 1:
            raw_audio = raw_audio.mean(axis=1)
        if converted_audio.ndim > 1:
            converted_audio = converted_audio.mean(axis=1)

        raw_audio = np.asarray(raw_audio, dtype=np.float32)
        source_audio = np.asarray(source_audio, dtype=np.float32)
        converted_audio = np.asarray(converted_audio, dtype=np.float32)

        if raw_sr != converted_sr:
            raw_audio = librosa.resample(
                raw_audio,
                orig_sr=raw_sr,
                target_sr=converted_sr,
            ).astype(np.float32)
        if source_sr != converted_sr:
            source_audio = librosa.resample(
                source_audio,
                orig_sr=source_sr,
                target_sr=converted_sr,
            ).astype(np.float32)

        aligned_len = min(len(raw_audio), len(source_audio), len(converted_audio))
        if aligned_len <= 0:
            return

        raw_main = raw_audio[:aligned_len]
        source_main = source_audio[:aligned_len]
        converted_main = converted_audio[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8

        raw_rms = librosa.feature.rms(
            y=raw_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        source_rms = librosa.feature.rms(
            y=source_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        converted_rms = librosa.feature.rms(
            y=converted_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        raw_hf = _preemphasis_rms(raw_main, frame_length, hop_length)
        source_hf = _preemphasis_rms(source_main, frame_length, hop_length)
        converted_hf = _preemphasis_rms(converted_main, frame_length, hop_length)
        raw_flatness = _spectral_flatness(raw_main, frame_length, hop_length)
        source_flatness = _spectral_flatness(source_main, frame_length, hop_length)
        converted_flatness = _spectral_flatness(converted_main, frame_length, hop_length)

        frame_count = min(
            raw_rms.size,
            source_rms.size,
            converted_rms.size,
            raw_hf.size,
            source_hf.size,
            converted_hf.size,
            raw_flatness.size,
            source_flatness.size,
            converted_flatness.size,
        )
        if frame_count <= 4:
            return

        raw_rms = raw_rms[:frame_count]
        source_rms = source_rms[:frame_count]
        converted_rms = converted_rms[:frame_count]
        raw_hf = raw_hf[:frame_count]
        source_hf = source_hf[:frame_count]
        converted_hf = converted_hf[:frame_count]
        raw_flatness = raw_flatness[:frame_count]
        source_flatness = source_flatness[:frame_count]
        converted_flatness = converted_flatness[:frame_count]

        source_db = 20.0 * np.log10(source_rms + eps)
        ref_db = float(np.percentile(source_db, 95))
        activity = np.square(np.clip((source_db - (ref_db - 20.0)) / 10.0, 0.0, 1.0))
        low_mid_energy = np.clip((ref_db - 18.0 - source_db) / 22.0, 0.0, 1.0)
        breath_like = (
            low_mid_energy
            * np.clip((source_flatness - 0.22) / 0.22, 0.0, 1.0)
            * np.clip(source_hf / (float(np.percentile(source_hf, 72)) + eps), 0.0, 1.25)
        )

        body_need = np.maximum(
            np.clip((0.93 * source_rms - converted_rms) / (0.93 * source_rms + eps), 0.0, 1.0),
            np.clip((0.90 * source_hf - converted_hf) / (0.90 * source_hf + eps), 0.0, 1.0),
        )
        source_delta = np.abs(np.diff(source_rms, prepend=source_rms[:1]))
        raw_delta = np.abs(np.diff(raw_rms, prepend=raw_rms[:1]))
        converted_delta = np.abs(np.diff(converted_rms, prepend=converted_rms[:1]))
        glitch_curve = np.clip(
            (converted_delta - (0.008 + 1.40 * source_delta)) / (converted_delta + eps),
            0.0,
            1.0,
        )
        body_gap = np.maximum(
            np.clip((0.90 * raw_rms - converted_rms) / (0.90 * raw_rms + eps), 0.0, 1.0),
            np.clip((0.84 * raw_hf - converted_hf) / (0.84 * raw_hf + eps), 0.0, 1.0),
        )
        raw_overshoot = np.maximum(
            np.clip((raw_rms - 1.18 * source_rms) / (raw_rms + eps), 0.0, 1.0),
            np.clip((raw_hf - 1.18 * source_hf) / (raw_hf + eps), 0.0, 1.0),
        )
        raw_roughness = np.maximum.reduce(
            [
                np.clip((raw_flatness - source_flatness - 0.015) / 0.085, 0.0, 1.0),
                np.clip((raw_hf - 1.08 * source_hf) / (raw_hf + eps), 0.0, 1.0),
                np.clip((raw_delta - (0.006 + 1.18 * source_delta)) / (raw_delta + eps), 0.0, 1.0),
                np.clip((raw_flatness - converted_flatness - 0.020) / 0.080, 0.0, 1.0),
            ]
        )
        stable_curve = (
            activity
            * (1.0 - 0.90 * breath_like)
            * (1.0 - 0.75 * glitch_curve)
            * (1.0 - 0.65 * raw_overshoot)
            * (1.0 - 0.82 * raw_roughness)
        )
        blend_curve = np.clip(stable_curve * body_gap * body_need, 0.0, 1.0).astype(np.float32)

        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)
        for _ in range(2):
            blend_curve = np.convolve(blend_curve, smooth_kernel, mode="same")
        blend_curve = self._hold_activity_curve(
            blend_curve,
            max(1, int(0.06 * converted_sr / hop_length)),
        )
        blend_curve = np.clip(blend_curve, 0.0, 1.0).astype(np.float32)

        need_curve = np.clip(activity * body_need * (1.0 - 0.75 * raw_roughness), 0.0, 1.0).astype(np.float32)
        if float(np.mean(need_curve)) < 0.035:
            return

        restored_frames = int(np.sum(blend_curve > 0.20))
        if restored_frames <= 0:
            return

        sample_blend = max_blend * self._frame_curve_to_sample_gain(
            blend_curve,
            aligned_len,
            hop_length,
        )[:aligned_len]
        sample_blend = np.clip(sample_blend, 0.0, max_blend).astype(np.float32)
        restored_main = (
            converted_main * (1.0 - sample_blend)
            + raw_main * sample_blend
        ).astype(np.float32)

        trim_weights = self._frame_curve_to_sample_gain(
            np.clip(stable_curve, 0.0, 1.0),
            aligned_len,
            hop_length,
        )[:aligned_len]
        ref_body_rms = self._weighted_rms(raw_main, trim_weights)
        out_body_rms = self._weighted_rms(restored_main, trim_weights)
        gain = 1.0
        if ref_body_rms > 1e-6 and out_body_rms > ref_body_rms * 0.99:
            gain = float(np.clip((0.99 * ref_body_rms) / (out_body_rms + eps), 0.92, 1.0))
            if gain < 0.999:
                restored_main = self._apply_weighted_gain(
                    restored_main,
                    trim_weights,
                    gain,
                ).astype(np.float32)

        restored_main = soft_clip(restored_main, threshold=0.92, ceiling=0.985)

        if len(converted_audio) > aligned_len:
            converted_audio = np.concatenate([restored_main, converted_audio[aligned_len:]])
        else:
            converted_audio = restored_main

        sf.write(converted_vocals_path, converted_audio.astype(np.float32), converted_sr)
        log.detail(
            "Raw body restore: "
            f"blended {restored_frames}/{frame_count} stable voiced frames from raw VC "
            f"(max_blend={max_blend:.2f}, gain={gain:.3f}, "
            f"rough_frames={int(np.sum(raw_roughness > 0.20))})"
        )

    def _apply_artifact_segment_rescue(
        self,
        source_vocals_path: str,
        converted_vocals_path: str,
        max_source_blend: float = 0.86,
    ) -> None:
        """Fallback toward dry source only on short segments where final VC still sounds unstable."""
        import librosa
        import soundfile as sf

        def _preemphasis_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            residual = np.empty_like(audio)
            residual[0] = audio[0]
            residual[1:] = audio[1:] - 0.97 * audio[:-1]
            return librosa.feature.rms(
                y=residual,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        def _spectral_flatness(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            return librosa.feature.spectral_flatness(
                y=audio + 1e-8,
                n_fft=frame_length,
                hop_length=hop_length,
                center=True,
            )[0].astype(np.float32)

        source_audio, source_sr = librosa.load(source_vocals_path, sr=None, mono=True)
        converted_audio, converted_sr = sf.read(converted_vocals_path)
        if converted_audio.ndim > 1:
            converted_audio = converted_audio.mean(axis=1)

        source_audio = np.asarray(source_audio, dtype=np.float32)
        converted_audio = np.asarray(converted_audio, dtype=np.float32)

        if source_sr != converted_sr:
            source_audio = librosa.resample(
                source_audio,
                orig_sr=source_sr,
                target_sr=converted_sr,
            ).astype(np.float32)

        aligned_len = min(len(source_audio), len(converted_audio))
        if aligned_len <= 0:
            return

        source_main = source_audio[:aligned_len]
        converted_main = converted_audio[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8

        source_rms = librosa.feature.rms(
            y=source_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        converted_rms = librosa.feature.rms(
            y=converted_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        source_hf = _preemphasis_rms(source_main, frame_length, hop_length)
        converted_hf = _preemphasis_rms(converted_main, frame_length, hop_length)
        source_flatness = _spectral_flatness(source_main, frame_length, hop_length)
        converted_flatness = _spectral_flatness(converted_main, frame_length, hop_length)

        frame_count = min(
            source_rms.size,
            converted_rms.size,
            source_hf.size,
            converted_hf.size,
            source_flatness.size,
            converted_flatness.size,
        )
        if frame_count <= 4:
            return

        source_rms = source_rms[:frame_count]
        converted_rms = converted_rms[:frame_count]
        source_hf = source_hf[:frame_count]
        converted_hf = converted_hf[:frame_count]
        source_flatness = source_flatness[:frame_count]
        converted_flatness = converted_flatness[:frame_count]

        source_db = 20.0 * np.log10(source_rms + eps)
        ref_db = float(np.percentile(source_db, 95))
        activity = np.square(np.clip((source_db - (ref_db - 24.0)) / 11.0, 0.0, 1.0))
        support_activity = np.clip((source_db - (ref_db - 30.0)) / 14.0, 0.0, 1.0)
        low_mid_energy = np.clip((ref_db - 18.0 - source_db) / 22.0, 0.0, 1.0)
        not_sustained_gap = 1.0 - np.clip((ref_db - 42.0 - source_db) / 8.0, 0.0, 1.0)
        quiet_curve = np.clip((ref_db - 35.0 - source_db) / 12.0, 0.0, 1.0)

        excess_rms = np.clip(
            (converted_rms - 1.05 * source_rms) / (converted_rms + eps),
            0.0,
            1.0,
        )
        excess_hf = np.clip(
            (converted_hf - 1.03 * source_hf) / (converted_hf + eps),
            0.0,
            1.0,
        )
        source_delta = np.abs(np.diff(source_rms, prepend=source_rms[:1]))
        converted_delta = np.abs(np.diff(converted_rms, prepend=converted_rms[:1]))
        spike_excess = np.clip(
            (converted_delta - (0.006 + 1.35 * source_delta)) / (converted_delta + eps),
            0.0,
            1.0,
        )

        tonalized_breath = (
            not_sustained_gap
            * np.clip((source_flatness - 0.20) / 0.24, 0.0, 1.0)
            * np.maximum(
                np.clip((source_flatness - converted_flatness) / 0.15, 0.0, 1.0),
                np.clip((converted_hf - 0.95 * source_hf) / (converted_hf + eps), 0.0, 1.0),
            )
        )
        quiet_residue = quiet_curve * np.maximum(excess_rms, excess_hf)
        active_residue = activity * np.maximum(
            spike_excess,
            np.clip((converted_hf - 1.15 * source_hf) / (converted_hf + eps), 0.0, 1.0),
        )
        converted_db = 20.0 * np.log10(converted_rms + eps)
        audible_output = np.clip((converted_db + 58.0) / 12.0, 0.0, 1.0)
        hf_ratio = converted_hf / (source_hf + eps)
        low_energy_tonal_hf = (
            audible_output
            * low_mid_energy
            * not_sustained_gap
            * np.maximum(
                np.clip((hf_ratio - 1.18) / 2.20, 0.0, 1.0),
                np.clip((converted_rms - 1.18 * source_rms) / (converted_rms + eps), 0.0, 1.0),
            )
        )

        breath_curve = 0.96 * tonalized_breath * np.maximum(excess_rms, excess_hf)
        quiet_fix_curve = 0.82 * quiet_residue
        active_fix_curve = 0.52 * active_residue
        low_energy_hf_curve = 0.90 * low_energy_tonal_hf
        blend_curve = np.maximum.reduce(
            [breath_curve, quiet_fix_curve, active_fix_curve, low_energy_hf_curve]
        )

        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)
        for _ in range(2):
            blend_curve = np.convolve(blend_curve, smooth_kernel, mode="same")

        blend_curve = self._hold_activity_curve(
            blend_curve,
            max(1, int(0.05 * converted_sr / hop_length)),
        )
        blend_curve = np.clip(blend_curve, 0.0, 1.0).astype(np.float32)

        rescued_frames = int(np.sum(blend_curve > 0.18))
        if rescued_frames <= 0:
            return

        frame_cap = np.maximum.reduce(
            [
                0.40 * activity,
                0.58 * support_activity * np.clip(spike_excess, 0.0, 1.0),
                0.70 * quiet_curve,
                0.84 * tonalized_breath,
                0.90 * low_energy_tonal_hf,
            ]
        )
        effective_curve = np.minimum(
            blend_curve,
            np.clip(frame_cap / max(max_source_blend, 1e-6), 0.0, 1.0),
        ).astype(np.float32)

        sample_blend = max_source_blend * self._frame_curve_to_sample_gain(
            effective_curve,
            aligned_len,
            hop_length,
        )[:aligned_len]
        sample_blend = np.clip(sample_blend, 0.0, max_source_blend).astype(np.float32)

        rescued_main = (
            converted_main * (1.0 - sample_blend)
            + source_main * sample_blend
        ).astype(np.float32)
        rescued_rms = librosa.feature.rms(
            y=rescued_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        rescued_rms = self._fit_frame_curve(rescued_rms, frame_count)
        risk_trim_curve = np.clip(
            low_energy_tonal_hf * (1.0 - 0.65 * activity),
            0.0,
            1.0,
        )
        for _ in range(2):
            risk_trim_curve = np.convolve(risk_trim_curve, smooth_kernel, mode="same")
        risk_trim_curve = self._hold_activity_curve(
            risk_trim_curve,
            max(1, int(0.04 * converted_sr / hop_length)),
        )
        risk_trim_curve = np.clip(risk_trim_curve, 0.0, 1.0).astype(np.float32)
        residual_floor = (10.0 ** (-58.0 / 20.0))
        target_rms = np.maximum(
            1.20 * source_rms + float(np.percentile(source_rms, 95)) * 0.0025,
            residual_floor,
        )
        trim_gain = np.clip(target_rms / (rescued_rms + eps), 0.42, 1.0)
        trim_gain = 1.0 - risk_trim_curve * (1.0 - trim_gain)
        trim_sample_gain = self._frame_curve_to_sample_gain(
            trim_gain,
            aligned_len,
            hop_length,
        )[:aligned_len]
        rescued_main = (rescued_main * trim_sample_gain).astype(np.float32)

        activity_weights = self._frame_curve_to_sample_gain(
            np.clip(activity, 0.0, 1.0),
            aligned_len,
            hop_length,
        )[:aligned_len]
        ref_active_rms = self._weighted_rms(converted_main, activity_weights)
        out_active_rms = self._weighted_rms(rescued_main, activity_weights)
        gain = 1.0
        if ref_active_rms > 1e-6 and out_active_rms < ref_active_rms * 0.92:
            gain = float(np.clip((0.95 * ref_active_rms) / (out_active_rms + eps), 1.0, 1.08))
            if gain > 1.001:
                rescued_main = self._apply_weighted_gain(
                    rescued_main,
                    activity_weights,
                    gain,
                ).astype(np.float32)

        rescued_main = soft_clip(rescued_main, threshold=0.92, ceiling=0.985)

        if len(converted_audio) > aligned_len:
            converted_audio = np.concatenate([rescued_main, converted_audio[aligned_len:]])
        else:
            converted_audio = rescued_main

        sf.write(converted_vocals_path, converted_audio.astype(np.float32), converted_sr)
        log.detail(
            "Artifact segment rescue: "
            f"rescued {rescued_frames}/{frame_count} frames "
            f"(breath={int(np.sum(breath_curve > 0.18))}, "
            f"quiet={int(np.sum(quiet_fix_curve > 0.18))}, "
            f"active={int(np.sum(active_fix_curve > 0.18))}, "
            f"tonal_hf={int(np.sum(low_energy_hf_curve > 0.18))}, "
            f"trim={int(np.sum(risk_trim_curve > 0.18))}, "
            f"gain={gain:.3f})"
        )

    @staticmethod
    def _compute_quiet_gap_sample_gain(
        reference_audio: np.ndarray,
        sr: int,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> Tuple[np.ndarray, int, int]:
        """Build a deep attenuation curve for sustained quiet gaps between vocal phrases."""
        import librosa

        reference_audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
        if reference_audio.size == 0:
            return np.zeros(0, dtype=np.float32), 0, 0

        eps = 1e-8
        frame_rms = librosa.feature.rms(
            y=reference_audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        frame_rms = np.asarray(frame_rms, dtype=np.float32)
        if frame_rms.size == 0:
            return np.ones(reference_audio.size, dtype=np.float32), 0, 0

        frame_db = 20.0 * np.log10(frame_rms + eps)
        ref_db = float(np.percentile(frame_db, 95))

        activity = np.clip((frame_db - (ref_db - 28.0)) / 14.0, 0.0, 1.0)
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        kernel /= np.sum(kernel)
        activity = np.convolve(activity, kernel, mode="same")
        activity = CoverPipeline._hold_activity_curve(
            activity,
            max(1, int(0.08 * sr / hop_length)),
        )

        quiet_mask = (
            (frame_db < (ref_db - 42.0))
            & (activity < 0.08)
        )

        min_frames = max(1, int(0.16 * sr / hop_length))
        gate = quiet_mask.astype(np.float32)
        filtered = np.zeros_like(gate)
        run_start = 0
        in_run = False
        for i in range(len(gate)):
            if gate[i] > 0.5:
                if not in_run:
                    run_start = i
                    in_run = True
            else:
                if in_run:
                    if (i - run_start) >= min_frames:
                        filtered[run_start:i] = 1.0
                    in_run = False
        if in_run and (len(gate) - run_start) >= min_frames:
            filtered[run_start:len(gate)] = 1.0

        transition_frames = max(1, int(0.05 * sr / hop_length))
        smooth_kernel = np.ones(transition_frames, dtype=np.float32) / transition_frames
        filtered = np.convolve(filtered, smooth_kernel, mode="same")
        filtered = np.clip(filtered, 0.0, 1.0)

        gain_curve = 1.0 - filtered * 0.55
        sample_gain = CoverPipeline._frame_curve_to_sample_gain(
            gain_curve,
            len(reference_audio),
            hop_length,
        )

        gated_count = int(np.sum(filtered > 0.5))
        return sample_gain.astype(np.float32), gated_count, len(filtered)

    def _compute_active_rms_gain(
        self,
        reference_audio: np.ndarray,
        target_audio: np.ndarray,
        sr: int,
        min_gain: float = 0.7,
        max_gain: float = 1.8,
    ) -> Tuple[float, float, float, np.ndarray]:
        """Estimate active-region gain and its sample-domain weight curve."""
        reference_audio = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
        target_audio = np.asarray(target_audio, dtype=np.float32).reshape(-1)
        aligned_len = min(reference_audio.size, target_audio.size)
        if aligned_len <= 0:
            return 1.0, 0.0, 0.0, np.zeros(0, dtype=np.float32)

        reference_audio = reference_audio[:aligned_len]
        target_audio = target_audio[:aligned_len]
        weights = self._compute_activity_sample_weights(reference_audio, sr)[:aligned_len]
        ref_rms = self._weighted_rms(reference_audio, weights)
        out_rms = self._weighted_rms(target_audio, weights)
        if ref_rms <= 1e-6 or out_rms <= 1e-6:
            return 1.0, ref_rms, out_rms, weights

        gain = float(np.clip(ref_rms / out_rms, min_gain, max_gain))
        return gain, ref_rms, out_rms, weights

    @staticmethod
    def _apply_weighted_gain(
        audio: np.ndarray,
        weights: np.ndarray,
        gain: float,
    ) -> np.ndarray:
        """Apply gain mainly on active vocal regions, not on tails/gaps."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        aligned_len = min(audio.size, weights.size)
        if aligned_len <= 0:
            return audio.astype(np.float32)

        output = audio.copy().astype(np.float32)
        gain_curve = 1.0 + np.clip(weights[:aligned_len], 0.0, 1.0) * float(gain - 1.0)
        output[:aligned_len] *= gain_curve.astype(np.float32)
        return output.astype(np.float32)

    @staticmethod
    def _gate_echo_tails(
        original: np.ndarray, dereverbed: np.ndarray, sr: int
    ) -> np.ndarray:
        """
        Gate echo-tail segments where dereverb removed most energy but
        residual noise would still trigger HuBERT feature extraction.
        """
        sample_gain, gated_count, total_frames = CoverPipeline._compute_echo_tail_sample_gain(
            original,
            dereverbed,
            sr,
        )
        if gated_count > 0:
            log.detail(f"回声尾音门控: {gated_count}/{total_frames} 帧被衰减")

        return (dereverbed * sample_gain).astype(np.float32)

    def _should_apply_source_constraint(
        self,
        vc_preprocessed: bool,
        source_constraint_mode: str,
    ) -> bool:
        """Decide whether to run source-guided post constraint."""
        normalized_mode = str(source_constraint_mode or "auto").strip().lower()
        if normalized_mode == "on":
            if self._last_vc_preprocess_mode == "direct":
                log.detail("源约束跳过: direct 模式下源未去回音，强制约束会放大回音伪影")
                return False
            return vc_preprocessed
        if normalized_mode == "auto":
            return vc_preprocessed and self._last_vc_preprocess_mode in {
                "uvr_deecho",
                "uvr_deecho_plus",
                "legacy",
                "advanced_dereverb",
            }
        return False

    def _refine_source_constrained_output(
        self,
        source_vocals_path: str,
        converted_vocals_path: str,
        source_constraint_mode: str,
        f0_method: str,
        original_vocals_path: Optional[str] = None,
        session_dir: Optional[Path] = None,
    ) -> None:
        """Apply extra cleanup passes for mature UVR DeEcho routing."""
        normalized_mode = str(source_constraint_mode or "auto").strip().lower()
        if normalized_mode != "auto":
            return
        if self._last_vc_preprocess_mode not in {"uvr_deecho", "uvr_deecho_plus"}:
            return

        self._apply_silence_gate_official(
            vocals_path=source_vocals_path,
            converted_path=converted_vocals_path,
            f0_method=f0_method,
            silence_threshold_db=-48.0,
            silence_smoothing_ms=35.0,
            silence_min_duration_ms=120.0,
            protect=0.35,
        )
        log.detail("Low-energy unvoiced cleanup: applied after source-guided reconstruction")

        self._apply_source_breath_cleanup(
            source_vocals_path=source_vocals_path,
            converted_vocals_path=converted_vocals_path,
        )

        self._apply_source_gap_suppression(
            source_vocals_path=source_vocals_path,
            converted_vocals_path=converted_vocals_path,
        )
        self._apply_source_transition_cleanup(
            source_vocals_path=source_vocals_path,
            converted_vocals_path=converted_vocals_path,
        )
        if original_vocals_path:
            self._restore_active_vocal_loudness(
                reference_vocals_path=original_vocals_path,
                converted_vocals_path=converted_vocals_path,
            )
        if session_dir:
            raw_snapshot_path = None
            for candidate_name in (
                "debug_converted_raw.wav",
                "debug_converted_raw_current.wav",
                "debug_converted_raw_upstream.wav",
                "debug_converted_raw_repair.wav",
            ):
                candidate_path = session_dir / candidate_name
                if candidate_path.exists():
                    raw_snapshot_path = str(candidate_path)
                    break
            if raw_snapshot_path:
                self._restore_voiced_body_from_raw(
                    raw_vocals_path=raw_snapshot_path,
                    source_vocals_path=source_vocals_path,
                    converted_vocals_path=converted_vocals_path,
                )
        self._apply_artifact_segment_rescue(
            source_vocals_path=source_vocals_path,
            converted_vocals_path=converted_vocals_path,
        )
        self._apply_source_gap_suppression(
            source_vocals_path=source_vocals_path,
            converted_vocals_path=converted_vocals_path,
        )
        log.detail("Source gap suppression: refined after source-guided reconstruction")

    @staticmethod
    def _blend_direct_with_deecho(
        direct_mono: np.ndarray,
        deecho_mono: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Blend direct lead with DeEcho result, using echo presence detection.

        Previous logic only applied DeEcho in low-activity (silent) regions,
        which meant echo during active singing passed straight through to HuBERT.
        Now we detect echo presence per-frame by comparing direct vs deecho energy:
        large energy difference = strong echo = higher DeEcho weight even while singing.
        """
        import librosa

        direct_mono = np.asarray(direct_mono, dtype=np.float32).reshape(-1)
        deecho_mono = np.asarray(deecho_mono, dtype=np.float32).reshape(-1)
        aligned_len = min(direct_mono.size, deecho_mono.size)
        if aligned_len <= 0:
            return direct_mono.astype(np.float32)

        direct_main = direct_mono[:aligned_len]
        deecho_main = deecho_mono[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8
        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)

        # --- Activity detection (unchanged) ---
        frame_rms = librosa.feature.rms(
            y=direct_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        frame_db = 20.0 * np.log10(frame_rms + eps)
        ref_db = float(np.percentile(frame_db, 95)) if frame_db.size > 0 else -20.0

        activity = np.clip((frame_db - (ref_db - 32.0)) / 14.0, 0.0, 1.0)
        activity = np.convolve(activity, smooth_kernel, mode="same")
        activity = CoverPipeline._hold_activity_curve(
            activity,
            max(1, int(0.04 * sr / hop_length)),
        )
        activity = np.clip(activity, 0.0, 1.0)

        # --- Echo presence detection ---
        # Compare per-frame RMS of direct vs deecho: if deecho removed a lot
        # of energy, that energy was echo/reverb.
        deecho_rms = librosa.feature.rms(
            y=deecho_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        n_frames = min(frame_rms.shape[-1], deecho_rms.shape[-1])
        frame_rms_aligned = frame_rms[..., :n_frames]
        deecho_rms_aligned = deecho_rms[..., :n_frames]

        # echo_ratio: how much energy was removed by deecho (0=none, 1=all)
        echo_ratio = np.clip(
            1.0 - (deecho_rms_aligned / (frame_rms_aligned + eps)),
            0.0,
            1.0,
        )
        # Smooth to avoid frame-level jitter
        if echo_ratio.ndim > 1:
            echo_ratio = echo_ratio[0]
        echo_ratio = np.convolve(echo_ratio, smooth_kernel, mode="same")
        # Widen with a hold window to cover reverb tails
        echo_ratio = CoverPipeline._hold_activity_curve(
            echo_ratio,
            max(1, int(0.08 * sr / hop_length)),
        )
        echo_ratio = np.clip(echo_ratio, 0.0, 1.0)

        # Align to activity length
        n_blend = min(len(activity), len(echo_ratio))
        activity = activity[:n_blend]
        echo_ratio = echo_ratio[:n_blend]

        # --- Blending weight ---
        # 全局回音水平驱动系数自适应
        global_echo = float(np.mean(echo_ratio))
        # 沉默段基权: 轻回音0.68, 重回音0.86
        base_coef = 0.68 + 0.18 * global_echo
        base_weight = base_coef * np.square(1.0 - activity[:n_blend])
        # 活跃唱段在检测到明显回声时更激进，但避免把主唱主体削得过干。
        echo_boost_coef = 0.60 + 0.20 * global_echo
        echo_boost = echo_boost_coef * echo_ratio * activity[:n_blend]
        active_floor = (
            activity[:n_blend]
            * np.clip((echo_ratio - 0.32) / 0.28, 0.0, 1.0)
            * (0.58 + 0.18 * global_echo)
        )
        tail_support = (
            (1.0 - activity[:n_blend])
            * np.clip((echo_ratio - 0.14) / 0.22, 0.0, 1.0)
            * (0.16 + 0.08 * global_echo)
        )
        deecho_weight = np.maximum(base_weight + echo_boost + tail_support, active_floor)
        deecho_weight = np.convolve(deecho_weight, smooth_kernel, mode="same")
        deecho_weight = np.clip(deecho_weight, 0.0, 0.95)
        deecho_weight = CoverPipeline._frame_curve_to_sample_gain(
            deecho_weight,
            aligned_len,
            hop_length,
        )

        blended = direct_main * (1.0 - deecho_weight) + deecho_main * deecho_weight
        if direct_mono.size > aligned_len:
            blended = np.concatenate([blended, direct_mono[aligned_len:]])
        return blended.astype(np.float32)

    @staticmethod
    def _merge_uvr_hotspot_cleanup(
        direct_mono: np.ndarray,
        deecho_mono: np.ndarray,
        aggressive_mono: np.ndarray,
        sr: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply the aggressive dereverb pass only where active echo is still obvious."""
        import librosa

        direct_mono = np.asarray(direct_mono, dtype=np.float32).reshape(-1)
        deecho_mono = np.asarray(deecho_mono, dtype=np.float32).reshape(-1)
        aggressive_mono = np.asarray(aggressive_mono, dtype=np.float32).reshape(-1)
        aligned_len = min(direct_mono.size, deecho_mono.size, aggressive_mono.size)
        if aligned_len <= 0:
            return deecho_mono.astype(np.float32), {
                "hotspot_frames": 0.0,
                "avg_weight": 0.0,
                "max_weight": 0.0,
                "coverage_ratio": 0.0,
            }

        direct_main = direct_mono[:aligned_len]
        deecho_main = deecho_mono[:aligned_len]
        aggressive_main = aggressive_mono[:aligned_len]

        frame_length = 2048
        hop_length = 512
        eps = 1e-8
        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)

        direct_rms = librosa.feature.rms(
            y=direct_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        deecho_rms = librosa.feature.rms(
            y=deecho_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        aggressive_rms = librosa.feature.rms(
            y=aggressive_main,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]

        frame_count = min(direct_rms.size, deecho_rms.size, aggressive_rms.size)
        if frame_count <= 0:
            return deecho_mono.astype(np.float32), {
                "hotspot_frames": 0.0,
                "avg_weight": 0.0,
                "max_weight": 0.0,
                "coverage_ratio": 0.0,
            }

        direct_rms = direct_rms[:frame_count].astype(np.float32)
        deecho_rms = deecho_rms[:frame_count].astype(np.float32)
        aggressive_rms = aggressive_rms[:frame_count].astype(np.float32)

        direct_db = 20.0 * np.log10(direct_rms + eps)
        ref_db = float(np.percentile(direct_db, 95))
        activity = np.square(np.clip((direct_db - (ref_db - 18.0)) / 9.0, 0.0, 1.0))
        support_activity = np.clip((direct_db - (ref_db - 24.0)) / 12.0, 0.0, 1.0)

        primary_removed = np.clip(1.0 - (deecho_rms / (direct_rms + eps)), 0.0, 1.0)
        secondary_removed = np.clip(1.0 - (aggressive_rms / (deecho_rms + eps)), 0.0, 1.0)
        hotspot_curve = (
            activity
            * np.clip((secondary_removed - 0.10) / 0.22, 0.0, 1.0)
            * np.clip((primary_removed - 0.02) / 0.10, 0.0, 1.0)
        )
        support_curve = (
            0.22
            * support_activity
            * np.clip((secondary_removed - 0.16) / 0.20, 0.0, 1.0)
        )
        aggressive_weight = np.clip(hotspot_curve + support_curve, 0.0, 0.62)
        aggressive_weight = np.convolve(aggressive_weight, smooth_kernel, mode="same")
        aggressive_weight = CoverPipeline._hold_activity_curve(
            aggressive_weight,
            max(1, int(0.03 * sr / hop_length)),
        )
        aggressive_weight = np.clip(aggressive_weight, 0.0, 0.62).astype(np.float32)

        sample_weight = CoverPipeline._frame_curve_to_sample_gain(
            aggressive_weight,
            aligned_len,
            hop_length,
        )[:aligned_len]
        cleaned_main = (
            deecho_main * (1.0 - sample_weight)
            + aggressive_main * sample_weight
        ).astype(np.float32)

        if deecho_mono.size > aligned_len:
            cleaned = np.concatenate([cleaned_main, deecho_mono[aligned_len:]])
        else:
            cleaned = cleaned_main

        return cleaned.astype(np.float32), {
            "hotspot_frames": float(np.sum(aggressive_weight > 0.20)),
            "avg_weight": float(np.mean(aggressive_weight)),
            "max_weight": float(np.max(aggressive_weight)),
            "coverage_ratio": float(np.mean(aggressive_weight > 0.20)),
        }

    def _prepare_vocals_for_vc(
        self,
        vocals_path: str,
        session_dir: Path,
        preprocess_mode: str = "auto",
    ) -> str:
        """
        Prepare vocals for VC using a mature-project-friendly routing strategy.

        Modes:
        - auto: prefer public RoFormer De-Reverb, then UVR DeEcho, otherwise advanced dereverb -> RVC
        - direct: pass separated lead directly to RVC
        - uvr_deecho: require learned DeEcho/DeReverb if available, else fallback to advanced dereverb
        - advanced_dereverb: use binary residual masking to separate dry/wet, convert dry only
        - legacy: old hand-crafted dereverb + tail gating chain
        """
        import librosa
        import soundfile as sf

        preprocess_mode = str(preprocess_mode or "auto").strip().lower()
        if preprocess_mode not in {"auto", "direct", "uvr_deecho", "advanced_dereverb", "legacy"}:
            preprocess_mode = "auto"

        # 保存原始混响用于后处理
        self._original_reverb_path = None
        self._uvr_deecho_metrics = None

        if preprocess_mode == "advanced_dereverb":
            # 使用高级去混响：分离干声和混响
            audio, sr = librosa.load(vocals_path, sr=None, mono=False)
            audio = self._ensure_2d(audio).astype(np.float32)
            mono = self._select_mono_for_vc(audio, sr)

            log.detail("VC preprocess: advanced dereverb (binary residual masking)")
            dry_signal, reverb_tail = advanced_dereverb(mono, sr)

            # 保存混响用于后处理
            reverb_path = session_dir / "original_reverb.wav"
            sf.write(str(reverb_path), reverb_tail, sr)
            self._original_reverb_path = str(reverb_path)

            mono = dry_signal
            self._last_vc_preprocess_mode = "advanced_dereverb"
            log.detail(f"Dry/Wet separation: dry RMS={np.sqrt(np.mean(dry_signal**2)):.4f}, reverb RMS={np.sqrt(np.mean(reverb_tail**2)):.4f}")

        elif preprocess_mode == "legacy":
            audio, sr = librosa.load(vocals_path, sr=None, mono=False)
            audio = self._ensure_2d(audio).astype(np.float32)
            mono = self._select_mono_for_vc(audio, sr)
            mono_dry = mono.copy()
            mono = self._dereverb_for_vc(mono, sr)
            mono = self._gate_echo_tails(mono_dry, mono, sr)
            self._last_vc_preprocess_mode = "legacy"
            log.detail("VC preprocess: legacy dereverb chain -> mono select")
        else:
            preprocess_input = vocals_path
            mono_resolved = False

            if preprocess_mode in {"auto", "uvr_deecho"}:
                preprocess_input = (
                    self._apply_roformer_deecho_for_vc(vocals_path, session_dir)
                    or self._apply_uvr_deecho_for_vc(vocals_path, session_dir)
                    or vocals_path
                )

            if preprocess_input == vocals_path:
                if preprocess_mode in {"auto", "uvr_deecho"}:
                    # auto / uvr_deecho 模式在学习型模型缺失时都回退到 advanced_dereverb
                    audio, sr = librosa.load(vocals_path, sr=None, mono=False)
                    audio = self._ensure_2d(audio).astype(np.float32)
                    mono = self._select_mono_for_vc(audio, sr)

                    fallback_name = "auto" if preprocess_mode == "auto" else "uvr_deecho"
                    log.detail(f"VC preprocess ({fallback_name}): learned DeEcho not available, using advanced dereverb")
                    dry_signal, reverb_tail = advanced_dereverb(mono, sr)

                    # 保存混响用于后处理
                    reverb_path = session_dir / "original_reverb.wav"
                    sf.write(str(reverb_path), reverb_tail, sr)
                    self._original_reverb_path = str(reverb_path)

                    mono = dry_signal
                    self._last_vc_preprocess_mode = "advanced_dereverb"
                    mono_resolved = True
                    log.detail(f"Dry/Wet separation: dry RMS={np.sqrt(np.mean(dry_signal**2)):.4f}, reverb RMS={np.sqrt(np.mean(reverb_tail**2)):.4f}")
                else:
                    # direct 模式
                    self._last_vc_preprocess_mode = "direct"
                    log.detail("VC preprocess: direct lead -> mono select")
            else:
                self._last_vc_preprocess_mode = "uvr_deecho"
                log.detail("VC preprocess: learned DeEcho/DeReverb -> mono select")

            # 最终 mono 确定（仅在 mono 未被上面解决时执行）
            if not mono_resolved:
                if preprocess_input == vocals_path:
                    audio, sr = librosa.load(preprocess_input, sr=None, mono=False)
                    audio = self._ensure_2d(audio).astype(np.float32)
                    mono = self._select_mono_for_vc(audio, sr)
                else:
                    direct_audio, sr = librosa.load(vocals_path, sr=None, mono=False)
                    deecho_audio, deecho_sr = librosa.load(preprocess_input, sr=None, mono=False)
                    direct_audio = self._ensure_2d(direct_audio).astype(np.float32)
                    deecho_audio = self._ensure_2d(deecho_audio).astype(np.float32)
                    direct_mono = self._select_mono_for_vc(direct_audio, sr)
                    deecho_mono = self._select_mono_for_vc(deecho_audio, deecho_sr)
                    if deecho_sr != sr:
                        deecho_mono = librosa.resample(
                            deecho_mono,
                            orig_sr=deecho_sr,
                            target_sr=sr,
                        ).astype(np.float32)

                    # DeEcho 质量检测：用 UVR 候选打分指标判断是否跳过 blend
                    uvr_metrics = getattr(self, '_uvr_deecho_metrics', None)
                    skip_blend = False
                    secondary_dry_cleanup = False
                    hotspot_cleaned = False
                    hotspot_stats = {
                        "hotspot_frames": 0.0,
                        "avg_weight": 0.0,
                        "max_weight": 0.0,
                        "coverage_ratio": 0.0,
                    }
                    if uvr_metrics:
                        sep_db = uvr_metrics.get('separation_db', 0.0)
                        corr = uvr_metrics.get('corr', 0.0)
                        active_ratio = uvr_metrics.get('active_ratio', 1.0)
                        reduction_ratio = uvr_metrics.get('reduction_ratio', 0.0)
                        log.detail(
                            "DeEcho quality: "
                            f"sep={sep_db:.2f}dB, corr={corr:.3f}, "
                            f"active_ratio={active_ratio:.3f}, reduction={reduction_ratio:.3f}"
                        )
                        # sep/corr only mean the output is coherent. We should skip blend
                        # only when the DeEcho branch is also meaningfully drier than source.
                        if (
                            sep_db > 30.0
                            and corr > 0.9
                            and not (active_ratio > 0.90 and reduction_ratio < 0.25)
                        ):
                            skip_blend = True
                        if active_ratio > 0.93 and reduction_ratio < 0.18:
                            secondary_dry_cleanup = True

                    if secondary_dry_cleanup:
                        dry_signal, reverb_tail = advanced_dereverb(deecho_mono, sr)
                        deecho_rms = float(np.sqrt(np.mean(np.square(deecho_mono)) + 1e-12))
                        delta_rms = float(
                            np.sqrt(np.mean(np.square(dry_signal.astype(np.float32) - deecho_mono)) + 1e-12)
                        )
                        delta_ratio = float(delta_rms / (deecho_rms + 1e-12))
                        if delta_ratio > 0.025:
                            cleaned_mono, hotspot_stats = CoverPipeline._merge_uvr_hotspot_cleanup(
                                direct_mono=direct_mono,
                                deecho_mono=deecho_mono,
                                aggressive_mono=dry_signal.astype(np.float32),
                                sr=sr,
                            )
                            if hotspot_stats.get("hotspot_frames", 0.0) > 0:
                                deecho_mono = cleaned_mono.astype(np.float32)
                                hotspot_cleaned = True
                        if hotspot_cleaned:
                            self._last_vc_preprocess_mode = "uvr_deecho_plus"
                            log.detail(
                                "VC preprocess: UVR deecho active-echo hotspot cleanup "
                                f"(delta_ratio={delta_ratio:.3f}, "
                                f"hotspot_frames={int(hotspot_stats.get('hotspot_frames', 0.0))}, "
                                f"coverage={hotspot_stats.get('coverage_ratio', 0.0):.3f}, "
                                f"avg_weight={hotspot_stats.get('avg_weight', 0.0):.3f}, "
                                f"max_weight={hotspot_stats.get('max_weight', 0.0):.3f})"
                            )
                        else:
                            log.detail(
                                "VC preprocess: advanced dereverb second pass had no strong active-echo hotspots, "
                                f"keeping UVR deecho output (delta_ratio={delta_ratio:.3f})"
                            )

                    if hotspot_cleaned:
                        skip_blend = False
                        coverage_ratio = float(hotspot_stats.get("coverage_ratio", 0.0))
                        if coverage_ratio > 0.35:
                            log.detail(
                                "VC preprocess: hotspot cleanup touched a wide region; "
                                "forcing direct/deecho blend to preserve vocal body"
                            )

                    if skip_blend:
                        mono = deecho_mono
                        log.detail("VC preprocess: UVR DeEcho quality sufficient, using deecho directly (skip blend)")
                    else:
                        mono = CoverPipeline._blend_direct_with_deecho(direct_mono, deecho_mono, sr)
                        log.detail("VC preprocess: blended direct lead with UVR DeEcho (enhanced)")

        mono = soft_clip(mono, threshold=0.9, ceiling=0.99)

        out_path = session_dir / "vocals_for_vc.wav"
        sf.write(str(out_path), mono, sr)
        return str(out_path)

    def _suppress_lead_bleed_from_backing(
        self,
        lead_audio: np.ndarray,
        backing_audio: np.ndarray,
    ) -> np.ndarray:
        """
        抑制 backing 里残留的主唱，减少 converted lead + 原主唱残留造成的重音。
        """
        import librosa

        n_fft = 4096
        hop_length = 1024
        suppression = 0.9
        min_mask = 0.08
        eps = 1e-8

        cleaned = np.zeros_like(backing_audio, dtype=np.float32)
        for ch in range(backing_audio.shape[0]):
            backing_ch = backing_audio[ch]
            lead_ch = lead_audio[ch]
            backing_spec = librosa.stft(
                backing_ch, n_fft=n_fft, hop_length=hop_length, win_length=n_fft
            )
            lead_spec = librosa.stft(
                lead_ch, n_fft=n_fft, hop_length=hop_length, win_length=n_fft
            )

            backing_mag = np.abs(backing_spec)
            lead_mag = np.abs(lead_spec)
            residual_mag = np.maximum(backing_mag - suppression * lead_mag, 0.0)
            soft_mask = residual_mag / (backing_mag + eps)
            soft_mask = np.clip(soft_mask, min_mask, 1.0)

            cleaned_spec = backing_spec * soft_mask
            cleaned[ch] = librosa.istft(
                cleaned_spec, hop_length=hop_length, win_length=n_fft, length=len(backing_ch)
            )

        return cleaned.astype(np.float32)

    def _duck_backing_under_lead(
        self,
        lead_audio: np.ndarray,
        backing_audio: np.ndarray,
        sr: int,
        base_gain: float = 0.92,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Keep backing present while making room for the lead on overlap sections."""
        import librosa

        lead_audio = self._ensure_2d(lead_audio).astype(np.float32)
        backing_audio = self._ensure_2d(backing_audio).astype(np.float32)
        backing_audio = self._match_channels(backing_audio, lead_audio.shape[0])

        output = backing_audio.copy().astype(np.float32)
        output *= float(base_gain)

        aligned_len = min(lead_audio.shape[1], backing_audio.shape[1])
        if aligned_len <= 2048:
            return output, {
                "base_gain": float(base_gain),
                "overlap_ratio": 0.0,
                "duck_need": 0.0,
                "avg_duck_db": 0.0,
                "max_duck_db": 0.0,
            }

        lead_mono = lead_audio[:, :aligned_len].mean(axis=0).astype(np.float32)
        backing_mono = backing_audio[:, :aligned_len].mean(axis=0).astype(np.float32)

        lead_weights = self._compute_activity_sample_weights(lead_mono, sr)[:aligned_len]
        backing_weights = self._compute_activity_sample_weights(backing_mono, sr)[:aligned_len]
        overlap_weights = np.clip(
            lead_weights * (0.35 + 0.65 * backing_weights),
            0.0,
            1.0,
        ).astype(np.float32)

        lead_overlap_rms = self._weighted_rms(lead_mono, overlap_weights)
        backing_overlap_rms = self._weighted_rms(backing_mono, overlap_weights)
        overlap_ratio = float(backing_overlap_rms / (lead_overlap_rms + 1e-12))

        duck_need = float(np.clip((overlap_ratio - 0.18) / 0.20, 0.0, 1.0))
        if duck_need <= 0.02:
            return output, {
                "base_gain": float(base_gain),
                "overlap_ratio": overlap_ratio,
                "duck_need": duck_need,
                "avg_duck_db": 0.0,
                "max_duck_db": 0.0,
            }

        frame_length = 2048
        hop_length = 512
        eps = 1e-8
        lead_rms = librosa.feature.rms(
            y=lead_mono,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)
        backing_rms = librosa.feature.rms(
            y=backing_mono,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0].astype(np.float32)

        frame_count = min(lead_rms.size, backing_rms.size)
        if frame_count <= 4:
            return output, {
                "base_gain": float(base_gain),
                "overlap_ratio": overlap_ratio,
                "duck_need": duck_need,
                "avg_duck_db": 0.0,
                "max_duck_db": 0.0,
            }

        lead_rms = lead_rms[:frame_count]
        backing_rms = backing_rms[:frame_count]

        lead_db = 20.0 * np.log10(lead_rms + eps)
        backing_db = 20.0 * np.log10(backing_rms + eps)
        lead_peak_db = float(np.percentile(lead_db, 95))
        backing_peak_db = float(np.percentile(backing_db, 92))

        lead_activity = np.square(
            np.clip((lead_db - (lead_peak_db - 24.0)) / 11.0, 0.0, 1.0)
        )
        backing_density = np.clip(
            (backing_db - (backing_peak_db - 18.0)) / 10.0,
            0.0,
            1.0,
        )

        smooth_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        smooth_kernel /= np.sum(smooth_kernel)
        lead_activity = np.convolve(lead_activity, smooth_kernel, mode="same")
        backing_density = np.convolve(backing_density, smooth_kernel, mode="same")
        lead_activity = self._hold_activity_curve(
            lead_activity,
            max(1, int(0.18 * sr / hop_length)),
        )

        duck_curve = np.clip(
            lead_activity * (0.35 + 0.65 * backing_density),
            0.0,
            1.0,
        ).astype(np.float32)
        sample_curve = self._frame_curve_to_sample_gain(
            duck_curve,
            aligned_len,
            hop_length,
        )

        max_duck_db = float(1.8 + 3.2 * duck_need)
        dynamic_gain = np.power(
            10.0,
            -(max_duck_db * sample_curve) / 20.0,
        ).astype(np.float32)
        output[:, :aligned_len] *= dynamic_gain[np.newaxis, :]

        avg_duck_db = float(max_duck_db * float(np.mean(sample_curve)))
        return output.astype(np.float32), {
            "base_gain": float(base_gain),
            "overlap_ratio": overlap_ratio,
            "duck_need": duck_need,
            "avg_duck_db": avg_duck_db,
            "max_duck_db": float(max_duck_db * float(np.max(sample_curve))),
        }

    def _merge_backing_into_accompaniment(
        self,
        backing_vocals_path: str,
        accompaniment_path: str,
        session_dir: Path,
        lead_vocals_path: Optional[str] = None,
        duck_reference_path: Optional[str] = None,
    ) -> str:
        """将和声轨混入伴奏轨；可选抑制 backing 内残留主唱"""
        import librosa
        import soundfile as sf

        backing, backing_sr = librosa.load(backing_vocals_path, sr=None, mono=False)
        accompaniment, accompaniment_sr = librosa.load(accompaniment_path, sr=None, mono=False)

        backing = self._ensure_2d(backing).astype(np.float32)
        accompaniment = self._ensure_2d(accompaniment).astype(np.float32)

        if backing_sr != accompaniment_sr:
            backing = self._resample_audio(backing, orig_sr=backing_sr, target_sr=accompaniment_sr)

        if lead_vocals_path:
            lead, lead_sr = librosa.load(lead_vocals_path, sr=None, mono=False)
            lead = self._ensure_2d(lead).astype(np.float32)
            if lead_sr != accompaniment_sr:
                lead = self._resample_audio(lead, orig_sr=lead_sr, target_sr=accompaniment_sr)
            lead = self._match_channels(lead, backing.shape[0])

            min_len = min(backing.shape[1], lead.shape[1])
            backing = backing[:, :min_len]
            lead = lead[:, :min_len]
            backing = self._suppress_lead_bleed_from_backing(
                lead_audio=lead,
                backing_audio=backing,
            )

        duck_reference = None
        if duck_reference_path and Path(duck_reference_path).exists():
            duck_reference, duck_sr = librosa.load(duck_reference_path, sr=None, mono=False)
            duck_reference = self._ensure_2d(duck_reference).astype(np.float32)
            if duck_sr != accompaniment_sr:
                duck_reference = self._resample_audio(
                    duck_reference,
                    orig_sr=duck_sr,
                    target_sr=accompaniment_sr,
                )
            duck_reference = self._match_channels(duck_reference, backing.shape[0])

        accompaniment = self._match_channels(accompaniment, backing.shape[0])
        max_len = max(accompaniment.shape[1], backing.shape[1])
        if accompaniment.shape[1] < max_len:
            accompaniment = np.pad(
                accompaniment, ((0, 0), (0, max_len - accompaniment.shape[1])), mode="constant"
            )
        if backing.shape[1] < max_len:
            backing = np.pad(backing, ((0, 0), (0, max_len - backing.shape[1])), mode="constant")

        backing_gain = 1.00
        backing = backing * backing_gain
        log.detail(f"和声混入伴奏增益: {backing_gain:.2f}")
        mixed = accompaniment + backing
        mixed = soft_clip(mixed, threshold=0.92, ceiling=0.98)

        out_path = session_dir / "accompaniment_with_backing.wav"
        sf.write(str(out_path), mixed.T, accompaniment_sr)
        return str(out_path)

    def _init_rvc_pipeline(self):
        """初始化 RVC 管道"""
        if self.rvc_pipeline is not None:
            return

        from infer.pipeline import VoiceConversionPipeline

        self.rvc_pipeline = VoiceConversionPipeline(device=self.device)

    def _apply_silence_gate_official(
        self,
        vocals_path: str,
        converted_path: str,
        f0_method: str,
        silence_threshold_db: float,
        silence_smoothing_ms: float,
        silence_min_duration_ms: float,
        protect: float
    ):
        """对官方转换后的人声应用静音门限（可选）"""
        from lib.audio import load_audio, save_audio
        from infer.pipeline import VoiceConversionPipeline
        import soundfile as sf

        # Load original vocals at 16k for RMS/F0 reference
        audio_in = load_audio(vocals_path, sr=16000)

        # Extract F0 using the configured method
        gate_pipe = VoiceConversionPipeline(device=self.device)
        root_dir = Path(__file__).parent.parent
        rmvpe_path = root_dir / "assets" / "rmvpe" / "rmvpe.pt"
        gate_policy = resolve_cover_f0_policy(f0_method)
        gate_method = gate_policy.gate_method
        if gate_method in ("rmvpe", "hybrid"):
            if not rmvpe_path.exists():
                raise FileNotFoundError(f"RMVPE 模型未找到: {rmvpe_path}")
            gate_pipe.load_f0_extractor(gate_method, str(rmvpe_path))
        else:
            gate_pipe.load_f0_extractor(gate_method, None)
        f0 = gate_pipe.f0_extractor.extract(audio_in)
        gate_pipe.unload_f0_extractor()

        # Load converted vocals (keep original sample rate)
        audio_out, sr_out = sf.read(converted_path)
        if audio_out.ndim > 1:
            audio_out = audio_out.mean(axis=1)
        audio_out = audio_out.astype(np.float32)

        audio_out = gate_pipe._apply_silence_gate(
            audio_out=audio_out,
            audio_in=audio_in,
            f0=f0,
            sr_out=sr_out,
            sr_in=16000,
            hop_length=160,
            threshold_db=silence_threshold_db,
            smoothing_ms=silence_smoothing_ms,
            min_silence_ms=silence_min_duration_ms,
            protect=protect
        )

        save_audio(converted_path, audio_out, sr=sr_out)

    def _blend_backing_vocals(
        self,
        converted_path: str,
        original_vocals_path: str,
        mix_ratio: float,
        output_path: Optional[str] = None
    ) -> str:
        """混入原始人声以恢复和声层"""
        if mix_ratio <= 0:
            return converted_path

        import librosa
        import soundfile as sf

        conv, sr = librosa.load(converted_path, sr=None, mono=True)
        orig, sr_orig = librosa.load(original_vocals_path, sr=None, mono=True)
        if sr_orig != sr:
            orig = librosa.resample(orig, orig_sr=sr_orig, target_sr=sr)

        min_len = min(len(conv), len(orig))
        conv = conv[:min_len]
        orig = orig[:min_len]

        mixed = conv * (1.0 - mix_ratio) + orig * mix_ratio
        mixed = soft_clip(mixed, threshold=0.9, ceiling=0.98)

        if output_path is None:
            output_path = str(Path(converted_path).with_suffix("").as_posix() + "_blend.wav")

        sf.write(output_path, mixed, sr)
        return output_path

    def _constrain_converted_to_source(
        self,
        source_vocals_path: str,
        converted_vocals_path: str,
        original_vocals_path: str = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Use source-vocal-guided spectral constraint to suppress artifacts that are
        absent from the source lead (e.g. spurious echo/noise produced by VC).
        """
        import librosa
        import soundfile as sf

        src, src_sr = librosa.load(source_vocals_path, sr=None, mono=True)
        conv, conv_sr = librosa.load(converted_vocals_path, sr=None, mono=True)
        src = src.astype(np.float32)
        conv = conv.astype(np.float32)

        if src_sr != conv_sr:
            src = librosa.resample(src, orig_sr=src_sr, target_sr=conv_sr).astype(np.float32)

        aligned_len = min(len(src), len(conv))
        if aligned_len <= 0:
            raise ValueError("源主唱或转换人声为空，无法执行源约束")

        src = src[:aligned_len]
        conv_main = conv[:aligned_len]
        conv_tail = conv[aligned_len:]

        n_fft = 2048
        hop_length = 512
        win_length = 2048
        eps = 1e-8

        src_spec = librosa.stft(
            src, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        conv_spec = librosa.stft(
            conv_main, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        src_mag = np.abs(src_spec).astype(np.float32)
        conv_mag = np.abs(conv_spec).astype(np.float32)

        frame_count = conv_spec.shape[1]

        # Echo-like component tends to persist from previous frames.
        prev_mag = np.concatenate([src_mag[:, :1], src_mag[:, :-1]], axis=1)
        echo_like = np.minimum(src_mag, 0.92 * prev_mag)
        echo_ratio = np.clip(echo_like / (src_mag + eps), 0.0, 1.0)
        direct_floor = (1.0 - echo_ratio) * 0.18 * src_mag
        direct_ref = np.maximum(src_mag - 0.60 * echo_like, direct_floor)

        extra_mag = np.maximum(conv_mag - direct_ref, 0.0)
        soft_mask = direct_ref / (direct_ref + 0.7 * extra_mag + eps)

        frame_ref = np.mean(direct_ref, axis=0)
        frame_conv = np.mean(conv_mag, axis=0)
        frame_mask = np.clip((frame_ref + eps) / (frame_conv + eps), 0.0, 1.0)
        frame_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        frame_kernel /= np.sum(frame_kernel)
        frame_mask = np.convolve(frame_mask, frame_kernel, mode="same")
        soft_mask *= frame_mask[np.newaxis, :]

        time_kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        time_kernel /= np.sum(time_kernel)
        soft_mask = np.apply_along_axis(
            lambda row: np.convolve(row, time_kernel, mode="same"),
            axis=1,
            arr=soft_mask,
        )
        soft_mask = np.clip(soft_mask, 0.0, 1.0)
        src_frame_rms = librosa.feature.rms(
            y=src,
            frame_length=win_length,
            hop_length=hop_length,
            center=True,
        )[0]
        src_frame_rms = self._fit_frame_curve(src_frame_rms, frame_count)
        src_frame_db = 20.0 * np.log10(src_frame_rms + eps)
        ref_db = float(np.percentile(src_frame_db, 95))
        frame_src_mag = np.mean(src_mag, axis=0)
        direct_ratio = np.clip(frame_ref / (frame_src_mag + eps), 0.0, 1.0)
        direct_ratio = self._fit_frame_curve(direct_ratio, frame_count)

        orig = None
        orig_frame_rms = src_frame_rms.copy()
        orig_frame_db = src_frame_db.copy()
        orig_ref_db = ref_db
        if original_vocals_path is not None:
            orig, orig_sr = librosa.load(original_vocals_path, sr=None, mono=True)
            if orig_sr != conv_sr:
                orig = librosa.resample(orig, orig_sr=orig_sr, target_sr=conv_sr).astype(np.float32)
            orig = orig[:aligned_len].astype(np.float32)
            orig_frame_rms = librosa.feature.rms(
                y=orig,
                frame_length=win_length,
                hop_length=hop_length,
                center=True,
            )[0]
            orig_frame_rms = self._fit_frame_curve(orig_frame_rms, frame_count)
            orig_frame_db = 20.0 * np.log10(orig_frame_rms + eps)
            orig_ref_db = float(np.percentile(orig_frame_db, 95))

        # Use time-domain RMS activity instead of STFT mean magnitude.
        # Echo-only frames often keep wide-band STFT energy but very low direct vocal RMS.
        direct_activity = np.clip((src_frame_db - (ref_db - 30.0)) / 18.0, 0.0, 1.0)
        direct_activity = np.convolve(direct_activity, frame_kernel, mode="same")
        direct_activity = self._fit_frame_curve(direct_activity, frame_count)

        vocal_activity = np.clip((orig_frame_db - (orig_ref_db - 30.0)) / 18.0, 0.0, 1.0)
        vocal_activity = np.convolve(vocal_activity, frame_kernel, mode="same")
        vocal_activity = self._fit_frame_curve(vocal_activity, frame_count)
        phrase_activity = self._hold_activity_curve(
            vocal_activity,
            max(1, int(0.28 * conv_sr / hop_length)),
        )

        activity = np.maximum(direct_activity, phrase_activity)

        mask_floor = 0.02 + 0.14 * (0.25 * direct_activity + 0.20 * direct_ratio + 0.55 * phrase_activity)
        mask_floor = np.convolve(mask_floor, frame_kernel, mode="same")
        mask_floor = self._fit_frame_curve(mask_floor, frame_count)
        soft_mask = np.maximum(soft_mask, mask_floor[np.newaxis, :])
        soft_mask = np.clip(soft_mask, 0.0, 1.0)

        # Step 1: Magnitude-only constraint in STFT domain
        # Instead of mixing source and converted complex spectra (which causes
        # phase interference / tearing artifacts), we only constrain the
        # MAGNITUDE toward the source envelope while preserving the converted
        # signal's phase.  This eliminates phase cancellation.
        source_replace = compute_active_source_replace(
            activity=activity,
            soft_mask=soft_mask,
            echo_ratio=echo_ratio,
            direct_ratio=direct_ratio,
        )

        # Target magnitude: blend toward source magnitude, keep converted phase
        target_mag = conv_mag * (1.0 - source_replace) + src_mag * source_replace
        # Compute gain per bin: how much to scale converted magnitude
        mag_gain = target_mag / (conv_mag + eps)
        mag_gain = np.clip(mag_gain, 0.05, 2.0)
        constrained_spec = conv_spec * mag_gain

        replaced_frames = int(np.sum(np.mean(source_replace, axis=0) > 0.05))
        if replaced_frames > 0:
            log.detail(
                f"源低活动段幅度约束: {replaced_frames}/{frame_count} 帧抑制幻觉噪声(相位保留)"
            )

        # Step 2: istft to get constrained main body
        constrained = librosa.istft(
            constrained_spec,
            hop_length=hop_length,
            win_length=win_length,
            length=aligned_len,
        ).astype(np.float32)

        # Step 3: Symmetric global gain (only on main body, before tail concat)
        # 增益目标用原始主唱（未去混响），避免目标偏低
        gain, ref_rms, out_rms, gain_weights = self._compute_active_rms_gain(
            reference_audio=orig if orig is not None else src,
            target_audio=constrained,
            sr=conv_sr,
            min_gain=0.85,
            max_gain=1.12,
        )
        if abs(gain - 1.0) > 1e-3 and out_rms > 1e-6 and ref_rms > 1e-6:
            constrained = self._apply_weighted_gain(constrained, gain_weights, gain)
            log.detail(
                f"Source-constrained active RMS: ref={ref_rms:.6f}, out={out_rms:.6f}, gain={gain:.3f}"
            )

        constrained_frame_rms = librosa.feature.rms(
            y=constrained,
            frame_length=win_length,
            hop_length=hop_length,
            center=True,
        )[0]
        constrained_frame_rms = self._fit_frame_curve(constrained_frame_rms, frame_count)
        base_budget_rms = np.maximum(src_frame_rms, orig_frame_rms)
        ref_frame_rms = float(np.percentile(base_budget_rms, 95))
        energy_guard = np.clip(0.20 * direct_activity + 0.15 * direct_ratio + 0.65 * phrase_activity, 0.0, 1.0)
        noise_floor = ref_frame_rms * (0.002 + 0.005 * (1.0 - phrase_activity))  # 降低noise_floor
        allowed_boost, cleanup_floor = compute_source_cleanup_budget(
            energy_guard=energy_guard,
            phrase_activity=phrase_activity,
        )
        frame_budget = base_budget_rms * allowed_boost + noise_floor
        cleanup_gain = np.clip(
            frame_budget / (constrained_frame_rms + eps),
            cleanup_floor,
            1.0,
        )
        cleanup_gain = np.convolve(cleanup_gain, frame_kernel, mode="same")
        cleanup_gain = self._fit_frame_curve(cleanup_gain, frame_count)
        attenuated_frames = int(np.sum(cleanup_gain < 0.98))
        if attenuated_frames > 0:
            constrained = constrained * self._frame_curve_to_sample_gain(
                cleanup_gain,
                len(constrained),
                hop_length,
            )
            log.detail(
                f"源能量预算清理: {attenuated_frames}/{frame_count} 帧抑制超额转换残留"
            )

        if original_vocals_path is not None:
            try:
                orig_gate, orig_gate_sr = librosa.load(original_vocals_path, sr=None, mono=True)
                if orig_gate_sr != conv_sr:
                    orig_gate = librosa.resample(
                        orig_gate,
                        orig_sr=orig_gate_sr,
                        target_sr=conv_sr,
                    ).astype(np.float32)
                orig_gate = orig_gate[:aligned_len].astype(np.float32)
                echo_tail_gain, gated_count, total_frames = self._compute_echo_tail_sample_gain(
                    original=orig_gate,
                    dereverbed=src,
                    sr=conv_sr,
                )
                if gated_count > 0:
                    constrained = constrained * echo_tail_gain[:len(constrained)]
                    log.detail(
                        f"源回声尾段同步抑制: {gated_count}/{total_frames} 帧应用到转换人声"
                    )
            except Exception as e:
                log.warning(f"源回声尾段同步抑制失败，跳过: {e}")

        # Step 4: Append tail with fade-out (tail is likely noise from VC overshoot)
        if conv_tail.size > 0:
            tail_fade = np.linspace(1.0, 0.0, len(conv_tail)).astype(np.float32)
            constrained = np.concatenate([constrained, conv_tail * tail_fade * 0.18])

        constrained = soft_clip(constrained, threshold=0.9, ceiling=0.99)

        if output_path is None:
            output_path = converted_vocals_path
        sf.write(output_path, constrained, conv_sr)
        return output_path

    def process(
        self,
        input_audio: str,
        model_path: str,
        index_path: Optional[str] = None,
        pitch_shift: int = 0,
        index_ratio: float = 0.5,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        speaker_id: int = 0,
        f0_method: str = "rmvpe",
        demucs_model: str = "htdemucs",
        demucs_shifts: int = 2,
        demucs_overlap: float = 0.25,
        demucs_split: bool = True,
        roformer_model: str = ROFORMER_DEFAULT_MODEL,
        separator: str = "uvr5",
        uvr5_model: Optional[str] = None,
        uvr5_agg: int = 10,
        uvr5_format: str = "wav",
        use_official: bool = True,
        hubert_layer: int = 12,
        silence_gate: bool = False,
        silence_threshold_db: float = -40.0,
        silence_smoothing_ms: float = 50.0,
        silence_min_duration_ms: float = 200.0,
        vocals_volume: float = 1.0,
        accompaniment_volume: float = 1.0,
        reverb_amount: float = 0.0,
        backing_mix: float = 0.0,
        karaoke_separation: bool = True,
        karaoke_model: str = KARAOKE_DEFAULT_MODEL,
        karaoke_merge_backing_into_accompaniment: bool = True,
        vc_preprocess_mode: str = "auto",
        source_constraint_mode: str = "auto",
        vc_pipeline_mode: str = "current",
        singing_repair: bool = False,
        output_dir: Optional[str] = None,
        model_display_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, str]:
        """
        执行完整的翻唱流程

        Args:
            input_audio: 输入歌曲路径
            model_path: RVC 模型路径
            index_path: 索引文件路径 (可选)
            pitch_shift: 音调偏移 (半音)
            index_ratio: 索引混合比率
            index_ratio: 索引混合比率
            filter_radius: 中值滤波半径
            rms_mix_rate: RMS 混合比率
            protect: 保护参数
            speaker_id: 说话人 ID（多说话人模型可调）
            f0_method: F0 提取方法
            demucs_model: Demucs 模型名称
            demucs_shifts: Demucs shifts 参数
            demucs_overlap: Demucs overlap 参数
            demucs_split: Demucs split 参数
            roformer_model: RoFormer / audio-separator 模型或 ensemble preset
            hubert_layer: HuBERT 输出层
            silence_gate: 是否启用静音门限
            silence_threshold_db: 静音阈值 (dB, 相对峰值)
            silence_smoothing_ms: 门限平滑时长 (ms)
            silence_min_duration_ms: 最短静音时长 (ms)
            vocals_volume: 人声音量 (0-2)
            accompaniment_volume: 伴奏音量 (0-2)
            reverb_amount: 人声混响量 (0-1)
            backing_mix: 原始人声混入比例 (0-1)
            output_dir: 输出目录 (可选)
            progress_callback: 进度回调 (message, current_step, total_steps)

        Returns:
            dict: {
                "cover": 最终翻唱路径,
                "vocals": 原始人声路径,
                "converted_vocals": 转换后人声路径,
                "accompaniment": 伴奏路径
            }
        """
        normalized_vc_pipeline_mode = str(vc_pipeline_mode or "current").strip().lower()
        if normalized_vc_pipeline_mode not in {"current", "official"}:
            normalized_vc_pipeline_mode = "current"
        effective_official_mode = normalized_vc_pipeline_mode == "official"
        effective_separator = "uvr5" if effective_official_mode else separator
        effective_karaoke_separation = False if effective_official_mode else karaoke_separation
        effective_karaoke_merge_backing = False if effective_official_mode else karaoke_merge_backing_into_accompaniment
        effective_use_official = True if effective_official_mode else use_official

        # 官方模式：强制使用官方推荐参数，确保1:1纯净推理
        if effective_official_mode:
            if f0_method != "rmvpe":
                log.warning(f"官方模式：F0方法从 {f0_method} 强制切换为 rmvpe（抗噪性最佳）")
                f0_method = "rmvpe"
            if protect != 0.33:
                log.warning(f"官方模式：保护系数从 {protect} 强制设为 0.33（官方推荐值）")
                protect = 0.33

        total_steps = 5 if effective_karaoke_separation else 4
        step_karaoke = 2 if effective_karaoke_separation else None
        step_convert = 3 if effective_karaoke_separation else 2
        step_mix = 4 if effective_karaoke_separation else 3
        step_finalize = 5 if effective_karaoke_separation else 4
        session_dir = self._get_session_dir()

        # 记录输入信息
        input_path = Path(input_audio)
        input_size = input_path.stat().st_size if input_path.exists() else 0
        input_duration = _get_audio_duration(input_audio)

        log.separator()
        log.info(f"开始翻唱处理: {input_path.name}")
        log.detail(f"输入文件: {input_audio}")
        log.detail(f"文件大小: {_format_size(input_size)}")
        log.detail(f"音频时长: {_format_duration(input_duration)}")
        log.detail(f"会话目录: {session_dir}")
        log.detail(f"Quality debug report: {session_dir / 'quality_debug.json'}")
        log.separator()
        # 记录参数配置
        log.config(f"RVC模型: {Path(model_path).name}")
        log.config(f"索引文件: {Path(index_path).name if index_path else '无'}")
        log.config(f"音调偏移: {pitch_shift} 半音")
        log.config(f"F0提取方法: {f0_method}")
        log.config(f"索引混合比率: {index_ratio}")
        log.config(f"说话人ID: {speaker_id}")
        log.config(f"VC管线模式: {normalized_vc_pipeline_mode}")
        if effective_official_mode:
            log.config("官方模式: 强制UVR5分离 + 去混响预处理 + 官方VC (rmvpe, protect=0.33)")
        log.config(f"人声分离器: {effective_separator}")
        if effective_separator == "uvr5":
            log.config(f"UVR5模型: {uvr5_model or '自动选择'}")
            log.config(f"UVR5激进度: {uvr5_agg}")
        elif effective_separator == "roformer":
            log.config(f"Roformer模型: {roformer_model}")
        else:
            log.config(f"Demucs模型: {demucs_model}")
            log.config(f"Demucs shifts: {demucs_shifts}")
        log.config(f"人声音量: {vocals_volume}")
        log.config(f"伴奏音量: {accompaniment_volume}")
        log.config(f"混响量: {reverb_amount}")
        log.separator()

        log.config(f"Karaoke分离: {'开启' if effective_karaoke_separation else '关闭'}")
        if effective_karaoke_separation:
            log.config(f"Karaoke模型: {karaoke_model}")
            log.config(
                "Karaoke和声混入伴奏: "
                f"{'开启' if effective_karaoke_merge_backing else '关闭'}"
            )
        elif effective_official_mode:
            log.config("Karaoke分离: 官方模式下关闭")

        def report_progress(msg: str, step: int):
            if progress_callback:
                progress_callback(msg, step, total_steps)
            log.step(step, total_steps, msg)

        try:
            # ===== 步骤 1: 人声分离 =====
            report_progress("正在分离人声和伴奏...", 1)

            if effective_official_mode:
                log.model("官方模式：使用内置官方UVR5进行人声分离")
                uvr_temp = session_dir / "official_uvr5"
                log.detail(f"官方UVR5临时目录: {uvr_temp}")
                vocals_path, accompaniment_path = separate_uvr5_official_upstream(
                    input_audio,
                    uvr_temp,
                    uvr5_model,
                    agg=uvr5_agg,
                    fmt=uvr5_format,
                )
            elif effective_use_official and effective_separator == "uvr5":
                log.model("使用当前项目官方封装UVR5进行人声分离")
                setup_official_env(Path(__file__).parent.parent)
                uvr_temp = session_dir / "uvr5"
                log.detail(f"UVR5临时目录: {uvr_temp}")
                vocals_path, accompaniment_path = separate_uvr5(
                    input_audio,
                    uvr_temp,
                    uvr5_model,
                    agg=uvr5_agg,
                    fmt=uvr5_format,
                )
                log.success("UVR5分离完成")
            elif effective_separator == "roformer":
                log.model("使用公开 SOTA RoFormer ensemble 进行人声分离")
                self._init_separator(
                    "roformer",
                    roformer_model=roformer_model,
                )
                vocals_path, accompaniment_path = self.separator.separate(
                    input_audio,
                    str(session_dir)
                )
                log.success("RoFormer ensemble 分离完成")
            else:
                log.model(f"使用Demucs进行人声分离: {demucs_model}")
                self._init_separator(
                    demucs_model,
                    shifts=demucs_shifts,
                    overlap=demucs_overlap,
                    split=demucs_split
                )
                vocals_path, accompaniment_path = self.separator.separate(
                    input_audio,
                    str(session_dir)
                )
                log.success("Demucs分离完成")
            gc.collect()
            empty_device_cache()
            log.detail("已清理设备缓存")

            # ===== 步骤 1.5: Karaoke 分离（主唱/和声）=====
            original_vocals_path = vocals_path
            lead_vocals_path = None
            backing_vocals_path = None

            if effective_karaoke_separation:
                report_progress("正在分离主唱和和声...", step_karaoke)
                lead_vocals_path, backing_vocals_path = self._separate_karaoke(
                    vocals_path=vocals_path,
                    session_dir=session_dir,
                    karaoke_model=karaoke_model,
                )
                lead_size = Path(lead_vocals_path).stat().st_size if Path(lead_vocals_path).exists() else 0
                backing_size = Path(backing_vocals_path).stat().st_size if Path(backing_vocals_path).exists() else 0
                log.audio(f"主唱文件: {Path(lead_vocals_path).name} ({_format_size(lead_size)})")
                log.audio(f"和声文件: {Path(backing_vocals_path).name} ({_format_size(backing_size)})")
                vocals_path = lead_vocals_path

            normalized_vc_preprocess_mode = str(vc_preprocess_mode or "auto").strip().lower()
            normalized_source_constraint_mode = str(source_constraint_mode or "auto").strip().lower()
            available_deecho_model = self._get_preferred_deecho_model_label()
            log.config(f"VC预处理模式: {normalized_vc_preprocess_mode}")
            if normalized_vc_preprocess_mode in {"auto", "uvr_deecho"}:
                if available_deecho_model:
                    log.config(f"Mature DeEcho模型: {available_deecho_model}")
                else:
                    log.config("Mature DeEcho模型: 未找到，将回退到 advanced dereverb")
            log.config(f"源约束模式: {normalized_source_constraint_mode}")

            # 官方模式也必须经过去混响预处理，确保输入RVC的是纯净干声
            # 官方模式下如果用户选了 direct，强制提升为 auto（带混响的人声会破坏F0提取）
            effective_preprocess_mode = normalized_vc_preprocess_mode
            if normalized_vc_pipeline_mode == "official" and effective_preprocess_mode == "direct":
                effective_preprocess_mode = "auto"
                log.warning("官方模式：direct预处理已提升为auto，确保去混响后再进入RVC推理")

            vc_input_path = vocals_path
            vc_preprocessed = False
            try:
                prepared_path = self._prepare_vocals_for_vc(vocals_path, session_dir, preprocess_mode=effective_preprocess_mode)
                vc_input_path = prepared_path
                vc_preprocessed = True
                log.audio(f"VC预处理输入: {Path(vc_input_path).name}")
            except Exception as e:
                log.warning(f"VC预处理失败，回退原始输入: {e}")

            report_progress("正在转换人声...", step_convert)
            self._record_quality_debug(
                session_dir=session_dir,
                stage="vc_input",
                candidate_path=vc_input_path,
                reference_path=vocals_path if vc_input_path != vocals_path else None,
                extra={
                    "vc_preprocessed": vc_preprocessed,
                    "requested_preprocess_mode": normalized_vc_preprocess_mode,
                    "effective_preprocess_mode": effective_preprocess_mode,
                },
            )
            converted_vocals_path = str(session_dir / "converted_vocals.wav")

            log.model(f"加载RVC模型: {Path(model_path).name}")
            log.detail(f"输入人声: {vc_input_path}")
            log.detail(f"输出路径: {converted_vocals_path}")
            if normalized_vc_pipeline_mode == "official" and not singing_repair:
                log.detail("使用内置官方VC实现进行转换")
                log.config(f"F0方法: {f0_method}, 音调: {pitch_shift}, 索引率: {index_ratio}")
                log.config(f"滤波半径: {filter_radius}, RMS混合: {rms_mix_rate}, 保护: {protect}")

                convert_vocals_official_upstream(
                    vocals_path=vc_input_path,
                    output_path=converted_vocals_path,
                    model_path=model_path,
                    index_path=index_path,
                    f0_method=f0_method,
                    pitch_shift=pitch_shift,
                    index_rate=index_ratio,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                    speaker_id=speaker_id,
                )
                self._record_quality_debug(
                    session_dir=session_dir,
                    stage="vc_raw_upstream",
                    candidate_path=converted_vocals_path,
                    reference_path=vc_input_path,
                    snapshot_label="debug_converted_raw_upstream",
                    extra={"source_constraint_mode": normalized_source_constraint_mode},
                )
                log.detail("内置官方模式：去混响干声 -> 官方RVC推理（纯净管道）")
                log.success("内置官方VC转换完成")
            elif normalized_vc_pipeline_mode == "official" and singing_repair:
                log.detail("使用官方兼容唱歌修复链进行转换")
                log.config(f"F0方法: {f0_method}, 音调: {pitch_shift}, 索引率: {index_ratio}")
                log.config(f"滤波半径: {filter_radius}, RMS混合: {rms_mix_rate}, 保护: {protect}")
                log.config("唱歌修复: 开启（FP32 + 保守F0兜底 + F0稳定/限速）")

                convert_vocals_official(
                    vocals_path=vc_input_path,
                    output_path=converted_vocals_path,
                    model_path=model_path,
                    index_path=index_path,
                    f0_method=f0_method,
                    pitch_shift=pitch_shift,
                    index_rate=index_ratio,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                    speaker_id=speaker_id,
                    repair_profile=True,
                )
                self._record_quality_debug(
                    session_dir=session_dir,
                    stage="vc_raw_repair",
                    candidate_path=converted_vocals_path,
                    reference_path=vc_input_path,
                    snapshot_label="debug_converted_raw_repair",
                    extra={"source_constraint_mode": normalized_source_constraint_mode},
                )
                try:
                    self._apply_silence_gate_official(
                        vocals_path=vc_input_path,
                        converted_path=converted_vocals_path,
                        f0_method=f0_method,
                        silence_threshold_db=-38.0,
                        silence_smoothing_ms=35.0,
                        silence_min_duration_ms=70.0,
                        protect=0.0,
                    )
                    log.detail("唱歌修复: 已应用低能量静音清理")
                except Exception as e:
                    log.warning(f"唱歌修复静音清理失败，保留原始转换结果: {e}")

                try:
                    self._apply_source_gap_suppression(
                        source_vocals_path=vc_input_path,
                        converted_vocals_path=converted_vocals_path,
                    )
                    log.detail("唱歌修复: 已应用源静音区抑制")
                except Exception as e:
                    log.warning(f"唱歌修复静音区抑制失败，保留当前结果: {e}")
                log.success("官方兼容唱歌修复转换完成")
            elif effective_use_official:
                log.detail("VC backend: upstream_official_raw + current postprocess")
                log.detail("VC route detail: vendored upstream official raw -> current cleanup chain")
                log.config(f"F0方法: {f0_method}, 音调: {pitch_shift}, 索引率: {index_ratio}")
                log.config(f"滤波半径: {filter_radius}, RMS混合: {rms_mix_rate}, 保护: {protect}")

                convert_vocals_official_upstream(
                    vocals_path=vc_input_path,
                    output_path=converted_vocals_path,
                    model_path=model_path,
                    index_path=index_path,
                    f0_method=f0_method,
                    pitch_shift=pitch_shift,
                    index_rate=index_ratio,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                    speaker_id=speaker_id,
                )
                self._record_quality_debug(
                    session_dir=session_dir,
                    stage="vc_raw",
                    candidate_path=converted_vocals_path,
                    reference_path=vc_input_path,
                    snapshot_label="debug_converted_raw",
                    extra={"source_constraint_mode": normalized_source_constraint_mode},
                )
                if silence_gate:
                    log.detail("启用静音门限（当前项目官方封装VC后处理）")
                    self._apply_silence_gate_official(
                        vocals_path=vc_input_path,
                        converted_path=converted_vocals_path,
                        f0_method=f0_method,
                        silence_threshold_db=silence_threshold_db,
                        silence_smoothing_ms=silence_smoothing_ms,
                        silence_min_duration_ms=silence_min_duration_ms,
                        protect=protect
                    )
                normalized_source_constraint_mode = str(source_constraint_mode or "auto").strip().lower()
                should_apply_source_constraint = self._should_apply_source_constraint(
                    vc_preprocessed=vc_preprocessed,
                    source_constraint_mode=normalized_source_constraint_mode,
                )

                if should_apply_source_constraint:
                    try:
                        self._constrain_converted_to_source(
                            source_vocals_path=vc_input_path,
                            converted_vocals_path=converted_vocals_path,
                            original_vocals_path=vocals_path,
                        )
                        self._record_quality_debug(
                            session_dir=session_dir,
                            stage="vc_source_constrained",
                            candidate_path=converted_vocals_path,
                            reference_path=vc_input_path,
                            snapshot_label="debug_converted_source_constrained",
                            extra={"source_constraint_mode": normalized_source_constraint_mode},
                        )
                        log.detail("Applied source-guided reconstruction to suppress echo/noise")
                        self._refine_source_constrained_output(
                            source_vocals_path=vc_input_path,
                            converted_vocals_path=converted_vocals_path,
                            source_constraint_mode=normalized_source_constraint_mode,
                            f0_method=f0_method,
                            original_vocals_path=vocals_path,
                            session_dir=session_dir,
                        )
                        self._record_quality_debug(
                            session_dir=session_dir,
                            stage="vc_refined",
                            candidate_path=converted_vocals_path,
                            reference_path=vc_input_path,
                            snapshot_label="debug_converted_refined",
                            extra={"source_constraint_mode": normalized_source_constraint_mode},
                        )
                    except Exception as e:
                        log.warning(f"Source-guided reconstruction failed, keeping raw conversion: {e}")
                elif vc_preprocessed and normalized_source_constraint_mode == "off":
                    log.detail("Source constraint: off")
                elif vc_preprocessed and normalized_source_constraint_mode == "auto":
                    try:
                        self._apply_source_gap_suppression(
                            source_vocals_path=vc_input_path,
                            converted_vocals_path=converted_vocals_path,
                        )
                        self._record_quality_debug(
                            session_dir=session_dir,
                            stage="vc_gap_suppressed",
                            candidate_path=converted_vocals_path,
                            reference_path=vc_input_path,
                            snapshot_label="debug_converted_gap_suppressed",
                            extra={"source_constraint_mode": normalized_source_constraint_mode},
                        )
                        log.detail("Source gap suppression: applied for mature/default route")
                    except Exception as e:
                        log.warning(f"Source gap suppression failed, keeping raw conversion: {e}")
                elif vc_preprocessed:
                    log.detail("Skipping source-guided reconstruction for this preprocess mode")
                else:
                    log.warning("VC preprocess unavailable, skipping source-guided reconstruction")
                log.success("官方VC转换完成")

            # 如果使用了advanced dereverb，重新应用原始混响（仅非官方模式）
            if (
                not effective_official_mode
                and not effective_use_official
                and hasattr(self, '_original_reverb_path')
                and self._original_reverb_path
                and Path(self._original_reverb_path).exists()
            ):
                log.detail("重新应用原始混响到转换后的干声...")
                import librosa
                import soundfile as sf

                converted_dry, sr = librosa.load(converted_vocals_path, sr=None, mono=True)
                original_reverb, reverb_sr = librosa.load(self._original_reverb_path, sr=None, mono=True)

                if reverb_sr != sr:
                    original_reverb = librosa.resample(original_reverb, orig_sr=reverb_sr, target_sr=sr).astype(np.float32)

                # 重新应用混响（80%强度）
                wet_signal = apply_reverb_to_converted(converted_dry, original_reverb, mix_ratio=0.8)

                # 保存带混响的版本
                sf.write(converted_vocals_path, wet_signal, sr)
                log.detail(f"混响重应用完成: mix_ratio=0.8")

            elif not effective_official_mode and not effective_use_official:
                # 使用自定义VC管道进行转换
                log.detail("使用自定义VC管道进行转换")
                self._init_rvc_pipeline()
                self.rvc_pipeline.hubert_layer = hubert_layer
                log.config(f"HuBERT层: {hubert_layer}")

                root_dir = Path(__file__).parent.parent
                hubert_path = root_dir / "assets" / "hubert" / "hubert_base.pt"
                rmvpe_path = root_dir / "assets" / "rmvpe" / "rmvpe.pt"

                if self.rvc_pipeline.hubert_model is None:
                    if hubert_path.exists():
                        log.model(f"加载HuBERT模型: {hubert_path}")
                        self.rvc_pipeline.load_hubert(str(hubert_path))
                        log.success("HuBERT模型加载完成")
                    else:
                        raise FileNotFoundError(f"HuBERT 模型未找到: {hubert_path}")

                if self.rvc_pipeline.f0_extractor is None:
                    if f0_method in ("rmvpe", "hybrid"):
                        if rmvpe_path.exists():
                            log.model(f"加载RMVPE模型: {rmvpe_path}")
                            self.rvc_pipeline.load_f0_extractor(f0_method, str(rmvpe_path))
                            log.success(f"{f0_method.upper()}模型加载完成")
                        else:
                            raise FileNotFoundError(f"RMVPE 模型未找到: {rmvpe_path}")
                    else:
                        log.model(f"加载F0提取器: {f0_method}")
                        self.rvc_pipeline.load_f0_extractor(f0_method, None)

                log.model(f"加载声音模型: {Path(model_path).name}")
                self.rvc_pipeline.load_voice_model(model_path)
                if index_path:
                    log.model(f"加载索引文件: {Path(index_path).name}")
                    self.rvc_pipeline.load_index(index_path)

                log.progress("开始人声转换...")
                self.rvc_pipeline.convert(
                    audio_path=vc_input_path,
                    output_path=converted_vocals_path,
                    pitch_shift=pitch_shift,
                    index_ratio=index_ratio,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                    speaker_id=speaker_id,
                    silence_gate=silence_gate,
                    silence_threshold_db=silence_threshold_db,
                    silence_smoothing_ms=silence_smoothing_ms,
                    silence_min_duration_ms=silence_min_duration_ms,
                )
                self._record_quality_debug(
                    session_dir=session_dir,
                    stage="vc_raw_current",
                    candidate_path=converted_vocals_path,
                    reference_path=vc_input_path,
                    snapshot_label="debug_converted_raw_current",
                    extra={"source_constraint_mode": normalized_source_constraint_mode},
                )
                normalized_source_constraint_mode = str(source_constraint_mode or "auto").strip().lower()
                should_apply_source_constraint = self._should_apply_source_constraint(
                    vc_preprocessed=vc_preprocessed,
                    source_constraint_mode=normalized_source_constraint_mode,
                )

                if should_apply_source_constraint:
                    try:
                        self._constrain_converted_to_source(
                            source_vocals_path=vc_input_path,
                            converted_vocals_path=converted_vocals_path,
                            original_vocals_path=vocals_path,
                        )
                        self._record_quality_debug(
                            session_dir=session_dir,
                            stage="vc_source_constrained_current",
                            candidate_path=converted_vocals_path,
                            reference_path=vc_input_path,
                            snapshot_label="debug_converted_source_constrained_current",
                            extra={"source_constraint_mode": normalized_source_constraint_mode},
                        )
                        log.detail("Applied source-guided reconstruction to suppress echo/noise")
                        self._refine_source_constrained_output(
                            source_vocals_path=vc_input_path,
                            converted_vocals_path=converted_vocals_path,
                            source_constraint_mode=normalized_source_constraint_mode,
                            f0_method=f0_method,
                            original_vocals_path=vocals_path,
                            session_dir=session_dir,
                        )
                        self._record_quality_debug(
                            session_dir=session_dir,
                            stage="vc_refined_current",
                            candidate_path=converted_vocals_path,
                            reference_path=vc_input_path,
                            snapshot_label="debug_converted_refined_current",
                            extra={"source_constraint_mode": normalized_source_constraint_mode},
                        )
                    except Exception as e:
                        log.warning(f"Source-guided reconstruction failed, keeping raw conversion: {e}")
                elif vc_preprocessed and normalized_source_constraint_mode == "off":
                    log.detail("Source constraint: off")
                elif vc_preprocessed and normalized_source_constraint_mode == "auto":
                    try:
                        self._apply_source_gap_suppression(
                            source_vocals_path=vc_input_path,
                            converted_vocals_path=converted_vocals_path,
                        )
                        self._record_quality_debug(
                            session_dir=session_dir,
                            stage="vc_gap_suppressed_current",
                            candidate_path=converted_vocals_path,
                            reference_path=vc_input_path,
                            snapshot_label="debug_converted_gap_suppressed_current",
                            extra={"source_constraint_mode": normalized_source_constraint_mode},
                        )
                        log.detail("Source gap suppression: applied for mature/default route")
                    except Exception as e:
                        log.warning(f"Source gap suppression failed, keeping raw conversion: {e}")
                elif vc_preprocessed:
                    log.detail("Skipping source-guided reconstruction for this preprocess mode")
                else:
                    log.warning("VC preprocess unavailable, skipping source-guided reconstruction")
                log.success("自定义VC转换完成")

                log.detail("释放RVC管道资源...")
                self.rvc_pipeline.unload_all()
                gc.collect()
                empty_device_cache()
                log.detail("已清理设备缓存")

            # 记录转换结果
            self._record_quality_debug(
                session_dir=session_dir,
                stage="vc_final_state",
                candidate_path=converted_vocals_path,
                reference_path=vc_input_path if Path(vc_input_path).exists() else None,
                extra={"source_constraint_mode": normalized_source_constraint_mode},
            )
            self._maybe_log_diagnostic_hint(
                session_dir=session_dir,
                model_path=model_path,
                index_path=index_path,
            )
            converted_size = Path(converted_vocals_path).stat().st_size if Path(converted_vocals_path).exists() else 0
            log.audio(f"转换后人声: {Path(converted_vocals_path).name} ({_format_size(converted_size)})")

            mix_vocals_path = converted_vocals_path
            if backing_mix > 0:
                try:
                    blended_path = str(session_dir / "converted_vocals_blend.wav")
                    mix_vocals_path = self._blend_backing_vocals(
                        converted_path=converted_vocals_path,
                        original_vocals_path=vocals_path,
                        mix_ratio=backing_mix,
                        output_path=blended_path
                    )
                    log.detail(f"已混入原始人声: ratio={backing_mix:.2f}")
                except Exception as e:
                    log.warning(f"混入原始人声失败，使用转换人声: {e}")

            if (
                effective_karaoke_separation
                and effective_karaoke_merge_backing
                and backing_vocals_path
            ):
                accompaniment_path = self._merge_backing_into_accompaniment(
                    backing_vocals_path=backing_vocals_path,
                    accompaniment_path=accompaniment_path,
                    session_dir=session_dir,
                    lead_vocals_path=lead_vocals_path,
                    duck_reference_path=mix_vocals_path,
                )
                log.detail("已将和声混入伴奏轨道")

            # ===== 步骤 3: 混音 =====
            report_progress("正在混合人声和伴奏...", step_mix)

            cover_path = str(session_dir / "cover.wav")
            log.detail(f"混音输出: {cover_path}")
            log.config(f"人声音量: {vocals_volume}, 伴奏音量: {accompaniment_volume}, 混响: {reverb_amount}")

            mix_vocals_and_accompaniment(
                vocals_path=mix_vocals_path,
                accompaniment_path=accompaniment_path,
                output_path=cover_path,
                vocals_volume=vocals_volume,
                accompaniment_volume=accompaniment_volume,
                reverb_amount=reverb_amount
            )

            self._record_quality_debug(
                session_dir=session_dir,
                stage="cover_mix",
                candidate_path=cover_path,
                reference_path=None,
                extra={
                    "vocals_volume": vocals_volume,
                    "accompaniment_volume": accompaniment_volume,
                    "reverb_amount": reverb_amount,
                },
            )
            cover_size = Path(cover_path).stat().st_size if Path(cover_path).exists() else 0
            log.success(f"混音完成: {_format_size(cover_size)}")

            # ===== 步骤 4: 整理输出 =====
            report_progress("正在整理输出文件...", step_finalize)

            # 如果指定了输出目录，复制文件
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                log.detail(f"输出目录: {output_path}")

                input_name = Path(input_audio).stem
                # Gradio 临时路径可能在 stem 里残留路径分隔符，只取最后一段
                if "/" in input_name or "\\" in input_name:
                    input_name = Path(input_name).name
                # 去掉 Gradio 上传时追加的随机后缀（如 -0-100）
                input_name = re.sub(r'-\d+-\d+$', '', input_name)
                # 拼上角色名
                tag = f"_{model_display_name}" if model_display_name else ""
                final_cover = str(output_path / f"{input_name}{tag}_cover.wav")
                final_vocals = str(output_path / f"{input_name}_vocals.wav")
                final_converted = str(output_path / f"{input_name}{tag}_converted.wav")
                final_accompaniment = str(output_path / f"{input_name}_accompaniment.wav")
                final_lead = str(output_path / f"{input_name}_lead_vocals.wav")
                final_backing = str(output_path / f"{input_name}_backing_vocals.wav")

                log.detail(f"复制翻唱文件: {final_cover}")
                shutil.copy(cover_path, final_cover)
                log.detail(f"复制原始人声: {final_vocals}")
                shutil.copy(original_vocals_path, final_vocals)
                log.detail(f"复制转换人声: {final_converted}")
                shutil.copy(converted_vocals_path, final_converted)
                log.detail(f"复制伴奏文件: {final_accompaniment}")
                shutil.copy(accompaniment_path, final_accompaniment)

                if effective_karaoke_separation and lead_vocals_path and backing_vocals_path:
                    log.detail(f"复制主唱文件: {final_lead}")
                    shutil.copy(lead_vocals_path, final_lead)
                    log.detail(f"复制和声文件: {final_backing}")
                    shutil.copy(backing_vocals_path, final_backing)

                # 完整保留本次会话所有中间文件（分离结果、主唱/和声、回灌前后文件等）
                all_files_dir = output_path / f"{input_name}{tag}_all_files_{session_dir.name}"
                log.detail(f"复制全部中间文件: {all_files_dir}")
                shutil.copytree(session_dir, all_files_dir, dirs_exist_ok=True)

                result = {
                    "cover": final_cover,
                    "vocals": final_vocals,
                    "converted_vocals": final_converted,
                    "accompaniment": final_accompaniment,
                    "all_files_dir": str(all_files_dir),
                }
                if effective_karaoke_separation and lead_vocals_path and backing_vocals_path:
                    result["lead_vocals"] = final_lead
                    result["backing_vocals"] = final_backing
            else:
                result = {
                    "cover": cover_path,
                    "vocals": original_vocals_path,
                    "converted_vocals": converted_vocals_path,
                    "accompaniment": accompaniment_path,
                    "all_files_dir": str(session_dir),
                }
                if effective_karaoke_separation and lead_vocals_path and backing_vocals_path:
                    result["lead_vocals"] = lead_vocals_path
                    result["backing_vocals"] = backing_vocals_path
                if karaoke_separation and lead_vocals_path and backing_vocals_path:
                    result["lead_vocals"] = lead_vocals_path
                    result["backing_vocals"] = backing_vocals_path

            log.separator()
            report_progress("翻唱完成!", step_finalize)
            log.success(f"最终输出: {result['cover']}")
            log.separator()
            return result

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            log.separator()
            log.error(f"处理失败: {e}")
            log.error(f"详细错误:\n{error_detail}")
            log.separator()
            report_progress(f"处理失败: {e}", 0)
            raise

    def cleanup_session(self, session_dir: str):
        """清理会话临时文件"""
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)

    def cleanup_all(self):
        """清理所有临时文件"""
        if self.separator is not None:
            self.separator.unload_model()
            self.separator = None
        if self.karaoke_separator is not None:
            self.karaoke_separator.unload_model()
            self.karaoke_separator = None

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)


# 全局实例
_cover_pipeline = None


def get_cover_pipeline(device: str = "cuda") -> CoverPipeline:
    """获取翻唱流水线单例"""
    global _cover_pipeline
    if _cover_pipeline is None:
        _cover_pipeline = CoverPipeline(device=device)
    return _cover_pipeline
