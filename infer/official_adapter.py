# -*- coding: utf-8 -*-
"""
Adapter for official RVC WebUI modules (VC + UVR5).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import soundfile as sf

from configs.config import Config as OfficialConfig
from infer.quality_policy import resolve_cover_f0_policy
from lib.logger import log


def _load_app_config(root_dir: Path) -> dict:
    config_path = root_dir / "configs" / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_cfg_value(app_cfg: dict, key: str, default):
    cover_cfg = app_cfg.get("cover")
    if isinstance(cover_cfg, dict) and key in cover_cfg:
        return cover_cfg.get(key, default)
    return app_cfg.get(key, default)


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resolve_index_path(model_path: Path, index_path: Optional[str]) -> Optional[Path]:
    """Best-effort resolve of the matching FAISS index for a model."""
    if index_path:
        idx_path = Path(index_path)
        if idx_path.exists():
            return idx_path

    direct_candidate = model_path.with_suffix(".index")
    if direct_candidate.exists():
        return direct_candidate

    index_files = list(model_path.parent.glob("*.index"))
    if not index_files:
        return None
    if len(index_files) == 1:
        return index_files[0]

    def _normalize_name(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _tokenize_name(text: str):
        return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 2]

    model_norm = _normalize_name(model_path.stem)
    model_tokens = set(_tokenize_name(model_path.stem))

    best_match = None
    best_score = -1
    for idx in index_files:
        idx_norm = _normalize_name(idx.stem)
        idx_tokens = set(_tokenize_name(idx.stem))
        score = 0
        if idx_norm == model_norm:
            score += 1000
        if model_norm and (model_norm in idx_norm or idx_norm in model_norm):
            score += 300
        score += len(model_tokens & idx_tokens) * 40
        if "added" in idx.stem.lower():
            score += 10
        if score > best_score:
            best_score = score
            best_match = idx

    if best_match is not None and best_score > 0:
        return best_match
    return None


def setup_official_env(root_dir: Path) -> dict:
    """Set env vars used by official modules."""
    log.detail("配置官方模块环境变量...")
    app_cfg = _load_app_config(root_dir)
    paths = app_cfg.get("paths", {})

    weights_dir = root_dir / app_cfg.get("weights_dir", paths.get("weights", "assets/weights"))
    rmvpe_root = root_dir / app_cfg.get("rmvpe_path", paths.get("rmvpe", "assets/rmvpe/rmvpe.pt"))
    rmvpe_root = rmvpe_root.parent
    uvr5_root = root_dir / "assets" / "uvr5_weights"

    official_models = weights_dir / "official_models"
    official_indexes = weights_dir / "official_indexes"
    official_models.mkdir(parents=True, exist_ok=True)
    official_indexes.mkdir(parents=True, exist_ok=True)
    uvr5_root.mkdir(parents=True, exist_ok=True)

    os.environ["weight_root"] = str(official_models)
    os.environ["index_root"] = str(official_indexes)
    os.environ["outside_index_root"] = str(official_indexes)
    os.environ["rmvpe_root"] = str(rmvpe_root)
    os.environ["weight_uvr5_root"] = str(uvr5_root)

    log.detail(f"模型目录: {official_models}")
    log.detail(f"索引目录: {official_indexes}")
    log.detail(f"UVR5目录: {uvr5_root}")
    log.detail(f"RMVPE目录: {rmvpe_root}")

    # Ensure official config cache directories exist.
    inuse_root = root_dir / "configs" / "inuse"
    (inuse_root / "v1").mkdir(parents=True, exist_ok=True)
    (inuse_root / "v2").mkdir(parents=True, exist_ok=True)

    return {
        "official_models": official_models,
        "official_indexes": official_indexes,
        "uvr5_root": uvr5_root,
    }


def export_model_to_official(
    official_models: Path,
    official_indexes: Path,
    model_path: str,
    index_path: Optional[str]
) -> Tuple[str, Optional[str]]:
    """Copy model/index into official layout and return sid + index path."""
    model_path = Path(model_path)
    sid = f"{model_path.stem}.pth"
    target_model = official_models / sid

    log.detail(f"导出模型到官方目录: {sid}")

    if not target_model.exists() or target_model.stat().st_size != model_path.stat().st_size:
        log.detail(f"复制模型文件: {model_path} -> {target_model}")
        shutil.copy(model_path, target_model)
    else:
        log.detail(f"模型文件已存在，跳过复制")

    target_index_path = None
    resolved_index = _resolve_index_path(model_path, index_path)
    if resolved_index is not None:
        if index_path and Path(index_path).exists():
            log.detail(f"使用指定索引文件: {resolved_index.name}")
        else:
            log.detail(f"自动匹配索引文件: {resolved_index.name}")
        target_index = official_indexes / f"{model_path.stem}.index"
        if not target_index.exists() or target_index.stat().st_size != resolved_index.stat().st_size:
            log.detail(f"复制索引文件: {resolved_index} -> {target_index}")
            shutil.copy(resolved_index, target_index)
        else:
            log.detail("索引文件已存在，跳过复制")
        target_index_path = str(target_index)

    return sid, target_index_path


def _resolve_uvr5_model(uvr5_root: Path, model_name: Optional[str]) -> Optional[str]:
    """Resolve UVR5 model name (without extension)."""
    if model_name:
        stem = model_name.replace(".pth", "").replace(".onnx", "")
        cand_pth = uvr5_root / f"{stem}.pth"
        cand_onnx = uvr5_root / f"{stem}.onnx"
        if cand_pth.exists() or cand_onnx.exists():
            return stem

    for name in os.listdir(uvr5_root):
        if name.endswith(".pth") or "onnx" in name:
            return name.replace(".pth", "").replace(".onnx", "")
    return None


def separate_uvr5(
    input_audio: str,
    temp_dir: Path,
    model_name: Optional[str],
    agg: int = 10,
    fmt: str = "wav"
) -> Tuple[str, str]:
    """Run UVR5 separation and return vocals/ins paths."""
    log.progress("开始UVR5人声分离...")
    log.detail(f"输入音频: {input_audio}")
    log.detail(f"临时目录: {temp_dir}")
    log.config(f"激进度: {agg}, 输出格式: {fmt}")

    setup_official_env(Path(__file__).parent.parent)
    from infer.modules.uvr5.modules import uvr as uvr5_run
    try:
        import ffmpeg  # noqa: F401
    except Exception as e:
        raise ImportError("请先安装 ffmpeg-python") from e
    temp_dir.mkdir(parents=True, exist_ok=True)
    vocals_dir = temp_dir / "vocal"
    ins_dir = temp_dir / "ins"
    vocals_dir.mkdir(parents=True, exist_ok=True)
    ins_dir.mkdir(parents=True, exist_ok=True)

    log.detail(f"人声输出目录: {vocals_dir}")
    log.detail(f"伴奏输出目录: {ins_dir}")

    uvr5_root = Path(os.environ["weight_uvr5_root"])
    model_name = _resolve_uvr5_model(uvr5_root, model_name)
    if not model_name:
        raise FileNotFoundError(
            f"UVR5 模型未找到，请将模型放入: {uvr5_root}"
        )

    log.model(f"使用UVR5模型: {model_name}")

    # Official UVR5 expects a directory input; ensure only one file is present.
    input_dir = temp_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / Path(input_audio).name
    log.detail(f"复制输入文件到: {input_file}")
    shutil.copy2(input_audio, input_file)

    log.progress("正在执行UVR5分离...")
    # generator yields progress
    for progress_info in uvr5_run(
        model_name,
        str(input_dir),
        str(vocals_dir),
        [],
        str(ins_dir),
        agg,
        fmt,
    ):
        if progress_info:
            log.detail(f"UVR5进度: {progress_info}")

    vocal_files = sorted(vocals_dir.glob(f"*.{fmt}"), key=lambda p: p.stat().st_mtime)
    ins_files = sorted(ins_dir.glob(f"*.{fmt}"), key=lambda p: p.stat().st_mtime)
    if not vocal_files or not ins_files:
        raise RuntimeError("UVR5 分离失败，未生成输出文件")

    log.success(f"UVR5分离完成")
    log.audio(f"人声文件: {vocal_files[-1].name}")
    log.audio(f"伴奏文件: {ins_files[-1].name}")

    return str(vocal_files[-1]), str(ins_files[-1])


def convert_vocals_official(
    vocals_path: str,
    output_path: str,
    model_path: str,
    index_path: Optional[str],
    f0_method: str,
    pitch_shift: int,
    index_rate: float,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
    speaker_id: int = 0,
    repair_profile: bool = False,
) -> str:
    """Run official VC pipeline on vocals."""
    rms_mix_rate = float(max(0.0, min(1.0, rms_mix_rate)))
    # Official vc pipeline uses the opposite convention: 1=off, 0=strongest.
    official_rms_mix_rate = 1.0 - rms_mix_rate
    root_dir = Path(__file__).parent.parent
    app_cfg = _load_app_config(root_dir)
    f0_policy = resolve_cover_f0_policy(
        requested_method=f0_method,
        configured_hybrid_mode=str(_get_cfg_value(app_cfg, "f0_hybrid_mode", "off")),
        repair_profile=repair_profile,
    )
    effective_f0_method = f0_policy.vc_method

    log.progress("开始官方VC人声转换...")
    log.detail(f"输入人声: {vocals_path}")
    log.detail(f"输出路径: {output_path}")
    log.model(f"RVC模型: {Path(model_path).name}")
    if index_path:
        log.model(f"索引文件: {Path(index_path).name}")

    log.config(f"F0方法: {f0_method}")
    if effective_f0_method != str(f0_method).strip().lower():
        log.detail(
            "F0路由解析: "
            f"requested={f0_policy.requested_method}, "
            f"vc={f0_policy.vc_method}, "
            f"hybrid_mode={f0_policy.hybrid_mode}, "
            f"gate={f0_policy.gate_method}"
        )
    log.config(f"音调偏移: {pitch_shift} 半音")
    log.config(f"索引率: {index_rate}")
    log.config(f"滤波半径: {filter_radius}")
    log.config(f"RMS混合率: {rms_mix_rate}")
    log.config(f"保护系数: {protect}")
    if repair_profile:
        log.config("唱歌修复: 开启")
    env_paths = setup_official_env(root_dir)

    log.detail("导入官方VC模块...")
    from infer.modules.vc.modules import VC

    sid, official_index = export_model_to_official(
        env_paths["official_models"],
        env_paths["official_indexes"],
        model_path,
        index_path
    )
    log.detail(f"模型SID: {sid}")
    if official_index:
        log.detail(f"官方索引路径: {official_index}")

    log.detail("初始化官方配置...")
    config = OfficialConfig()
    config.disable_chunking = bool(app_cfg.get("disable_chunking", False))
    if "cover" in app_cfg and isinstance(app_cfg["cover"], dict):
        config.disable_chunking = bool(app_cfg["cover"].get("disable_chunking", config.disable_chunking))
    config.f0_min = _to_float(_get_cfg_value(app_cfg, "f0_min", 50), 50)
    config.f0_max = _to_float(_get_cfg_value(app_cfg, "f0_max", 1100), 1100)
    if config.f0_max <= config.f0_min:
        config.f0_max = max(config.f0_min + 1.0, 1100.0)
    # Keep RMVPE extraction aligned with RVC training pitch embedding range.
    # Allowing much wider ranges (e.g. 1600Hz) often tracks higher harmonics
    # instead of the fundamental and introduces synthetic buzzing artifacts.
    if effective_f0_method == "rmvpe":
        if config.f0_min != 50.0 or config.f0_max != 1100.0:
            log.warning(
                "检测到RMVPE F0范围偏离RVC训练范围，已强制使用 50-1100Hz 以避免误跟踪高次谐波"
            )
        config.f0_min = 50.0
        config.f0_max = 1100.0
    config.rmvpe_threshold = _to_float(_get_cfg_value(app_cfg, "rmvpe_threshold", 0.02), 0.02)
    config.f0_energy_threshold_db = _to_float(
        _get_cfg_value(app_cfg, "f0_energy_threshold_db", -50), -50
    )
    config.f0_hybrid_mode = f0_policy.hybrid_mode
    config.crepe_pd_threshold = _to_float(
        _get_cfg_value(app_cfg, "crepe_pd_threshold", 0.1), 0.1
    )
    config.crepe_force_ratio = _to_float(
        _get_cfg_value(app_cfg, "crepe_force_ratio", 0.05), 0.05
    )
    config.crepe_replace_semitones = _to_float(
        _get_cfg_value(app_cfg, "crepe_replace_semitones", 0.0), 0.0
    )
    config.unvoiced_feature_gate_floor = _to_float(
        _get_cfg_value(app_cfg, "unvoiced_feature_gate_floor", 0.28), 0.28
    )
    config.breath_active_margin_db = _to_float(
        _get_cfg_value(app_cfg, "breath_active_margin_db", 52.0), 52.0
    )
    config.f0_stabilize = bool(_get_cfg_value(app_cfg, "f0_stabilize", False))
    config.f0_stabilize_window = int(_get_cfg_value(app_cfg, "f0_stabilize_window", 2))
    config.f0_stabilize_max_semitones = _to_float(
        _get_cfg_value(app_cfg, "f0_stabilize_max_semitones", 6.0), 6.0
    )
    config.f0_stabilize_octave = bool(_get_cfg_value(app_cfg, "f0_stabilize_octave", True))
    config.f0_rate_limit = bool(_get_cfg_value(app_cfg, "f0_rate_limit", False))
    config.f0_rate_limit_semitones = _to_float(
        _get_cfg_value(app_cfg, "f0_rate_limit_semitones", 8.0), 8.0
    )
    if f0_policy.requested_method == "hybrid":
        config.f0_fallback_context_radius = 12
        config.f0_fallback_repair_gap = 6
        config.f0_fallback_post_gap = 4
        config.f0_fallback_use_crepe = True
        config.f0_fallback_crepe_max_ratio = 0.006
        config.f0_fallback_crepe_max_frames = 160
        if not repair_profile:
            config.f0_stabilize = True
            config.f0_rate_limit = True
            config.f0_stabilize_max_semitones = min(
                5.0,
                float(config.f0_stabilize_max_semitones),
            )
            config.f0_rate_limit_semitones = min(
                7.0,
                float(config.f0_rate_limit_semitones),
            )
            log.detail("Hybrid唱歌护栏已启用: F0稳定器 + F0限速")
    if repair_profile:
        config.is_half = False
        config.f0_hybrid_mode = "fallback"
        config.f0_energy_threshold_db = -42.0
        config.unvoiced_feature_gate_floor = max(
            0.32, float(config.unvoiced_feature_gate_floor)
        )
        config.f0_fallback_context_radius = 12
        config.f0_fallback_repair_gap = 6
        config.f0_fallback_post_gap = 4
        config.f0_fallback_use_crepe = True
        config.f0_fallback_crepe_max_ratio = 0.006
        config.f0_fallback_crepe_max_frames = 160
        config.f0_stabilize = True
        config.f0_rate_limit = True
        log.detail("唱歌修复配置已应用: FP32, 更保守F0兜底, F0稳定器, F0限速")
    log.detail(f"设备: {config.device}, 半精度: {config.is_half}")
    log.config(f"F0范围: {config.f0_min}-{config.f0_max}Hz")
    log.config(f"RMVPE阈值: {config.rmvpe_threshold}")
    log.config(f"F0能量阈值: {config.f0_energy_threshold_db}dB")
    log.config(
        f"无声气声特征地板: {config.unvoiced_feature_gate_floor:.2f}, "
        f"呼吸激活边界: ref-{config.breath_active_margin_db:.1f}dB"
    )
    log.config(
        f"F0混合: {config.f0_hybrid_mode}, CREPE阈值: {config.crepe_pd_threshold}, "
        f"强制比率: {config.crepe_force_ratio}, 替换阈值(半音): {config.crepe_replace_semitones}"
    )
    log.config(
        f"F0稳定器: {config.f0_stabilize}, 窗口: {config.f0_stabilize_window}, "
        f"最大跳变(半音): {config.f0_stabilize_max_semitones}, "
        f"八度修正: {config.f0_stabilize_octave}"
    )
    log.config(
        f"F0限速: {config.f0_rate_limit}, 最大跳变/帧(半音): {config.f0_rate_limit_semitones}"
    )

    log.model("初始化VC实例...")
    vc = VC(config)

    log.progress(f"加载模型: {sid}")
    vc.get_vc(sid)

    spk_max = 1
    try:
        if getattr(vc, "cpt", None) is not None:
            spk_max = int(vc.cpt["config"][-3])
    except Exception:
        spk_max = 1
    spk_max = max(1, spk_max)
    spk_id = int(max(0, min(spk_max - 1, int(speaker_id))))
    if spk_id != int(speaker_id):
        log.warning(f"说话人ID超出范围，已自动修正为 {spk_id} (可用范围: 0-{spk_max - 1})")
    log.progress("执行人声转换...")
    log.detail(f"说话人ID: {spk_id}")

    info, (sr, audio) = vc.vc_single(
        spk_id,
        vocals_path,
        pitch_shift,
        None,
        effective_f0_method,
        official_index or "",
        "",
        index_rate,
        filter_radius,
        0,
        official_rms_mix_rate,
        protect,
    )

    if sr is None or audio is None:
        log.error(f"VC转换失败: {info}")
        raise RuntimeError(info)

    log.detail(f"转换信息: {info}")
    log.detail(f"输出采样率: {sr} Hz")
    log.detail(f"输出音频长度: {len(audio)} 样本")

    log.progress(f"保存输出文件: {output_path}")
    sf.write(output_path, audio, sr)

    output_size = Path(output_path).stat().st_size
    log.success(f"官方VC转换完成: {output_path}")
    log.audio(f"输出文件大小: {output_size / 1024 / 1024:.2f} MB")

    return output_path






def _sync_upstream_reference_asset(src: Path, dst: Path, label: str) -> None:
    """Ensure vendored official tree has the same runtime asset expected upstream."""
    if not src.exists():
        raise FileNotFoundError(f"{label} not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
        log.detail(f"同步官方资源: {src.name} -> {dst}")



def setup_upstream_official_env(root_dir: Path) -> dict:
    """Prepare vendored upstream RVC layout and environment."""
    log.detail("准备内置官方 RVC 环境...")
    official_root = root_dir / "_official_rvc"
    if not official_root.exists():
        raise FileNotFoundError(f"Upstream RVC directory not found: {official_root}")

    official_models = official_root / "assets" / "weights"
    official_indexes = official_root / "assets" / "indices"
    official_rmvpe_root = official_root / "assets" / "rmvpe"
    official_hubert_root = official_root / "assets" / "hubert"
    official_uvr5_root = official_root / "assets" / "uvr5_weights"
    official_models.mkdir(parents=True, exist_ok=True)
    official_indexes.mkdir(parents=True, exist_ok=True)
    official_rmvpe_root.mkdir(parents=True, exist_ok=True)
    official_hubert_root.mkdir(parents=True, exist_ok=True)
    official_uvr5_root.mkdir(parents=True, exist_ok=True)

    _sync_upstream_reference_asset(
        root_dir / "assets" / "hubert" / "hubert_base.pt",
        official_hubert_root / "hubert_base.pt",
        "HuBERT model",
    )
    _sync_upstream_reference_asset(
        root_dir / "assets" / "rmvpe" / "rmvpe.pt",
        official_rmvpe_root / "rmvpe.pt",
        "RMVPE model",
    )

    os.environ["weight_root"] = str(official_models)
    os.environ["index_root"] = str(official_indexes)
    os.environ["outside_index_root"] = str(official_indexes)
    os.environ["rmvpe_root"] = str(official_rmvpe_root)
    os.environ["weight_uvr5_root"] = str(official_uvr5_root)

    log.detail(f"官方根目录: {official_root}")
    log.detail(f"官方模型目录: {official_models}")
    log.detail(f"官方索引目录: {official_indexes}")
    log.detail(f"官方RMVPE目录: {official_rmvpe_root}")
    log.detail(f"官方HuBERT目录: {official_hubert_root}")
    log.detail(f"官方UVR5目录: {official_uvr5_root}")

    return {
        "official_root": official_root,
        "official_models": official_models,
        "official_indexes": official_indexes,
        "official_rmvpe_root": official_rmvpe_root,
        "official_hubert_root": official_hubert_root,
        "official_uvr5_root": official_uvr5_root,
    }



def convert_vocals_official_upstream(
    vocals_path: str,
    output_path: str,
    model_path: str,
    index_path: Optional[str],
    f0_method: str,
    pitch_shift: int,
    index_rate: float,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
    speaker_id: int = 0,
) -> str:
    """Run vendored upstream official RVC in an isolated subprocess."""
    root_dir = Path(__file__).parent.parent
    app_cfg = _load_app_config(root_dir)
    f0_policy = resolve_cover_f0_policy(
        requested_method=f0_method,
        configured_hybrid_mode=str(_get_cfg_value(app_cfg, "f0_hybrid_mode", "off")),
        repair_profile=False,
    )
    effective_f0_method = f0_policy.vc_method
    env_paths = setup_upstream_official_env(root_dir)

    sid, official_index = export_model_to_official(
        env_paths["official_models"],
        env_paths["official_indexes"],
        model_path,
        index_path,
    )

    official_rms_mix_rate = 1.0 - float(rms_mix_rate)
    runner_path = root_dir / "infer" / "official_upstream_runner.py"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    def _build_command(selected_index_path: Optional[str], selected_index_rate: float) -> list[str]:
        return [
            sys.executable,
            str(runner_path),
            "--sid",
            sid,
            "--vocals-path",
            str(vocals_path),
            "--output-path",
            str(output_path),
            "--f0-method",
            str(effective_f0_method),
            "--pitch-shift",
            str(int(pitch_shift)),
            "--index-path",
            str(selected_index_path or ""),
            "--index-rate",
            str(float(selected_index_rate)),
            "--filter-radius",
            str(int(filter_radius)),
            "--rms-mix-rate",
            str(float(official_rms_mix_rate)),
            "--protect",
            str(float(protect)),
            "--speaker-id",
            str(int(speaker_id)),
        ]

    command = _build_command(official_index, index_rate)

    log.progress("开始内置官方VC转换...")
    log.detail(f"官方模型SID: {sid}")
    if official_index:
        log.detail(f"官方索引路径: {official_index}")
    if effective_f0_method != str(f0_method).strip().lower():
        log.detail(
            "官方F0路由解析: "
            f"requested={f0_policy.requested_method}, "
            f"vc={f0_policy.vc_method}, "
            f"hybrid_mode={f0_policy.hybrid_mode}, "
            f"gate={f0_policy.gate_method}"
        )
    log.detail(f"官方RMS混合率: {official_rms_mix_rate}")

    used_index_fallback = False
    try:
        subprocess.run(
            command,
            cwd=env_paths["official_root"],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        should_retry_without_index = bool(official_index) and float(index_rate) > 0.0
        if not should_retry_without_index:
            raise RuntimeError(f"内置官方VC转换失败，退出码: {exc.returncode}") from exc

        log.warning("内置官方VC索引推理失败，正在自动禁用索引重试；上方 traceback 来自第一次索引尝试，可忽略。")
        log.detail(f"失败索引: {official_index}")

        output_file = Path(output_path)
        if output_file.exists():
            try:
                output_file.unlink()
                log.detail(f"已删除失败尝试残留输出: {output_path}")
            except OSError as cleanup_exc:
                log.warning(f"清理失败输出文件时出错，继续重试: {cleanup_exc}")

        retry_command = _build_command(None, 0.0)
        log.detail("回退设置: index_path=<empty>, index_rate=0.0")
        try:
            subprocess.run(
                retry_command,
                cwd=env_paths["official_root"],
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as retry_exc:
            raise RuntimeError(
                f"内置官方VC转换失败，索引回退后仍退出码: {retry_exc.returncode}"
            ) from retry_exc
        used_index_fallback = True

    output_file = Path(output_path)
    if not output_file.exists():
        raise RuntimeError(f"内置官方VC未生成输出文件: {output_path}")

    output_size = output_file.stat().st_size
    if used_index_fallback:
        log.warning("内置官方VC检测到异常索引，本次已自动改为无索引完成转换")
    log.success(f"内置官方VC转换完成: {output_path}")
    log.audio(f"输出文件大小: {output_size / 1024 / 1024:.2f} MB")
    return output_path



def _sync_upstream_uvr5_model(root_dir: Path, official_uvr5_root: Path, model_name: Optional[str]) -> str:
    """Copy the selected UVR5 model into vendored official layout and return the stem."""
    source_root = root_dir / "assets" / "uvr5_weights"
    if not source_root.exists():
        raise FileNotFoundError(f"UVR5 模型目录未找到: {source_root}")

    candidates = []
    if model_name:
        stem = model_name.replace('.pth', '').replace('.onnx', '')
        candidates.extend([source_root / f"{stem}.pth", source_root / f"{stem}.onnx", source_root / stem])
    else:
        candidates.extend(sorted(source_root.glob('*.pth')))
        candidates.extend(sorted(source_root.glob('*.onnx')))

    source_model = next((candidate for candidate in candidates if candidate.exists()), None)
    if source_model is None:
        raise FileNotFoundError(f"未找到可用的 UVR5 模型: {model_name or '自动选择'}")

    target_model = official_uvr5_root / source_model.name
    if not target_model.exists() or target_model.stat().st_size != source_model.stat().st_size:
        shutil.copy2(source_model, target_model)
        log.detail(f"同步官方UVR5模型: {source_model.name} -> {target_model}")
    return source_model.stem



def separate_uvr5_official_upstream(
    input_audio: str,
    temp_dir: Path,
    model_name: Optional[str],
    agg: int = 10,
    fmt: str = "wav",
) -> Tuple[str, str]:
    """Run vendored upstream UVR5 separation in an isolated subprocess."""
    root_dir = Path(__file__).parent.parent
    env_paths = setup_upstream_official_env(root_dir)
    resolved_model_name = _sync_upstream_uvr5_model(root_dir, env_paths["official_uvr5_root"], model_name)

    temp_dir.mkdir(parents=True, exist_ok=True)
    input_dir = temp_dir / "input"
    vocals_dir = temp_dir / "vocal"
    ins_dir = temp_dir / "ins"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    vocals_dir.mkdir(parents=True, exist_ok=True)
    ins_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / Path(input_audio).name
    shutil.copy2(input_audio, input_file)

    runner_path = root_dir / "infer" / "official_upstream_uvr_runner.py"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        str(runner_path),
        "--model-name",
        resolved_model_name,
        "--input-dir",
        str(input_dir),
        "--save-root-vocal",
        str(vocals_dir),
        "--save-root-ins",
        str(ins_dir),
        "--agg",
        str(int(agg)),
        "--format",
        str(fmt),
    ]

    log.progress("开始内置官方UVR5分离...")
    log.detail(f"官方UVR5模型: {resolved_model_name}")
    log.detail(f"官方UVR5输入目录: {input_dir}")
    log.detail(f"官方UVR5人声输出: {vocals_dir}")
    log.detail(f"官方UVR5伴奏输出: {ins_dir}")

    try:
        subprocess.run(
            command,
            cwd=env_paths["official_root"],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"内置官方UVR5分离失败，退出码: {exc.returncode}") from exc

    vocal_files = sorted(vocals_dir.glob(f"*.{fmt}"), key=lambda p: p.stat().st_mtime)
    ins_files = sorted(ins_dir.glob(f"*.{fmt}"), key=lambda p: p.stat().st_mtime)
    if not vocal_files or not ins_files:
        raise RuntimeError("内置官方UVR5分离失败，未生成输出文件")

    log.success("内置官方UVR5分离完成")
    log.audio(f"人声文件: {vocal_files[-1].name}")
    log.audio(f"伴奏文件: {ins_files[-1].name}")
    return str(vocal_files[-1]), str(ins_files[-1])
