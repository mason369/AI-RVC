# -*- coding: utf-8 -*-
"""
Adapter for official RVC WebUI modules (VC + UVR5).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import soundfile as sf

from configs.config import Config as OfficialConfig
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
    if index_path:
        idx_path = Path(index_path)
        if idx_path.exists():
            target_index = official_indexes / f"{model_path.stem}.index"
            if not target_index.exists() or target_index.stat().st_size != idx_path.stat().st_size:
                log.detail(f"复制索引文件: {idx_path} -> {target_index}")
                shutil.copy(idx_path, target_index)
            else:
                log.detail(f"索引文件已存在，跳过复制")
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
) -> str:
    """Run official VC pipeline on vocals."""
    rms_mix_rate = float(max(0.0, min(1.0, rms_mix_rate)))
    # Official vc pipeline uses the opposite convention: 1=off, 0=strongest.
    official_rms_mix_rate = 1.0 - rms_mix_rate

    log.progress("开始官方VC人声转换...")
    log.detail(f"输入人声: {vocals_path}")
    log.detail(f"输出路径: {output_path}")
    log.model(f"RVC模型: {Path(model_path).name}")
    if index_path:
        log.model(f"索引文件: {Path(index_path).name}")

    log.config(f"F0方法: {f0_method}")
    log.config(f"音调偏移: {pitch_shift} 半音")
    log.config(f"索引率: {index_rate}")
    log.config(f"滤波半径: {filter_radius}")
    log.config(f"RMS混合率: {rms_mix_rate}")
    log.config(f"保护系数: {protect}")

    root_dir = Path(__file__).parent.parent
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
    app_cfg = _load_app_config(root_dir)
    config.disable_chunking = bool(app_cfg.get("disable_chunking", False))
    if "cover" in app_cfg and isinstance(app_cfg["cover"], dict):
        config.disable_chunking = bool(app_cfg["cover"].get("disable_chunking", config.disable_chunking))
    config.f0_min = _to_float(_get_cfg_value(app_cfg, "f0_min", 50), 50)
    config.f0_max = _to_float(_get_cfg_value(app_cfg, "f0_max", 1100), 1100)
    if config.f0_max <= config.f0_min:
        config.f0_max = max(config.f0_min + 1.0, 1100.0)
    config.rmvpe_threshold = _to_float(_get_cfg_value(app_cfg, "rmvpe_threshold", 0.02), 0.02)
    config.f0_energy_threshold_db = _to_float(
        _get_cfg_value(app_cfg, "f0_energy_threshold_db", -50), -50
    )
    config.f0_hybrid_mode = str(_get_cfg_value(app_cfg, "f0_hybrid_mode", "off"))
    config.crepe_pd_threshold = _to_float(
        _get_cfg_value(app_cfg, "crepe_pd_threshold", 0.1), 0.1
    )
    config.crepe_force_ratio = _to_float(
        _get_cfg_value(app_cfg, "crepe_force_ratio", 0.05), 0.05
    )
    config.crepe_replace_semitones = _to_float(
        _get_cfg_value(app_cfg, "crepe_replace_semitones", 0.0), 0.0
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
    log.detail(f"设备: {config.device}, 半精度: {config.is_half}")
    log.config(f"F0范围: {config.f0_min}-{config.f0_max}Hz")
    log.config(f"RMVPE阈值: {config.rmvpe_threshold}")
    log.config(f"F0能量阈值: {config.f0_energy_threshold_db}dB")
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

    spk_id = 0
    log.progress("执行人声转换...")
    log.detail(f"说话人ID: {spk_id}")

    info, (sr, audio) = vc.vc_single(
        spk_id,
        vocals_path,
        pitch_shift,
        None,
        f0_method,
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



