# -*- coding: utf-8 -*-
"""
翻唱流水线 - 整合人声分离、RVC转换、混音的完整流程
"""
import os
import gc
import uuid
import shutil
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

from infer.separator import (
    VocalSeparator,
    RoformerSeparator,
    ROFORMER_DEFAULT_MODEL,
    check_demucs_available,
    check_roformer_available,
    get_available_models,
)
from infer.official_adapter import (
    setup_official_env,
    separate_uvr5,
    convert_vocals_official,
)
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
        self.rvc_pipeline = None
        self.temp_dir = Path(__file__).parent.parent / "temp" / "cover"

    def _get_session_dir(self, session_id: str = None) -> Path:
        """获取会话临时目录"""
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        session_dir = self.temp_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _init_separator(
        self,
        model_name: str = "htdemucs",
        shifts: int = 2,
        overlap: float = 0.25,
        split: bool = True
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
            ):
                return
            if self.separator is not None:
                self.separator.unload_model()
                self.separator = None
            self.separator = RoformerSeparator(device=self.device)
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
        if f0_method == "rmvpe":
            if not rmvpe_path.exists():
                raise FileNotFoundError(f"RMVPE 模型未找到: {rmvpe_path}")
            gate_pipe.load_f0_extractor("rmvpe", str(rmvpe_path))
        else:
            gate_pipe.load_f0_extractor(f0_method, None)
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
        max_val = np.max(np.abs(mixed))
        if max_val > 0.98:
            mixed = mixed * (0.98 / max_val)

        if output_path is None:
            output_path = str(Path(converted_path).with_suffix("").as_posix() + "_blend.wav")

        sf.write(output_path, mixed, sr)
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
        f0_method: str = "rmvpe",
        demucs_model: str = "htdemucs",
        demucs_shifts: int = 2,
        demucs_overlap: float = 0.25,
        demucs_split: bool = True,
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
        output_dir: Optional[str] = None,
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
            f0_method: F0 提取方法
            demucs_model: Demucs 模型名称
            demucs_shifts: Demucs shifts 参数
            demucs_overlap: Demucs overlap 参数
            demucs_split: Demucs split 参数
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
        total_steps = 4
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
        log.separator()

        # 记录参数配置
        log.config(f"RVC模型: {Path(model_path).name}")
        log.config(f"索引文件: {Path(index_path).name if index_path else '无'}")
        log.config(f"音调偏移: {pitch_shift} 半音")
        log.config(f"F0提取方法: {f0_method}")
        log.config(f"索引混合比率: {index_ratio}")
        log.config(f"人声分离器: {separator}")
        if separator == "uvr5":
            log.config(f"UVR5模型: {uvr5_model or '自动选择'}")
            log.config(f"UVR5激进度: {uvr5_agg}")
        elif separator == "roformer":
            log.config(f"Roformer模型: {ROFORMER_DEFAULT_MODEL}")
        else:
            log.config(f"Demucs模型: {demucs_model}")
            log.config(f"Demucs shifts: {demucs_shifts}")
        log.config(f"人声音量: {vocals_volume}")
        log.config(f"伴奏音量: {accompaniment_volume}")
        log.config(f"混响量: {reverb_amount}")
        log.separator()

        def report_progress(msg: str, step: int):
            if progress_callback:
                progress_callback(msg, step, total_steps)
            log.step(step, total_steps, msg)

        try:
            # ===== 步骤 1: 人声分离 =====
            report_progress("正在分离人声和伴奏...", 1)

            if use_official and separator == "uvr5":
                log.model(f"使用官方UVR5进行人声分离")
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
                log.success(f"UVR5分离完成")
            elif separator == "roformer":
                log.model(f"使用 Mel-Band Roformer 进行人声分离")
                self._init_separator("roformer")
                vocals_path, accompaniment_path = self.separator.separate(
                    input_audio,
                    str(session_dir)
                )
                log.success(f"Mel-Band Roformer 分离完成")
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
                log.success(f"Demucs分离完成")

            # 记录分离结果
            vocals_size = Path(vocals_path).stat().st_size if Path(vocals_path).exists() else 0
            accomp_size = Path(accompaniment_path).stat().st_size if Path(accompaniment_path).exists() else 0
            log.audio(f"人声文件: {Path(vocals_path).name} ({_format_size(vocals_size)})")
            log.audio(f"伴奏文件: {Path(accompaniment_path).name} ({_format_size(accomp_size)})")

            # 释放分离器显存
            if self.separator is not None:
                log.detail("释放分离器显存...")
                self.separator.unload_model()
            gc.collect()
            empty_device_cache()
            log.detail("已清理设备缓存")

            # ===== 步骤 2: RVC 人声转换 =====
            report_progress("正在转换人声...", 2)
            converted_vocals_path = str(session_dir / "converted_vocals.wav")

            log.model(f"加载RVC模型: {Path(model_path).name}")
            log.detail(f"输入人声: {vocals_path}")
            log.detail(f"输出路径: {converted_vocals_path}")

            if use_official:
                log.detail("使用官方VC管道进行转换")
                log.config(f"F0方法: {f0_method}, 音调: {pitch_shift}, 索引率: {index_ratio}")
                log.config(f"滤波半径: {filter_radius}, RMS混合: {rms_mix_rate}, 保护: {protect}")

                convert_vocals_official(
                    vocals_path=vocals_path,
                    output_path=converted_vocals_path,
                    model_path=model_path,
                    index_path=index_path,
                    f0_method=f0_method,
                    pitch_shift=pitch_shift,
                    index_rate=index_ratio,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                )
                if silence_gate:
                    log.detail("启用静音门限(官方VC后处理..)")
                    self._apply_silence_gate_official(
                        vocals_path=vocals_path,
                        converted_path=converted_vocals_path,
                        f0_method=f0_method,
                        silence_threshold_db=silence_threshold_db,
                        silence_smoothing_ms=silence_smoothing_ms,
                        silence_min_duration_ms=silence_min_duration_ms,
                        protect=protect
                    )
                log.success("官方VC转换完成")
            else:
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
                    if f0_method == "rmvpe":
                        if rmvpe_path.exists():
                            log.model(f"加载RMVPE模型: {rmvpe_path}")
                            self.rvc_pipeline.load_f0_extractor("rmvpe", str(rmvpe_path))
                            log.success("RMVPE模型加载完成")
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
                    audio_path=vocals_path,
                    output_path=converted_vocals_path,
                    pitch_shift=pitch_shift,
                    index_ratio=index_ratio,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                    silence_gate=silence_gate,
                    silence_threshold_db=silence_threshold_db,
                    silence_smoothing_ms=silence_smoothing_ms,
                    silence_min_duration_ms=silence_min_duration_ms,
                )
                log.success("自定义VC转换完成")

                log.detail("释放RVC管道资源...")
                self.rvc_pipeline.unload_all()
                gc.collect()
                empty_device_cache()
                log.detail("已清理设备缓存")

            # 记录转换结果
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

            # ===== 步骤 3: 混音 =====
            report_progress("正在混合人声和伴奏...", 3)

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

            cover_size = Path(cover_path).stat().st_size if Path(cover_path).exists() else 0
            log.success(f"混音完成: {_format_size(cover_size)}")

            # ===== 步骤 4: 整理输出 =====
            report_progress("正在整理输出文件...", 4)

            # 如果指定了输出目录，复制文件
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                log.detail(f"输出目录: {output_path}")

                input_name = Path(input_audio).stem
                final_cover = str(output_path / f"{input_name}_cover.wav")
                final_vocals = str(output_path / f"{input_name}_vocals.wav")
                final_converted = str(output_path / f"{input_name}_converted.wav")
                final_accompaniment = str(output_path / f"{input_name}_accompaniment.wav")

                log.detail(f"复制翻唱文件: {final_cover}")
                shutil.copy(cover_path, final_cover)
                log.detail(f"复制原始人声: {final_vocals}")
                shutil.copy(vocals_path, final_vocals)
                log.detail(f"复制转换人声: {final_converted}")
                shutil.copy(converted_vocals_path, final_converted)
                log.detail(f"复制伴奏文件: {final_accompaniment}")
                shutil.copy(accompaniment_path, final_accompaniment)

                result = {
                    "cover": final_cover,
                    "vocals": final_vocals,
                    "converted_vocals": final_converted,
                    "accompaniment": final_accompaniment
                }
            else:
                result = {
                    "cover": cover_path,
                    "vocals": vocals_path,
                    "converted_vocals": converted_vocals_path,
                    "accompaniment": accompaniment_path
                }

            log.separator()
            report_progress("翻唱完成!", 4)
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
