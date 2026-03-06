import os
import traceback
import logging

logger = logging.getLogger(__name__)

import ffmpeg
import torch

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

# 导入彩色日志
try:
    from lib.logger import log
except ImportError:
    log = None

config = Config()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        if log:
            log.progress(f"开始UVR5人声分离...")
            log.model(f"模型: {model_name}")
            log.detail(f"输入目录: {inp_root}")
            log.detail(f"人声输出: {save_root_vocal}")
            log.detail(f"伴奏输出: {save_root_ins}")
            log.config(f"激进度: {agg}, 格式: {format0}")

        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            if log:
                log.model("加载MDXNet去混响模型...")
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            if log:
                log.model(f"加载VR模型: {func.__name__}")
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )
        is_hp3 = "HP3" in model_name
        if log:
            log.detail(f"HP3模式: {is_hp3}")

        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]

        if log:
            log.detail(f"待处理文件数: {len(paths)}")

        for idx, path in enumerate(paths):
            if log:
                log.progress(f"处理文件 {idx+1}/{len(paths)}: {os.path.basename(path)}")

            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                channels = info["streams"][0]["channels"]
                sample_rate = info["streams"][0]["sample_rate"]
                if log:
                    log.audio(f"音频信息: {channels}声道, {sample_rate}Hz")

                if (
                    channels == 2
                    and sample_rate == "44100"
                ):
                    need_reformat = 0
                    if log:
                        log.detail("格式符合要求，直接处理")
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
                if log:
                    log.warning("无法探测音频格式，将进行重格式化")

            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                if log:
                    log.detail(f"重格式化音频: {tmp_path}")
                os.system(
                    'ffmpeg -i "%s" -vn -acodec pcm_s16le -ac 2 -ar 44100 "%s" -y'
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    if log:
                        log.progress("执行人声分离...")
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                if log:
                    log.success(f"{os.path.basename(inp_path)} 处理成功")
                yield "\n".join(infos)
            except:
                try:
                    if done == 0:
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    if log:
                        log.success(f"{os.path.basename(inp_path)} 处理成功(重试)")
                    yield "\n".join(infos)
                except:
                    error_msg = traceback.format_exc()
                    infos.append(
                        "%s->%s" % (os.path.basename(inp_path), error_msg)
                    )
                    if log:
                        log.error(f"{os.path.basename(inp_path)} 处理失败:\n{error_msg}")
                    yield "\n".join(infos)
    except:
        error_msg = traceback.format_exc()
        infos.append(error_msg)
        if log:
            log.error(f"UVR5处理失败:\n{error_msg}")
        yield "\n".join(infos)
    finally:
        try:
            if log:
                log.detail("清理模型资源...")
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
            if log:
                log.detail("已清理CUDA缓存")

    if log:
        log.success("UVR5处理完成")
    yield "\n".join(infos)
