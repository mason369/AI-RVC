# -*- coding: utf-8 -*-
"""
Gradio 界面 - RVC 语音转换
"""
import os
import json
import tempfile
import gradio as gr
from pathlib import Path
from typing import Optional, Tuple

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 加载语言包
def load_i18n(lang: str = "zh_CN") -> dict:
    """加载语言包"""
    i18n_path = ROOT_DIR / "i18n" / f"{lang}.json"
    if i18n_path.exists():
        with open(i18n_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 加载配置
def load_config() -> dict:
    """加载配置"""
    config_path = ROOT_DIR / "configs" / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 全局变量
i18n = load_i18n()
config = load_config()
pipeline = None


def get_text(key: str, section: str = None) -> str:
    """获取翻译文本"""
    if section:
        return i18n.get(section, {}).get(key, key)
    return i18n.get(key, key)


def init_pipeline():
    """初始化推理管道"""
    global pipeline

    if pipeline is not None:
        return pipeline

    from infer.pipeline import VoiceConversionPipeline

    device = config.get("device", "cuda")
    pipeline = VoiceConversionPipeline(device=device)

    # 加载 HuBERT
    hubert_path = ROOT_DIR / config.get("hubert_path", "assets/hubert/hubert_base.pt")
    if hubert_path.exists():
        pipeline.load_hubert(str(hubert_path))

    # 加载 F0 提取器
    rmvpe_path = ROOT_DIR / config.get("rmvpe_path", "assets/rmvpe/rmvpe.pt")
    if rmvpe_path.exists():
        pipeline.load_f0_extractor("rmvpe", str(rmvpe_path))

    return pipeline


def get_model_list() -> list:
    """获取模型列表"""
    from infer.pipeline import list_voice_models

    weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")
    models = list_voice_models(str(weights_dir))

    return [m["name"] for m in models]


def get_model_info(model_name: str) -> dict:
    """获取模型信息"""
    from infer.pipeline import list_voice_models

    weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")
    models = list_voice_models(str(weights_dir))

    for m in models:
        if m["name"] == model_name:
            return m

    return None


def convert_voice(
    audio_path: str,
    model_name: str,
    pitch_shift: float,
    f0_method: str,
    index_ratio: float,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float
) -> Tuple[Optional[str], str]:
    """
    执行语音转换

    Returns:
        Tuple[音频路径, 状态消息]
    """
    # 验证输入
    if audio_path is None:
        return None, get_text("no_audio", "errors")

    if not model_name:
        return None, get_text("no_model", "errors")

    try:
        # 初始化管道
        pipe = init_pipeline()

        # 加载语音模型
        model_info = get_model_info(model_name)
        if model_info is None:
            return None, get_text("model_not_found", "errors")

        pipe.load_voice_model(model_info["model_path"])

        # 加载索引 (如果存在)
        if model_info.get("index_path"):
            pipe.load_index(model_info["index_path"])

        # 创建输出目录
        output_dir = ROOT_DIR / config.get("output_dir", "output")
        output_dir.mkdir(exist_ok=True)

        # 生成输出文件名
        input_name = Path(audio_path).stem
        output_path = output_dir / f"{input_name}_{model_name}.wav"

        # 执行转换
        result_path = pipe.convert(
            audio_path=audio_path,
            output_path=str(output_path),
            pitch_shift=pitch_shift,
            index_ratio=index_ratio,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect
        )

        return result_path, get_text("success", "convert")

    except Exception as e:
        error_msg = get_text("conversion_failed", "errors").format(error=str(e))
        return None, error_msg


def download_base_models() -> str:
    """下载基础模型"""
    from tools.download_models import download_required_models

    try:
        success = download_required_models()
        if success:
            return get_text("download_success", "models")
        else:
            return "下载过程中出现错误，请检查网络连接"
    except Exception as e:
        return f"下载失败: {str(e)}"


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    with gr.Blocks(
        title=get_text("title"),
        theme=gr.themes.Soft()
    ) as app:

        gr.Markdown(f"# {get_text('title')}")

        with gr.Tabs():
            # ===== 语音转换标签页 =====
            with gr.Tab(get_text("convert", "tabs")):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 输入音频
                        input_audio = gr.Audio(
                            label=get_text("input_audio", "convert"),
                            type="filepath"
                        )

                        # 模型选择
                        model_dropdown = gr.Dropdown(
                            label=get_text("select_model", "convert"),
                            choices=get_model_list(),
                            interactive=True
                        )

                        # 刷新模型列表按钮
                        refresh_btn = gr.Button(
                            get_text("refresh_btn", "models"),
                            size="sm"
                        )

                    with gr.Column(scale=1):
                        # 参数设置
                        pitch_shift = gr.Slider(
                            label=get_text("pitch_shift", "convert"),
                            minimum=-12,
                            maximum=12,
                            value=0,
                            step=1,
                            info=get_text("pitch_shift_desc", "convert")
                        )

                        f0_method = gr.Dropdown(
                            label=get_text("f0_method", "convert"),
                            choices=["rmvpe", "pm", "harvest", "crepe"],
                            value="rmvpe",
                            info=get_text("f0_method_desc", "convert")
                        )

                        index_ratio = gr.Slider(
                            label=get_text("index_ratio", "convert"),
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.1,
                            info=get_text("index_ratio_desc", "convert")
                        )

                        filter_radius = gr.Slider(
                            label=get_text("filter_radius", "convert"),
                            minimum=0,
                            maximum=7,
                            value=3,
                            step=1,
                            info=get_text("filter_radius_desc", "convert")
                        )

                        rms_mix_rate = gr.Slider(
                            label=get_text("rms_mix_rate", "convert"),
                            minimum=0,
                            maximum=1,
                            value=0.25,
                            step=0.05,
                            info=get_text("rms_mix_rate_desc", "convert")
                        )

                        protect = gr.Slider(
                            label=get_text("protect", "convert"),
                            minimum=0,
                            maximum=0.5,
                            value=0.33,
                            step=0.01,
                            info=get_text("protect_desc", "convert")
                        )

                # 转换按钮
                convert_btn = gr.Button(
                    get_text("convert_btn", "convert"),
                    variant="primary"
                )

                # 输出
                with gr.Row():
                    output_audio = gr.Audio(
                        label=get_text("output_audio", "convert"),
                        type="filepath"
                    )
                    status_text = gr.Textbox(
                        label="状态",
                        interactive=False
                    )

                # 事件绑定
                convert_btn.click(
                    fn=convert_voice,
                    inputs=[
                        input_audio,
                        model_dropdown,
                        pitch_shift,
                        f0_method,
                        index_ratio,
                        filter_radius,
                        rms_mix_rate,
                        protect
                    ],
                    outputs=[output_audio, status_text]
                )

                refresh_btn.click(
                    fn=lambda: gr.update(choices=get_model_list()),
                    outputs=[model_dropdown]
                )

            # ===== 模型管理标签页 =====
            with gr.Tab(get_text("models", "tabs")):
                gr.Markdown(f"### {get_text('available_models', 'models')}")

                model_list_display = gr.Dataframe(
                    headers=["模型名称", "模型路径", "索引路径"],
                    interactive=False
                )

                with gr.Row():
                    refresh_models_btn = gr.Button(
                        get_text("refresh_btn", "models")
                    )
                    download_btn = gr.Button(
                        get_text("download_btn", "models"),
                        variant="primary"
                    )

                download_status = gr.Textbox(
                    label="下载状态",
                    interactive=False
                )

                def refresh_model_table():
                    from infer.pipeline import list_voice_models
                    weights_dir = ROOT_DIR / config.get("weights_dir", "assets/weights")
                    models = list_voice_models(str(weights_dir))
                    data = [[m["name"], m["model_path"], m.get("index_path", "")] for m in models]
                    return data

                refresh_models_btn.click(
                    fn=refresh_model_table,
                    outputs=[model_list_display]
                )

                download_btn.click(
                    fn=download_base_models,
                    outputs=[download_status]
                )

            # ===== 设置标签页 =====
            with gr.Tab(get_text("settings", "tabs")):
                device_radio = gr.Radio(
                    label=get_text("device", "settings"),
                    choices=[
                        (get_text("cuda", "settings"), "cuda"),
                        (get_text("cpu", "settings"), "cpu")
                    ],
                    value=config.get("device", "cuda")
                )

                save_settings_btn = gr.Button(
                    get_text("save_btn", "settings"),
                    variant="primary"
                )

                settings_status = gr.Textbox(
                    label="状态",
                    interactive=False
                )

                def save_settings(device):
                    global config
                    config["device"] = device

                    config_path = ROOT_DIR / "configs" / "config.json"
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4, ensure_ascii=False)

                    return get_text("saved", "settings")

                save_settings_btn.click(
                    fn=save_settings,
                    inputs=[device_radio],
                    outputs=[settings_status]
                )

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """启动 Gradio 界面"""
    app = create_ui()
    app.launch(
        server_name=host,
        server_port=port,
        share=share
    )


if __name__ == "__main__":
    launch()
