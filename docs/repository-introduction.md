# AI-RVC 仓库介绍与搜索可见性文案

这份文档用于配置 GitHub About、仓库 topics、Hugging Face Space 简介或第三方项目介绍页。目标是让搜索引擎和 GitHub 主题页更容易理解：AI-RVC 是一个围绕 **AI 翻唱、RVC v2 声音转换、人声分离、角色声线模型和混音合成** 的项目。

## GitHub About 简介

中文推荐版：

> 一键 AI 翻唱与 RVC v2 声音转换 WebUI：自动人声分离、HuBERT + RMVPE + FAISS 音色转换、角色模型下载、混音预设，并支持 Windows、Linux、WSL2、Google Colab 和 Hugging Face Spaces。

English version:

> One-click AI cover and RVC v2 voice conversion WebUI with vocal separation, HuBERT + RMVPE + FAISS inference, character model downloads, mixing presets, and support for Windows, Linux, WSL2, Google Colab, and Hugging Face Spaces.

## GitHub Topics

GitHub topics 建议使用小写字母、数字和连字符。下面这组控制在 20 个以内，便于直接粘贴到仓库 About 面板：

`rvc`, `rvc-v2`, `voice-conversion`, `ai-cover`, `song-cover`, `singing-voice-conversion`, `voice-changer`, `voice-cloning`, `vocal-separation`, `audio-separation`, `rmvpe`, `hubert`, `faiss`, `gradio`, `pytorch`, `colab`, `uvr`, `demucs`, `roformer`, `ai-music`

## 搜索摘要

短版：

> AI-RVC 是一个基于 RVC v2 的一键 AI 翻唱工具，自动完成人声分离、角色声线转换、音高提取和混音合成，支持本地 WebUI、Google Colab 与 Hugging Face Spaces。

长版：

> AI-RVC 面向想做 AI cover、RVC 翻唱和角色声线转换的用户。项目把歌曲处理流程串成一条完整流水线：先用 `audio-separator` / RoFormer 分离人声与伴奏，再通过 HuBERT、RMVPE、FAISS 和 RVC v2 模型转换主唱音色，最后用混音预设生成完整翻唱作品。它提供中文 Gradio WebUI、117 个可下载角色模型、卡拉OK分离、VC 预处理、双 VC 管道，并支持 Windows、Linux、WSL2、Google Colab 和 Hugging Face Spaces。

## 关键词组合

中文关键词：

AI 翻唱、RVC 翻唱、RVC v2 声音转换、AI 声音转换、AI cover、角色声线转换、人声分离、伴奏分离、卡拉OK分离、AI 歌曲翻唱、Gradio WebUI、Colab AI 翻唱、HuBERT、RMVPE、FAISS

English keywords:

AI cover generator, RVC voice conversion, Retrieval-based Voice Conversion, RVC v2, singing voice conversion, voice changer, voice cloning, vocal separation, audio separation, karaoke separation, Gradio WebUI, Google Colab AI cover, HuBERT, RMVPE, FAISS

## 写法原则

- 用真实功能词做关键词，不堆无关热词。
- 仓库简介先说清楚“做什么”，再说“用什么技术”，最后说“在哪些平台可用”。
- README 首页保留原有技术结构，搜索关键词只放在简介、功能点和仓库介绍中自然出现。
- 不建议加入与项目无关的关键词，例如“风扇自动控制”。这类页面可以参考它们的一句话简介和 topics 写法，但不能把无关词塞进 AI-RVC 仓库，否则搜索命中会变脏，读者也容易误会项目用途。
