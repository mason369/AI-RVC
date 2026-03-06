# -*- coding: utf-8 -*-
"""
角色模型管理 - 从 HuggingFace 下载 RVC 角色模型
"""
import os
import json
import re
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Callable

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# HuggingFace 仓库配置
HF_REPO_ID = "trioskosmos/rvc_models"
_VERSION_NOTE_CACHE: Dict[str, Optional[str]] = {}
_VERSION_NOTE_CACHE_LOADED = False


def _get_hf_token() -> Optional[str]:
    """获取 HuggingFace Token（支持 HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HUGGINGFACE_TOKEN）"""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

# 作品归类（用于 UI 分类筛选）
SERIES_ALIASES = {
    "Love Live!": "Love Live!",
    "Love Live! Sunshine!!": "Love Live!",
    "Love Live! Superstar!!": "Love Live!",
    "Love Live! 虹咲学园": "Love Live!",
    "Love Live! 虹咲学園": "Love Live!",
    "Love Live! 莲之空女学院学园偶像俱乐部": "Love Live!",
    "Love Live! Sunshine!! / 幻日夜羽": "Love Live!",
    "Hololive Japan": "Hololive",
    "Hololive English": "Hololive",
    "Hololive Indonesia": "Hololive",
    "崩坏：星穹铁道": "崩坏系列",
    "崩坏3rd": "崩坏系列",
    "偶像大师 灰姑娘女孩": "偶像大师",
}


def normalize_series(source: str) -> str:
    """将来源归类到系列"""
    if not source:
        return "未知"
    for key, series in SERIES_ALIASES.items():
        if source.startswith(key):
            return series
    return source


def _get_display_name(info: Dict, fallback: str) -> str:
    """拼接中文名 / 英文名 / 日文名用于展示"""
    zh_name = info.get("zh_name") or info.get("description") or fallback
    en_name = info.get("en_name")
    jp_name = info.get("jp_name")
    parts = [zh_name]
    if en_name and en_name != zh_name:
        parts.append(en_name)
    if jp_name and jp_name != zh_name and jp_name != en_name:
        parts.append(jp_name)
    display = " / ".join(parts)
    variant = info.get("variant")
    if variant:
        display = f"{display} - {variant}"
    variant_note = _get_version_note(fallback, info)
    if variant_note:
        display = f"{display} ({variant_note})"
    return display


def _find_index_file(pth_file: Path) -> Optional[Path]:
    """尝试找到对应的索引文件"""
    candidate = pth_file.with_suffix(".index")
    if candidate.exists():
        return candidate
    for idx in pth_file.parent.glob("*.index"):
        if idx.stem.lower() == pth_file.stem.lower():
            return idx
    return None


def _safe_print(message: str):
    """避免控制台编码问题导致的输出异常"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("utf-8", "backslashreplace").decode("utf-8"))


def _load_version_note_cache():
    global _VERSION_NOTE_CACHE_LOADED
    if _VERSION_NOTE_CACHE_LOADED:
        return
    _VERSION_NOTE_CACHE_LOADED = True
    try:
        cache_path = get_character_models_dir() / "_version_notes.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _VERSION_NOTE_CACHE.update(data)
    except Exception:
        pass


def _save_version_note_cache():
    try:
        cache_path = get_character_models_dir() / "_version_notes.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(_VERSION_NOTE_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _normalize_note(note: str) -> Optional[str]:
    if not note:
        return None
    # 取首行，移除链接与多余空白
    line = note.strip().splitlines()[0].strip()
    if not line:
        return None
    line = re.sub(r"https?://\S+", "", line).strip()
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return None
    if len(line) > 60:
        return line[:57] + "..."
    return line


def _note_from_metadata(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    parts = []
    title = str(data.get("title") or "")
    desc = str(data.get("description") or "")
    type_val = str(data.get("type") or data.get("version") or "")
    text = f"{title}\n{desc}"

    epoch_match = re.search(r"(\d+)\s*epoch", text, re.IGNORECASE)
    if epoch_match:
        parts.append(f"{epoch_match.group(1)} epochs")

    if type_val:
        t = type_val.lower()
        if t.startswith("v"):
            parts.append(f"RVC {type_val}")
        elif "rvc" in t:
            parts.append(type_val)

    if parts:
        return " · ".join(dict.fromkeys(parts))

    # fallback to title/description first line
    return _normalize_note(desc) or _normalize_note(title)


def _note_from_pth(path: Path) -> Optional[str]:
    try:
        import torch
    except Exception:
        return None

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    parts = []
    info = obj.get("info")
    if isinstance(info, str):
        match = re.search(r"(\d+)\s*epoch", info, re.IGNORECASE)
        if match:
            parts.append(f"{match.group(1)} epochs")

    sr = obj.get("sr")
    if sr:
        if isinstance(sr, (int, float)):
            sr_note = f"{int(sr/1000)}k" if sr >= 1000 else str(int(sr))
        else:
            sr_note = str(sr)
        parts.append(sr_note)

    if parts:
        return " · ".join(dict.fromkeys(parts))
    return None


def _note_from_filename(name: str) -> Optional[str]:
    if not name:
        return None
    lower = name.lower()
    parts = []

    if "rmvpe" in lower:
        parts.append("RMVPE")
    if "ov2" in lower or "ov2super" in lower:
        parts.append("OV2")
    if "pre-anime" in lower or "preanime" in lower:
        parts.append("预TV")
    # 训练轮次与步数
    epoch_match = re.search(r"(?:^|[_-])e(\d{2,5})(?:[_-]|$)", lower)
    if epoch_match:
        parts.append(f"e{epoch_match.group(1)}")
    step_match = re.search(r"(?:^|[_-])s(\d{3,7})(?:[_-]|$)", lower)
    if step_match:
        parts.append(f"s{step_match.group(1)}")

    if parts:
        return " · ".join(dict.fromkeys(parts))
    return None


def _get_version_note(name: str, info: Dict) -> Optional[str]:
    _load_version_note_cache()
    note = info.get("variant_note")
    if note:
        return _normalize_note(note)

    cached = _VERSION_NOTE_CACHE.get(name)
    if cached is not None:
        return cached

    # 1) 读取本地模型目录中的 metadata / info
    char_dir = get_character_models_dir() / name
    if char_dir.exists():
        for candidate in ("metadata.json", "model_info.json", "info.json"):
            path = char_dir / candidate
            if path.exists():
                note = _note_from_metadata(path)
                if note:
                    _VERSION_NOTE_CACHE[name] = note
                    _save_version_note_cache()
                    return note

        # 2) 尝试从本地权重读取 info/版本
        pth_files = sorted(char_dir.glob("*.pth"))
        if pth_files:
            note = _note_from_pth(pth_files[0]) or _note_from_filename(pth_files[0].name)
            if note:
                _VERSION_NOTE_CACHE[name] = note
                _save_version_note_cache()
                return note
        index_files = sorted(char_dir.glob("*.index"))
        if index_files:
            note = _note_from_filename(index_files[0].name)
            if note:
                _VERSION_NOTE_CACHE[name] = note
                _save_version_note_cache()
                return note

    # 3) 未下载时，从配置文件名解析
    file_name = info.get("file") or ""
    note = _note_from_filename(Path(str(file_name)).name)
    if not note:
        files = info.get("files") or []
        if files:
            note = _note_from_filename(Path(str(files[0])).name)

    if not note and info.get("variant"):
        note = "未提供版本说明"

    _VERSION_NOTE_CACHE[name] = note
    _save_version_note_cache()
    return note


def refresh_version_notes(force: bool = False) -> Dict[str, Optional[str]]:
    """批量读取本地模型的版本说明，写入缓存文件"""
    if force:
        _VERSION_NOTE_CACHE.clear()
        # 避免读取旧缓存文件
        global _VERSION_NOTE_CACHE_LOADED
        _VERSION_NOTE_CACHE_LOADED = True
    notes = {}
    for name in CHARACTER_MODELS.keys():
        notes[name] = _get_version_note(name, CHARACTER_MODELS.get(name, {}))
    return notes


def _get_confirm_token(response) -> Optional[str]:
    """获取 Google Drive 下载确认 token"""
    import re
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    try:
        # 兼容通过页面确认下载的场景
        match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def _download_gdrive_file(file_id: str, dest_path: Path) -> bool:
    """下载 Google Drive 文件（支持大文件确认）"""
    import requests

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True, timeout=30)
    token = _get_confirm_token(response)
    if token:
        response = session.get(
            url, params={"id": file_id, "confirm": token}, stream=True, timeout=30
        )

    if response.status_code != 200:
        print(f"  下载失败: HTTP {response.status_code}")
        return False
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        print("  下载失败: 需要手动确认或无访问权限")
        return False

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # 简单进度输出
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    if int(percent) % 10 == 0:
                        print(f"  下载进度: {percent:.0f}%")

    return dest_path.exists() and dest_path.stat().st_size > 0


# 角色模型列表
CHARACTER_MODELS = {
    "Aimi": {
        "file": "Aimi.zip",
        "zh_name": "爱美",
        "en_name": "Aimi",
        "jp_name": "愛美",
        "source": "原创角色"
    },
    "hanayo": {
        "file": "kubotayurika.zip",
        "zh_name": "小泉花阳",
        "en_name": "Hanayo Koizumi",
        "jp_name": "小泉花陽",
        "source": "Love Live!"
    },
    "rin": {
        "file": "rin.zip",
        "zh_name": "星空凛",
        "en_name": "Rin Hoshizora",
        "jp_name": "星空凛",
        "source": "Love Live!"
    },
    "umi": {
        "file": "umi.zip",
        "zh_name": "园田海未",
        "en_name": "Umi Sonoda",
        "jp_name": "園田海未",
        "source": "Love Live!"
    },
    "nozomi": {
        "file": "nozomi.zip",
        "zh_name": "东条希",
        "en_name": "Nozomi Tojo",
        "jp_name": "東條希",
        "source": "Love Live!"
    },
    "dia": {
        "file": "dia.zip",
        "zh_name": "黑泽黛雅",
        "en_name": "Dia Kurosawa",
        "jp_name": "黒澤ダイヤ",
        "source": "Love Live! Sunshine!!"
    },
    "ayumu": {
        "file": "ayumu.zip",
        "zh_name": "上原步梦",
        "en_name": "Ayumu Uehara",
        "jp_name": "上原歩夢",
        "source": "Love Live! 虹咲学园"
    },
    "chika": {
        "file": "TakamiChika.zip",
        "zh_name": "高海千歌",
        "en_name": "Chika Takami",
        "jp_name": "高海千歌",
        "source": "Love Live! Sunshine!!",
        "repo": "Icchan/LoveLive"
    },
    "riko": {
        "file": "SakurauchiRiko.zip",
        "zh_name": "樱内梨子",
        "en_name": "Riko Sakurauchi",
        "jp_name": "桜内梨子",
        "source": "Love Live! Sunshine!!",
        "repo": "Icchan/LoveLive"
    },
    "mari": {
        "file": "Mari Ohara (Love Live! Sunshine!!) - Weights.gg Model.zip",
        "zh_name": "小原鞠莉",
        "en_name": "Mari Ohara",
        "jp_name": "小原鞠莉",
        "source": "Love Live! Sunshine!!",
        "repo": "ChocoKat/Mari_Ohara"
    },
    "yohane": {
        "file": "Yoshiko Tsushima.zip",
        "zh_name": "津岛善子（夜羽）",
        "en_name": "Yoshiko Tsushima (Yohane)",
        "jp_name": "ヨハネ",
        "source": "Love Live! Sunshine!! / 幻日夜羽",
        "repo": "HarunaKasuga/YoshikoTsushima"
    },
    "nico": {
        "url": "https://huggingface.co/Zurakichi/RVC/resolve/main/Models/Love%20Live/V2/%C2%B5%27s/NicoYazawa.zip",
        "filename": "NicoYazawa.zip",
        "zh_name": "矢泽妮可",
        "en_name": "Nico Yazawa",
        "jp_name": "矢澤にこ",
        "source": "Love Live!",
        "repo": "Zurakichi/RVC"
    },
    "kanata": {
        "file": "Models/Love Live/V2/Nijigasaki/KanataKonoe.zip",
        "zh_name": "近江彼方",
        "en_name": "Kanata Konoe",
        "jp_name": "近江彼方",
        "source": "Love Live! 虹咲学园",
        "repo": "Zurakichi/RVC"
    },
    "setsuna_yuki_nana": {
        "file": "Models/Love Live/V2/Nijigasaki/SetsunaYuki.zip",
        "zh_name": "优木雪菜",
        "en_name": "Setsuna Yuki",
        "jp_name": "優木せつ菜",
        "variant": "Zurakichi v2",
        "source": "Love Live! 虹咲学园",
        "repo": "Zurakichi/RVC"
    },
    "hanamaru_zurakichi": {
        "file": "Models/Love Live/V2/Aqours/HanamaruKunikida.zip",
        "zh_name": "国木田花丸",
        "en_name": "Hanamaru Kunikida",
        "jp_name": "国木田花丸",
        "variant": "Zurakichi v2",
        "source": "Love Live! Sunshine!!",
        "repo": "Zurakichi/RVC"
    },
    "kanan": {
        "gdrive_id": "16dPSDGb3ciLsy1HtEXG2OSPhU7BuCty_",
        "filename": "Kanan_Matsuura.zip",
        "zh_name": "松浦果南",
        "en_name": "Kanan Matsuura",
        "jp_name": "松浦果南",
        "source": "Love Live! Sunshine!!"
    },
    "yu_takasaki": {
        "gdrive_id": "1xjIG_bsBzOTOwghGaLSMnO_vL2GgMPNw",
        "filename": "Yu_Takasaki.zip",
        "zh_name": "高咲侑",
        "en_name": "Yu Takasaki",
        "jp_name": "高咲侑",
        "source": "Love Live! 虹咲学园"
    },
    "shizuku_osaka": {
        "url": "https://mega.nz/file/UbZDEaRY#YnxExpDIJzh-rEDfEo2khTPAH1p6GZ5FzaMCfWdUQ34",
        "zh_name": "樱坂雫",
        "en_name": "Shizuku Osaka",
        "jp_name": "桜坂しずく",
        "source": "Love Live! 虹咲学园"
    },
    "sarah_kazuno": {
        "file": "SarahKazuno2.zip",
        "zh_name": "鹿角圣良",
        "en_name": "Sarah Kazuno",
        "jp_name": "鹿角聖良",
        "variant": "v2",
        "source": "Love Live! Sunshine!!",
        "repo": "thebuddyadrian/RVC_Models"
    },
    "leah_kazuno": {
        "file": "LeahKazuno2.zip",
        "zh_name": "鹿角理亚",
        "en_name": "Leah Kazuno",
        "jp_name": "鹿角理亞",
        "variant": "v2",
        "source": "Love Live! Sunshine!!",
        "repo": "thebuddyadrian/RVC_Models"
    },
    "eli": {
        "file": "AyaseEli.zip",
        "zh_name": "绚濑绘里",
        "en_name": "Eli Ayase",
        "jp_name": "絢瀬絵里",
        "source": "Love Live!",
        "repo": "Icchan/LoveLive"
    },
    "you": {
        "file": "WatanabeYou.zip",
        "zh_name": "渡边曜",
        "en_name": "You Watanabe",
        "jp_name": "渡辺曜",
        "source": "Love Live! Sunshine!!",
        "repo": "Icchan/LoveLive"
    },
    "honoka": {
        "files": ["weights/Honoka.pth", "weights/honoka.index"],
        "zh_name": "高坂穗乃果",
        "en_name": "Honoka Kosaka",
        "jp_name": "高坂穂乃果",
        "source": "Love Live!",
        "repo": "trioskosmos/rvc_models"
    },
    "kotori": {
        "files": ["weights/Kotori.pth", "weights/kotori.index"],
        "zh_name": "南小鸟",
        "en_name": "Kotori Minami",
        "jp_name": "南ことり",
        "source": "Love Live!",
        "repo": "trioskosmos/rvc_models"
    },
    "maki": {
        "files": ["weights2/Maki.pth", "weights2/maki.index"],
        "zh_name": "西木野真姬",
        "en_name": "Maki Nishikino",
        "jp_name": "西木野真姫",
        "source": "Love Live!",
        "repo": "trioskosmos/rvc_models"
    },
    "ruby": {
        "files": ["weights2/Ruby.pth", "weights2/ruby.index"],
        "zh_name": "黑泽露比",
        "en_name": "Ruby Kurosawa",
        "jp_name": "黒澤ルビィ",
        "source": "Love Live! Sunshine!!",
        "repo": "trioskosmos/rvc_models"
    },
    "kasumi": {
        "files": ["weights/Kasumi.pth", "weights/kasumi.index"],
        "zh_name": "中须霞",
        "en_name": "Kasumi Nakasu",
        "jp_name": "中須かすみ",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "karin": {
        "files": ["weights/Karin.pth", "weights/karin.index"],
        "zh_name": "朝香果林",
        "en_name": "Karin Asaka",
        "jp_name": "朝香果林",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "rina": {
        "files": ["weights2/Rina.pth", "weights2/rina.index"],
        "zh_name": "天王寺璃奈",
        "en_name": "Rina Tennoji",
        "jp_name": "天王寺璃奈",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "lanzhu": {
        "files": ["weights/Lanzhu.pth", "weights/lanzhu.index"],
        "zh_name": "钟岚珠",
        "en_name": "Lanzhu Zhong",
        "jp_name": "鐘嵐珠",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "keke": {
        "files": ["weights/Keke.pth", "weights/keke.index"],
        "zh_name": "唐可可",
        "en_name": "Keke Tang",
        "jp_name": "唐可可",
        "source": "Love Live! Superstar!!",
        "repo": "trioskosmos/rvc_models"
    },
    "ai_miyashita": {
        "file": "AiMiyashitaV2.zip",
        "zh_name": "宫下爱",
        "en_name": "Ai Miyashita",
        "jp_name": "宮下愛",
        "source": "Love Live! 虹咲学园",
        "repo": "0xMifune/LoveLive"
    },
    "emma_verde": {
        "file": "EmmaVerdeV2.zip",
        "zh_name": "艾玛·维尔德",
        "en_name": "Emma Verde",
        "jp_name": "エマ・ヴェルデ",
        "source": "Love Live! 虹咲学园",
        "repo": "0xMifune/LoveLive"
    },
    "shioriko_mifune": {
        "file": "ShiorikoMifuneV2.zip",
        "zh_name": "三船栞子",
        "en_name": "Shioriko Mifune",
        "jp_name": "三船栞子",
        "source": "Love Live! 虹咲学园",
        "repo": "0xMifune/LoveLive"
    },
    "chisato_arashi": {
        "file": "ChisatoArashiV2.zip",
        "zh_name": "岚千砂都",
        "en_name": "Chisato Arashi",
        "jp_name": "嵐千砂都",
        "source": "Love Live! Superstar!!",
        "repo": "0xMifune/LoveLive"
    },
    "ren_hazuki": {
        "file": "RenHazukiV2.zip",
        "zh_name": "叶月恋",
        "en_name": "Ren Hazuki",
        "jp_name": "葉月恋",
        "source": "Love Live! Superstar!!",
        "repo": "0xMifune/LoveLive"
    },
    "sumire_heanna": {
        "file": "SumireHeannaV2.zip",
        "zh_name": "平安名堇",
        "en_name": "Sumire Heanna",
        "jp_name": "平安名すみれ",
        "source": "Love Live! Superstar!!",
        "repo": "0xMifune/LoveLive"
    },
    "kinako_sakurakoji": {
        "file": "SakurakojiKinakoV2.zip",
        "zh_name": "樱小路希奈子",
        "en_name": "Kinako Sakurakoji",
        "jp_name": "桜小路きな子",
        "source": "Love Live! Superstar!!",
        "repo": "0xMifune/LoveLive"
    },
    "mei_yoneme": {
        "file": "MeiYonemeV2.zip",
        "zh_name": "米女芽衣",
        "en_name": "Mei Yoneme",
        "jp_name": "米女芽衣",
        "source": "Love Live! Superstar!!",
        "repo": "0xMifune/LoveLive"
    },
    "shiki_wakana": {
        "file": "ShikiWakanaV2.zip",
        "zh_name": "若菜四季",
        "en_name": "Shiki Wakana",
        "jp_name": "若菜四季",
        "source": "Love Live! Superstar!!",
        "repo": "0xMifune/LoveLive"
    },
    "natsumi_onitsuka": {
        "file": "NatsumiOnitsukaV2.zip",
        "zh_name": "鬼塚夏美",
        "en_name": "Natsumi Onitsuka",
        "jp_name": "鬼塚夏美",
        "source": "Love Live! Superstar!!",
        "repo": "0xMifune/LoveLive"
    },
    "sayaka_murano": {
        "file": "SayakaMuranoV2.zip",
        "zh_name": "村野纱香",
        "en_name": "Sayaka Murano",
        "jp_name": "村野さやか",
        "source": "Love Live! 莲之空女学院学园偶像俱乐部",
        "repo": "0xMifune/LoveLive"
    },
    "tsuzuri_yugiri": {
        "file": "TsuzuriYugiri.zip",
        "zh_name": "夕雾缀理",
        "en_name": "Tsuzuri Yugiri",
        "jp_name": "夕霧綴理",
        "source": "Love Live! 莲之空女学院学园偶像俱乐部",
        "repo": "0xMifune/LoveLive"
    },
    "hanamaru": {
        "file": "Aqours_Hanamaru-Kunikida-RMVPE-Ov2.zip",
        "zh_name": "国木田花丸",
        "en_name": "Hanamaru Kunikida",
        "jp_name": "国木田花丸",
        "source": "Love Live! Sunshine!!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "setsuna_yuki_og": {
        "file": "Nijigasaki_Setsuna-Yuki-OG-RMVPE-Ov2.zip",
        "zh_name": "优木雪菜",
        "en_name": "Setsuna Yuki",
        "jp_name": "優木せつ菜",
        "variant": "OG",
        "source": "Love Live! 虹咲学园",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "setsuna_yuki_coco": {
        "file": "setsuna-yuki-coco-hayashi-ver-555epochs.zip",
        "zh_name": "优木雪菜",
        "en_name": "Setsuna Yuki",
        "jp_name": "優木せつ菜",
        "variant": "林鼓子版",
        "source": "Love Live! 虹咲学园",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "hanayo_pre_anime_v0": {
        "file": "Hanayo-Pre-Anime-v0.zip",
        "zh_name": "小泉花阳",
        "en_name": "Hanayo Koizumi",
        "jp_name": "小泉花陽",
        "variant": "预TV v0",
        "source": "Love Live!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "hanayo_pre_anime_v1": {
        "file": "Hanayo-Pre-Anime-v1.zip",
        "zh_name": "小泉花阳",
        "en_name": "Hanayo Koizumi",
        "jp_name": "小泉花陽",
        "variant": "预TV v1",
        "source": "Love Live!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "kotori_pre_anime": {
        "file": "Kotori-Pre-Anime.zip",
        "zh_name": "南小鸟",
        "en_name": "Kotori Minami",
        "jp_name": "南ことり",
        "variant": "预TV",
        "source": "Love Live!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "kotori_pre_anime_v2": {
        "file": "Kotori-Pre-Anime-v2.zip",
        "zh_name": "南小鸟",
        "en_name": "Kotori Minami",
        "jp_name": "南ことり",
        "variant": "预TV v2",
        "source": "Love Live!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "wien_margarete": {
        "file": "Liella_Wien-Margarete-RMVPE-Ov2.zip",
        "zh_name": "维恩·玛格丽特",
        "en_name": "Wien Margarete",
        "jp_name": "ウィーン・マルガレーテ",
        "source": "Love Live! Superstar!!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "mia_taylor": {
        "file": "Mia-Taylor-SIFAS-Lines-Only.zip",
        "zh_name": "米娅·泰勒",
        "en_name": "Mia Taylor",
        "jp_name": "ミア・テイラー",
        "source": "Love Live! 虹咲学园",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "anju_yuki_arise": {
        "file": "A-RISE_Anju-Yuki-RMVPE.zip",
        "zh_name": "优木安朱",
        "en_name": "Anju Yuki",
        "jp_name": "優木あんじゅ",
        "source": "Love Live!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "erena_todo_arise": {
        "file": "A-RISE_Erena-Todo-RMVPE.zip",
        "zh_name": "统堂英玲奈",
        "en_name": "Erena Todo",
        "jp_name": "統堂英玲奈",
        "source": "Love Live!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "tsubasa_kira_arise": {
        "file": "A-RISE_Tsubasa-Kira-RMVPE.zip",
        "zh_name": "绮罗翼",
        "en_name": "Tsubasa Kira",
        "jp_name": "綺羅ツバサ",
        "source": "Love Live!",
        "repo": "Swordsmagus/Love-Live-RVC"
    },
    "kanon": {
        "file": "ShibuyaKanonRVC.zip",
        "zh_name": "涩谷香音",
        "en_name": "Kanon Shibuya",
        "jp_name": "渋谷かのん",
        "source": "Love Live! Superstar!!",
        "repo": "Phos252/RVCmodels"
    },
    "setsuna": {
        "files": ["weights2/Setsuna.pth", "weights2/setsuna.index"],
        "zh_name": "优木雪菜",
        "en_name": "Setsuna Yuki",
        "jp_name": "優木せつ菜",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "setsuna_v2": {
        "files": ["weights2/Setsuna2.pth", "weights2/setsuna2.index"],
        "zh_name": "优木雪菜",
        "en_name": "Setsuna Yuki",
        "jp_name": "優木せつ菜",
        "variant": "v2",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "chika_v2": {
        "files": ["weights/Chika2.pth", "weights/chika2.index"],
        "zh_name": "高海千歌",
        "en_name": "Chika Takami",
        "jp_name": "高海千歌",
        "variant": "v2",
        "source": "Love Live! Sunshine!!",
        "repo": "trioskosmos/rvc_models"
    },
    "dia_v2": {
        "files": ["weights/Dia2.pth", "weights/dia2.index"],
        "zh_name": "黑泽黛雅",
        "en_name": "Dia Kurosawa",
        "jp_name": "黒澤ダイヤ",
        "variant": "v2",
        "source": "Love Live! Sunshine!!",
        "repo": "trioskosmos/rvc_models"
    },
    "hanamaru_v2": {
        "files": ["weights/Hanamaru2.pth", "weights/hanamaru2.index"],
        "zh_name": "国木田花丸",
        "en_name": "Hanamaru Kunikida",
        "jp_name": "国木田花丸",
        "variant": "v2",
        "source": "Love Live! Sunshine!!",
        "repo": "trioskosmos/rvc_models"
    },
    "honoka_v2": {
        "files": ["weights/Honoka2.pth", "weights/honoka2.index"],
        "zh_name": "高坂穗乃果",
        "en_name": "Honoka Kosaka",
        "jp_name": "高坂穂乃果",
        "variant": "v2",
        "source": "Love Live!",
        "repo": "trioskosmos/rvc_models"
    },
    "ayumu_v2": {
        "files": ["weights/Ayumu2.pth", "weights/ayumu2.index"],
        "zh_name": "上原步梦",
        "en_name": "Ayumu Uehara",
        "jp_name": "上原歩夢",
        "variant": "v2",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "ayumu_v3": {
        "files": ["weights/Ayumu3.pth", "weights/ayumu3.index"],
        "zh_name": "上原步梦",
        "en_name": "Ayumu Uehara",
        "jp_name": "上原歩夢",
        "variant": "v3",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "ayumu_v4": {
        "files": ["weights/Ayumu4.pth", "weights/ayumu4.index"],
        "zh_name": "上原步梦",
        "en_name": "Ayumu Uehara",
        "jp_name": "上原歩夢",
        "variant": "v4",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "karin_v2": {
        "files": ["weights/Karin2.pth", "weights/karin2.index"],
        "zh_name": "朝香果林",
        "en_name": "Karin Asaka",
        "jp_name": "朝香果林",
        "variant": "v2",
        "source": "Love Live! 虹咲学园",
        "repo": "trioskosmos/rvc_models"
    },
    "keke_v2": {
        "files": ["weights/Keke2.pth", "weights/keke2.index"],
        "zh_name": "唐可可",
        "en_name": "Keke Tang",
        "jp_name": "唐可可",
        "variant": "v2",
        "source": "Love Live! Superstar!!",
        "repo": "trioskosmos/rvc_models"
    },
    "riko_v2": {
        "files": ["weights2/Riko2.pth", "weights2/riko2.index"],
        "zh_name": "樱内梨子",
        "en_name": "Riko Sakurauchi",
        "jp_name": "桜内梨子",
        "variant": "v2",
        "source": "Love Live! Sunshine!!",
        "repo": "trioskosmos/rvc_models"
    },
    "umi_v2": {
        "files": ["weights2/Umi2.pth", "weights2/umi2.index"],
        "zh_name": "园田海未",
        "en_name": "Umi Sonoda",
        "jp_name": "園田海未",
        "variant": "v2",
        "source": "Love Live!",
        "repo": "trioskosmos/rvc_models"
    },
    "wien_v2": {
        "files": ["weights2/Wien2.pth", "weights2/wien.index"],
        "zh_name": "维恩·玛格丽特",
        "en_name": "Wien Margarete",
        "jp_name": "ウィーン・マルガレーテ",
        "variant": "v2",
        "source": "Love Live! Superstar!!",
        "repo": "trioskosmos/rvc_models"
    },
    "yohane_trios": {
        "files": ["weights2/Yohane.pth", "weights2/yohane.index"],
        "zh_name": "津岛善子（夜羽）",
        "en_name": "Yoshiko Tsushima (Yohane)",
        "jp_name": "ヨハネ",
        "variant": "trios",
        "source": "Love Live! Sunshine!! / 幻日夜羽",
        "repo": "trioskosmos/rvc_models"
    },
    "yoshiko_trios": {
        "files": ["weights2/yoshiko.pth", "weights2/yoshiko.index"],
        "zh_name": "津岛善子",
        "en_name": "Yoshiko Tsushima",
        "jp_name": "津島善子",
        "variant": "trios",
        "source": "Love Live! Sunshine!!",
        "repo": "trioskosmos/rvc_models"
    },
    "yoshiko_v2": {
        "files": ["weights2/Yoshiko2.pth", "weights2/yoshiko2.index"],
        "zh_name": "津岛善子",
        "en_name": "Yoshiko Tsushima",
        "jp_name": "津島善子",
        "variant": "v2",
        "source": "Love Live! Sunshine!!",
        "repo": "trioskosmos/rvc_models"
    },
    "tokai_teio": {
        "file": "Tokai Teio (Uma Musume) - Weights Model.zip",
        "zh_name": "东海帝皇",
        "en_name": "Tokai Teio",
        "jp_name": "トウカイテイオー",
        "source": "赛马娘",
        "repo": "kohaku12/RVC-MODELS"
    },
    "focalors": {
        "file": "Focalors.zip",
        "zh_name": "芙卡洛斯",
        "en_name": "Focalors",
        "jp_name": "フォカロルス",
        "source": "原神",
        "repo": "makiligon/RVC-Models"
    },
    "essex": {
        "file": "Essex.zip",
        "zh_name": "埃塞克斯",
        "en_name": "Essex",
        "jp_name": "エセックス",
        "source": "碧蓝航线",
        "repo": "makiligon/RVC-Models"
    },
    "ellie": {
        "file": "ellie.zip",
        "zh_name": "艾莉",
        "en_name": "Ellie",
        "jp_name": "エリー",
        "source": "社区模型",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "furina": {
        "file": "Furina.zip",
        "zh_name": "芙宁娜",
        "en_name": "Furina",
        "jp_name": "フリーナ",
        "source": "原神",
        "repo": "makiligon/RVC-Models"
    },
    "ayaka": {
        "file": "Ayaka.zip",
        "zh_name": "神里绫华",
        "en_name": "Kamisato Ayaka",
        "jp_name": "神里綾華",
        "source": "原神",
        "repo": "makiligon/RVC-Models"
    },
    "takane_shijou": {
        "file": "TakaneShijou.zip",
        "zh_name": "四条贵音",
        "en_name": "Takane Shijou",
        "jp_name": "四条貴音",
        "source": "偶像大师",
        "repo": "makiligon/RVC-Models"
    },
    "kobo": {
        "file": "kobo.zip",
        "zh_name": "可波·卡娜埃露",
        "en_name": "Kobo Kanaeru",
        "jp_name": "こぼ・かなえる",
        "source": "Hololive Indonesia",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "kaela": {
        "file": "kaela.zip",
        "zh_name": "卡埃拉·科瓦尔斯基亚",
        "en_name": "Kaela Kovalskia",
        "jp_name": "カエラ・コヴァルスキア",
        "source": "Hololive Indonesia",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "pekora": {
        "file": "pekora.zip",
        "zh_name": "兔田佩克拉",
        "en_name": "Usada Pekora",
        "jp_name": "兎田ぺこら",
        "source": "Hololive Japan",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "kizuna_ai": {
        "file": "kizuna-ai.zip",
        "zh_name": "绊爱",
        "en_name": "Kizuna AI",
        "jp_name": "キズナアイ",
        "source": "虚拟主播",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "fuwawa": {
        "file": "fuwawa.zip",
        "zh_name": "软软·阿比斯加德",
        "en_name": "Fuwawa Abyssgard",
        "jp_name": "フワワ・アビスガード",
        "source": "Hololive English",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "mococo": {
        "file": "mococo.zip",
        "zh_name": "茸茸·阿比斯加德",
        "en_name": "Mococo Abyssgard",
        "jp_name": "モココ・アビスガード",
        "source": "Hololive English",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "vo_furina_kr": {
        "file": "vo_furina_kr.zip",
        "zh_name": "芙宁娜",
        "en_name": "Furina",
        "jp_name": "フリーナ",
        "variant": "韩语",
        "lang": "韩文",
        "source": "原神",
        "repo": "jarari/RVC-v2"
    },
    "silverwolf_kr": {
        "file": "silverwolf_kr.zip",
        "zh_name": "银狼",
        "en_name": "Silver Wolf",
        "jp_name": "銀狼",
        "variant": "韩语",
        "lang": "韩文",
        "source": "崩坏：星穹铁道",
        "repo": "jarari/RVC-v2"
    },
    "sparkle_e40": {
        "file": "sparkle_e40.zip",
        "zh_name": "花火",
        "en_name": "Sparkle",
        "jp_name": "花火",
        "variant": "日语",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "ranko": {
        "file": "ranko.zip",
        "zh_name": "神崎兰子",
        "en_name": "Ranko Kanzaki",
        "jp_name": "神崎蘭子",
        "source": "偶像大师 灰姑娘女孩"
    },
    "yumemiriamu": {
        "file": "yumemiriamu.zip",
        "zh_name": "梦见莉亚梦",
        "en_name": "Riamu Yumemi",
        "jp_name": "夢見りあむ",
        "source": "偶像大师 灰姑娘女孩"
    },
    # ===== VOCALOID =====
    "hatsune_miku": {
        "file": "infamous_miku_v2.zip",
        "zh_name": "初音未来",
        "en_name": "Hatsune Miku",
        "jp_name": "初音ミク",
        "source": "VOCALOID",
        "repo": "javinfamous/infamous_miku_v2"
    },
    # ===== 原神 (更多角色) =====
    "nahida": {
        "file": "Nahida JP (VA_ Yukari Tamura) - Weights.gg Model.zip",
        "zh_name": "纳西妲",
        "en_name": "Nahida",
        "jp_name": "ナヒーダ",
        "source": "原神",
        "repo": "kohaku12/RVC-MODELS"
    },
    "nilou": {
        "file": "Nilou%20JP%20(Kanemoto%20Hisako)%20-%20Weights.gg%20Model.zip",
        "zh_name": "妮露",
        "en_name": "Nilou",
        "jp_name": "ニィロウ",
        "source": "原神",
        "repo": "kohaku12/RVC-MODELS"
    },
    # ===== 崩坏：星穹铁道 (更多角色) =====
    "herta_hsr": {
        "file": "Herta JP (VA_ Haruka Yamazaki) (Honkai_ Star Rail) - Weights.gg Model.zip",
        "zh_name": "黑塔",
        "en_name": "Herta",
        "jp_name": "ヘルタ",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "the_herta": {
        "file": "The Herta -Honkai Star Rail - Weights Model.zip",
        "zh_name": "大黑塔",
        "en_name": "The Herta",
        "jp_name": "マダム・ヘルタ",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "firefly_hsr": {
        "file": "Firefly _ Honkai_ Star Rail - Weights.gg Model.zip",
        "zh_name": "流萤",
        "en_name": "Firefly",
        "jp_name": "ホタル",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "tingyun_hsr": {
        "file": "Tingyun (Honkai_ Star Rail) (VA_ Yuuki Takada) - Weights.gg Model.zip",
        "zh_name": "停云",
        "en_name": "Tingyun",
        "jp_name": "停雲",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "yunli_hsr": {
        "file": "Yunli (Japanese Voice) Honkai Star Rail - Weights.gg Model.zip",
        "zh_name": "云璃",
        "en_name": "Yunli",
        "jp_name": "雲璃",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "tribbie_hsr": {
        "file": "Tribbie _ Honkai Star Rail JP (CV_ Hikaru Tono) - Weights Model.zip",
        "zh_name": "缇宝",
        "en_name": "Tribbie",
        "jp_name": "トリビー",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "huohuo_hsr": {
        "file": "huohuo2.zip",
        "zh_name": "藿藿",
        "en_name": "Huohuo",
        "jp_name": "フォフォ",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    "castorice_hsr": {
        "file": "Castorice (Honkai_ Star Rail, Japanese - CV_ Chiwa Saito) [Preliminary] - Weights Model.zip",
        "zh_name": "遐蝶",
        "en_name": "Castorice",
        "jp_name": "キャストリス",
        "source": "崩坏：星穹铁道",
        "repo": "kohaku12/RVC-MODELS"
    },
    # ===== 崩坏3rd =====
    "seele_hi3": {
        "file": "Seele (Honkai impact 3rd) - Weights.gg Model.zip",
        "zh_name": "希儿",
        "en_name": "Seele",
        "jp_name": "ゼーレ",
        "source": "崩坏3rd",
        "repo": "kohaku12/RVC-MODELS"
    },
    "herrscher_ego_hi3": {
        "file": "Herrscher of Human_ Ego (Honkai Impact 3rd) - Weights.gg Model.zip",
        "zh_name": "真我·人之律者",
        "en_name": "Herrscher of Human: Ego",
        "jp_name": "真我・人の律者",
        "source": "崩坏3rd",
        "repo": "kohaku12/RVC-MODELS"
    },
    # ===== 绝区零 =====
    "miyabi_zzz": {
        "file": "hoshimi miyabi (Zenless Zone Zero)JP - Weights Model.zip",
        "zh_name": "星见雅",
        "en_name": "Hoshimi Miyabi",
        "jp_name": "星見雅",
        "source": "绝区零",
        "repo": "kohaku12/RVC-MODELS"
    },
    # ===== Project SEKAI =====
    "nene_kusanagi": {
        "file": "Nene Kusanagi (Project Sekai) - Weights Model.zip",
        "zh_name": "草薙宁宁",
        "en_name": "Nene Kusanagi",
        "jp_name": "草薙寧々",
        "source": "Project SEKAI",
        "repo": "kohaku12/RVC-MODELS"
    },
    # ===== Hololive (更多角色) =====
    "miko_sakura": {
        "file": "miko.zip",
        "zh_name": "樱巫女",
        "en_name": "Sakura Miko",
        "jp_name": "さくらみこ",
        "source": "Hololive Japan",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "subaru_oozora": {
        "file": "subaru.zip",
        "zh_name": "大空昴",
        "en_name": "Oozora Subaru",
        "jp_name": "大空スバル",
        "source": "Hololive Japan",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "moona_hoshinova": {
        "file": "moona.zip",
        "zh_name": "穆娜・霍希诺瓦",
        "en_name": "Moona Hoshinova",
        "jp_name": "ムーナ・ホシノヴァ",
        "source": "Hololive Indonesia",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "risu_ayunda": {
        "file": "risu.zip",
        "zh_name": "阿云达·莉苏",
        "en_name": "Ayunda Risu",
        "jp_name": "アユンダ・リス",
        "source": "Hololive Indonesia",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "reine_pavolia": {
        "file": "reine.zip",
        "zh_name": "帕沃莉亚・蕾妮",
        "en_name": "Pavolia Reine",
        "jp_name": "パヴォリア・レイネ",
        "source": "Hololive Indonesia",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "zeta_vestia": {
        "file": "zeta.zip",
        "zh_name": "贝斯蒂亚・泽塔",
        "en_name": "Vestia Zeta",
        "jp_name": "ヴェスティア・ゼータ",
        "source": "Hololive Indonesia",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "anya_melfissa": {
        "file": "anya.zip",
        "zh_name": "安亚・梅尔菲莎",
        "en_name": "Anya Melfissa",
        "jp_name": "アーニャ・メルフィッサ",
        "source": "Hololive Indonesia",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    "luna_himemori": {
        "file": "luna.zip",
        "zh_name": "姬森露娜",
        "en_name": "Himemori Luna",
        "jp_name": "姫森ルーナ",
        "source": "Hololive Japan",
        "repo": "megaaziib/my-rvc-models-collection"
    },
    # ===== 碧蓝航线 =====
    "boothill": {
        "file": "Boothill.zip",
        "zh_name": "波提欧",
        "en_name": "Boothill",
        "jp_name": "ブートヒル",
        "source": "崩坏：星穹铁道",
        "repo": "makiligon/RVC-Models"
    },
    # ===== 明日方舟 =====
    "shiroko_rosmontis": {
        "file": "ShirokoRosmontis.zip",
        "zh_name": "白子/迷迭香",
        "en_name": "Shiroko / Rosmontis",
        "jp_name": "シロコ / ロスモンティス",
        "source": "蔚蓝档案 / 明日方舟",
        "repo": "makiligon/RVC-Models"
    },
    # ===== 赛马娘 (更多角色) =====
    "rice_shower": {
        "file": "RiceShowerSinging.zip",
        "zh_name": "米浴",
        "en_name": "Rice Shower",
        "jp_name": "ライスシャワー",
        "source": "赛马娘",
        "repo": "makiligon/RVC-Models"
    },
}


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def get_character_models_dir() -> Path:
    """获取角色模型目录"""
    return get_project_root() / "assets" / "weights" / "characters"


def list_available_characters() -> List[Dict]:
    """
    列出可用的角色模型

    Returns:
        list: 角色信息列表
    """
    result = []
    for name, info in CHARACTER_MODELS.items():
        source = info.get("source", "未知")
        display = _get_display_name(info, name)
        result.append({
            "name": name,
            "description": info.get("description", display),
            "display": display,
            "source": source,
            "series": normalize_series(source),
            "file": info.get("file"),
            "files": info.get("files"),
            "url": info.get("url"),
            "repo": info.get("repo", HF_REPO_ID)
        })
    return result


def list_downloaded_characters() -> List[Dict]:
    """
    列出已下载的角色模型

    Returns:
        list: 已下载的角色信息
    """
    models_dir = get_character_models_dir()
    if not models_dir.exists():
        return []

    downloaded = []
    seen = set()

    # 递归搜索 .pth，优先使用顶层目录名作为角色名
    for pth_file in models_dir.rglob("*.pth"):
        rel_path = pth_file.relative_to(models_dir)
        parts = rel_path.parts
        if len(parts) == 1:
            char_name = pth_file.stem
        else:
            char_name = parts[0]

        if char_name in seen:
            continue
        seen.add(char_name)

        info = CHARACTER_MODELS.get(char_name, {})
        source = info.get("source", "未知")
        display = _get_display_name(info, char_name)
        index_file = _find_index_file(pth_file)
        downloaded.append({
            "name": char_name,
            "description": info.get("description", display),
            "display": display,
            "source": source,
            "series": normalize_series(source),
            "model_path": str(pth_file),
            "index_path": str(index_file) if index_file else None
        })

    return downloaded


def get_character_model_path(name: str) -> Optional[Dict]:
    """
    获取角色模型路径

    Args:
        name: 角色名称

    Returns:
        dict: 包含 model_path 和 index_path 的字典，或 None
    """
    models_dir = get_character_models_dir()
    char_dir = models_dir / name

    # 1) 标准目录结构: characters/<name>/*.pth
    if char_dir.exists():
        pth_files = list(char_dir.glob("*.pth"))
        if pth_files:
            index_file = _find_index_file(pth_files[0])
            return {
                "model_path": str(pth_files[0]),
                "index_path": str(index_file) if index_file else None
            }

    # 2) 允许直接放在 characters 目录
    direct_pth = models_dir / f"{name}.pth"
    if direct_pth.exists():
        index_file = _find_index_file(direct_pth)
        return {
            "model_path": str(direct_pth),
            "index_path": str(index_file) if index_file else None
        }

    # 3) 兜底：在 characters 目录递归查找同名模型
    for pth_file in models_dir.rglob("*.pth"):
        if pth_file.stem.lower() == name.lower():
            index_file = _find_index_file(pth_file)
            return {
                "model_path": str(pth_file),
                "index_path": str(index_file) if index_file else None
            }

    return None


def download_character_model(
    name: str,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> bool:
    """
    下载角色模型

    Args:
        name: 角色名称
        progress_callback: 进度回调 (message, progress)

    Returns:
        bool: 是否成功
    """
    if name not in CHARACTER_MODELS:
        _safe_print(f"未知角色: {name}")
        return False

    char_info = CHARACTER_MODELS[name]
    repo_id = char_info.get("repo", HF_REPO_ID)
    zip_file = char_info.get("file")
    file_list = char_info.get("files")
    direct_url = char_info.get("url")
    gdrive_id = char_info.get("gdrive_id")
    pattern = char_info.get("pattern")

    # 检查是否已下载
    char_dir = get_character_models_dir() / name
    if char_dir.exists() and list(char_dir.glob("*.pth")):
        _safe_print(f"角色模型已存在: {name}")
        return True

    if progress_callback:
        progress_callback(f"正在下载 {name} 模型...", 0.1)

    hf_token = _get_hf_token()

    try:
        # Google Drive 下载
        if gdrive_id:
            filename = char_info.get("filename") or f"{name}.zip"
            temp_dir = get_project_root() / "temp" / "downloads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / filename

            _safe_print(f"正在从 Google Drive 下载: {filename}")
            if not _download_gdrive_file(gdrive_id, temp_path):
                return False

            char_dir.mkdir(parents=True, exist_ok=True)
            if temp_path.suffix.lower() == ".zip":
                if progress_callback:
                    progress_callback(f"正在解压 {name} 模型...", 0.6)
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(char_dir)
                _flatten_extracted_dir(char_dir)
            else:
                shutil.copy(str(temp_path), str(char_dir / temp_path.name))

        # 直链下载（可用于非 HuggingFace 源）
        elif direct_url:
            if "mega.nz" in direct_url:
                _safe_print("Mega 下载暂不支持，请手动下载并放入角色目录")
                return False
            from tools.download_models import download_file

            filename = char_info.get("filename") or Path(direct_url).name
            temp_dir = get_project_root() / "temp" / "downloads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / filename

            _safe_print(f"正在下载: {direct_url}")
            if not download_file(direct_url, temp_path, name):
                return False

            char_dir.mkdir(parents=True, exist_ok=True)
            if temp_path.suffix.lower() == ".zip":
                if progress_callback:
                    progress_callback(f"正在解压 {name} 模型...", 0.6)
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(char_dir)
                _flatten_extracted_dir(char_dir)
            else:
                shutil.copy(str(temp_path), str(char_dir / temp_path.name))

        # HuggingFace 多文件下载
        elif file_list:
            if not HF_AVAILABLE:
                raise ImportError("请安装 huggingface_hub: pip install huggingface_hub")

            _safe_print(f"正在从 HuggingFace 下载: {repo_id} (files)")
            char_dir.mkdir(parents=True, exist_ok=True)
            total = len(file_list)
            for idx, hf_file in enumerate(file_list, start=1):
                if progress_callback and total > 0:
                    progress_callback(
                        f"正在下载 {name} 文件 {idx}/{total}...",
                        0.1 + 0.8 * (idx / total)
                    )
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_file,
                    cache_dir=str(get_project_root() / "temp" / "hf_cache"),
                    token=hf_token
                )
                target_name = Path(hf_file).name
                shutil.copy(str(downloaded_path), str(char_dir / target_name))

        # HuggingFace: 根据 pattern 自动选择文件
        elif pattern:
            if not HF_AVAILABLE:
                raise ImportError("请安装 huggingface_hub: pip install huggingface_hub")

            _safe_print(f"正在从 HuggingFace 下载: {repo_id} (auto)")
            files = list_repo_files(repo_id, token=hf_token)
            if isinstance(pattern, str):
                patterns = [pattern]
            else:
                patterns = list(pattern)

            candidates = []
            for f in files:
                for p in patterns:
                    if p in f or (p == ".zip" and f.lower().endswith(".zip")):
                        candidates.append(f)
                        break

            if not candidates:
                _safe_print(f"未找到匹配文件: {pattern}")
                return False

            # 优先 zip
            zip_candidates = [c for c in candidates if c.lower().endswith(".zip")]
            selected = zip_candidates[0] if zip_candidates else candidates[0]

            if selected.lower().endswith(".zip"):
                zip_file = selected
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=zip_file,
                    cache_dir=str(get_project_root() / "temp" / "hf_cache"),
                    token=hf_token
                )
                if progress_callback:
                    progress_callback(f"正在解压 {name} 模型...", 0.6)
                char_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                    zip_ref.extractall(char_dir)
                _flatten_extracted_dir(char_dir)
            else:
                # 多文件下载
                file_list = candidates
                char_dir.mkdir(parents=True, exist_ok=True)
                total = len(file_list)
                for idx, hf_file in enumerate(file_list, start=1):
                    if progress_callback and total > 0:
                        progress_callback(
                            f"正在下载 {name} 文件 {idx}/{total}...",
                            0.1 + 0.8 * (idx / total)
                        )
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=hf_file,
                        cache_dir=str(get_project_root() / "temp" / "hf_cache"),
                        token=hf_token
                    )
                    target_name = Path(hf_file).name
                    shutil.copy(str(downloaded_path), str(char_dir / target_name))

        # HuggingFace zip 下载
        else:
            if not HF_AVAILABLE:
                raise ImportError("请安装 huggingface_hub: pip install huggingface_hub")
            _safe_print(f"正在从 HuggingFace 下载: {repo_id}/{zip_file}")

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=zip_file,
                cache_dir=str(get_project_root() / "temp" / "hf_cache"),
                token=hf_token
            )

            if progress_callback:
                progress_callback(f"正在解压 {name} 模型...", 0.6)

            char_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                zip_ref.extractall(char_dir)

            _flatten_extracted_dir(char_dir)

        if progress_callback:
            progress_callback(f"{name} 模型下载完成", 1.0)

        # 下载完成后更新版本说明缓存
        _get_version_note(name, char_info)
        _safe_print(f"角色模型已下载: {name}")
        return True

    except Exception as e:
        _safe_print(f"下载失败: {e}")
        if progress_callback:
            progress_callback(f"下载失败: {e}", 0)
        return False


def _flatten_extracted_dir(char_dir: Path):
    """
    处理解压后可能的嵌套目录结构
    """
    # 检查是否有嵌套的单一目录
    items = list(char_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        nested_dir = items[0]
        # 移动内容到上级目录
        for item in nested_dir.iterdir():
            shutil.move(str(item), str(char_dir / item.name))
        nested_dir.rmdir()


def delete_character_model(name: str) -> bool:
    """
    删除角色模型

    Args:
        name: 角色名称

    Returns:
        bool: 是否成功
    """
    char_dir = get_character_models_dir() / name
    if char_dir.exists():
        shutil.rmtree(char_dir)
        print(f"已删除角色模型: {name}")
        return True
    return False


def check_hf_available() -> bool:
    """检查 huggingface_hub 是否可用"""
    return HF_AVAILABLE


def list_available_series() -> List[str]:
    """
    获取可用的作品/系列列表（去重）
    """
    series_set = set()
    for info in CHARACTER_MODELS.values():
        source = info.get("source", "未知")
        series_set.add(normalize_series(source))
    return sorted(series_set)


def download_all_character_models(
    series: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, List[str]]:
    """
    批量下载角色模型

    Args:
        series: 仅下载指定系列（如 "Love Live!"），None 表示全部
        progress_callback: 进度回调 (message, progress)

    Returns:
        dict: { "success": [...], "failed": [...] }
    """
    targets = list_available_characters()
    if series and series != "全部":
        targets = [c for c in targets if c.get("series") == series]

    success = []
    failed = []
    total = max(len(targets), 1)

    for idx, char in enumerate(targets, start=1):
        if progress_callback:
            progress_callback(
                f"正在下载 {char['name']} ({idx}/{total})...",
                idx / total
            )
        ok = download_character_model(char["name"])
        if ok:
            success.append(char["name"])
        else:
            failed.append(char["name"])

    return {
        "success": success,
        "failed": failed
    }


def get_character_choices() -> List[str]:
    """
    获取角色选择列表 (用于 UI 下拉框)

    Returns:
        list: 角色名称列表
    """
    downloaded = list_downloaded_characters()
    return [c["name"] for c in downloaded]
