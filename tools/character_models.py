# -*- coding: utf-8 -*-
"""
角色模型管理 - 从 HuggingFace 下载 RVC 角色模型
"""
import os
import json
import re
import zipfile
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.console_i18n import console_print as print

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# HuggingFace 仓库配置
HF_REPO_ID = "trioskosmos/rvc_models"
_VERSION_NOTE_CACHE: Dict[str, Optional[str]] = {}
_VERSION_NOTE_CACHE_LOADED = False
LOCAL_MODEL_INFO_FILENAME = "ai_rvc_model.json"
INTEGRATED_REPO_IDS = {
    "trioskosmos/rvc_models",
    "Icchan/LoveLive",
    "0xMifune/LoveLive",
    "Zurakichi/RVC",
    "Swordsmagus/Love-Live-RVC",
    "makiligon/RVC-Models",
    "kohaku12/RVC-MODELS",
    "megaaziib/my-rvc-models-collection",
}


def _get_hf_token() -> Optional[str]:
    """获取 HuggingFace Token（支持 HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HUGGINGFACE_TOKEN）"""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

# 作品归类（用于 UI 分类筛选）。顺序很重要：更具体的来源必须先匹配。
SERIES_ALIASES = {
    "Love Live! Sunshine!! / 幻日夜羽": "Love Live! / 幻日夜羽",
    "Love Live! Sunshine!!": "Love Live! / Sunshine!!",
    "Love Live! Superstar!!": "Love Live! / Superstar!!",
    "Love Live! 虹咲学园": "Love Live! / 虹咲学园",
    "Love Live! 虹咲学園": "Love Live! / 虹咲学园",
    "Love Live! 莲之空女学院学园偶像俱乐部": "Love Live! / 莲之空",
    "Love Live!": "Love Live! / μ's",
    "Hololive Japan": "Hololive / JP",
    "Hololive English": "Hololive / EN",
    "Hololive Indonesia": "Hololive / ID",
    "Holostars Japan": "Holostars / JP",
    "Holostars English": "Holostars / EN",
    "虚拟主播": "VTuber / 其他",
    "NIJISANJI English": "NIJISANJI / EN",
    "NIJISANJI Indonesia": "NIJISANJI / ID",
    "原神": "米哈游 / 原神",
    "崩坏：星穹铁道": "米哈游 / 崩坏：星穹铁道",
    "崩坏3rd": "米哈游 / 崩坏3rd",
    "绝区零": "米哈游 / 绝区零",
    "偶像大师 灰姑娘女孩": "偶像大师 / 灰姑娘女孩",
    "偶像大师": "偶像大师 / 本家",
    "赛马娘": "赛马娘",
    "Project SEKAI": "Project SEKAI",
    "VOCALOID": "VOCALOID",
    "碧蓝航线": "碧蓝航线",
    "蔚蓝档案 / 明日方舟": "跨作品 / 蔚蓝档案 + 明日方舟",
    "社区模型": "社区 / 其他模型",
    "原创角色": "社区 / 原创角色",
}


def normalize_series(source: str) -> str:
    """将来源归类到系列"""
    if not source:
        return "未知"
    for key, series in SERIES_ALIASES.items():
        if source.startswith(key):
            return series
    return source


def _get_character_category(info: Dict) -> str:
    category = str(info.get("category") or "").strip()
    if category:
        return category
    return normalize_series(str(info.get("source") or "未知"))


def _dedupe_parts(parts: List[str]) -> List[str]:
    result = []
    seen = set()
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _get_registry_repo_id(info: Dict) -> Optional[str]:
    repo_id = str(info.get("repo") or "").strip()
    if repo_id:
        return repo_id
    if info.get("gdrive_id") or info.get("url"):
        return None
    if info.get("file") or info.get("files") or info.get("pattern"):
        return HF_REPO_ID
    return None


def _build_repo_page_url(repo_id: Optional[str]) -> Optional[str]:
    repo_id = str(repo_id or "").strip()
    if not repo_id:
        return None
    return f"https://huggingface.co/{repo_id}"


def _build_repo_file_url(repo_id: Optional[str], file_name: Optional[str]) -> Optional[str]:
    repo_id = str(repo_id or "").strip()
    file_name = str(file_name or "").strip()
    if not repo_id or not file_name:
        return None
    return f"https://huggingface.co/{repo_id}/resolve/main/{quote(file_name, safe='/')}"


def _build_gdrive_view_url(file_id: Optional[str]) -> Optional[str]:
    file_id = str(file_id or "").strip()
    if not file_id:
        return None
    return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"


def _infer_download_method(info: Dict) -> str:
    if info.get("gdrive_id"):
        return "google_drive"
    if info.get("url"):
        return "direct_url"
    if info.get("files"):
        return "huggingface_files"
    if info.get("pattern"):
        return "huggingface_pattern"
    if info.get("file"):
        return "huggingface_zip"
    return "unknown"


def _infer_distribution(info: Dict) -> str:
    distribution = str(info.get("distribution") or "").strip()
    if distribution:
        return distribution
    if info.get("gdrive_id"):
        return "社区直链"
    if info.get("url"):
        if "huggingface.co" in str(info.get("url", "")):
            return "独立仓库"
        return "外部直链"
    repo_id = _get_registry_repo_id(info)
    if repo_id in INTEGRATED_REPO_IDS:
        return "整合仓库"
    if repo_id:
        return "独立仓库"
    return "未知来源"


def _infer_continuity(info: Dict) -> Optional[str]:
    continuity = str(info.get("continuity") or "").strip()
    if continuity:
        return continuity

    source = str(info.get("source") or "").strip()
    joined_name = " ".join(
        str(info.get(key) or "")
        for key in ("zh_name", "en_name", "jp_name", "variant")
    )

    if "幻日夜羽" in source or "夜羽" in joined_name or "ヨハネ" in joined_name:
        return "幻日夜羽"
    if source.startswith("Love Live! Sunshine!!"):
        return "Sunshine 正篇"
    if source.startswith("Love Live! 虹咲"):
        return "虹咲学园"
    if source.startswith("Love Live! Superstar!!"):
        return "Superstar!!"
    if source.startswith("Love Live! 莲之空"):
        return "莲之空"
    if source == "Love Live!":
        return "μ's"
    return None


def _humanize_variant(info: Dict) -> Optional[str]:
    variant = str(info.get("variant") or "").strip()
    if not variant:
        return None
    if variant.lower() == "trios":
        return "trios 版"
    return variant


def _build_variant_label(name: str, info: Dict) -> Optional[str]:
    parts = []
    variant = _humanize_variant(info)
    if variant:
        parts.append(variant)
    variant_note = _get_version_note(name, info)
    if variant_note and variant_note != "未提供版本说明":
        parts.append(variant_note)
    deduped = _dedupe_parts(parts)
    return " · ".join(deduped) if deduped else None


def _get_base_display_name(info: Dict, fallback: str) -> str:
    zh_name = info.get("zh_name") or info.get("description") or fallback
    en_name = info.get("en_name")
    jp_name = info.get("jp_name")
    parts = [zh_name]
    if en_name and en_name != zh_name:
        parts.append(en_name)
    if jp_name and jp_name != zh_name and jp_name != en_name:
        parts.append(jp_name)
    return " / ".join(parts)


def _get_display_name(info: Dict, fallback: str) -> str:
    """拼接中文名 / 英文名 / 日文名用于展示"""
    display = _get_base_display_name(info, fallback)
    variant_label = _build_variant_label(fallback, info)
    if variant_label:
        display = f"{display} - {variant_label}"
    return display


def _find_index_file(pth_file: Path) -> Optional[Path]:
    """尝试找到对应的索引文件"""
    candidate = pth_file.with_suffix(".index")
    if candidate.exists():
        return candidate

    index_files = list(pth_file.parent.glob("*.index"))
    if not index_files:
        return None

    for idx in pth_file.parent.glob("*.index"):
        if idx.stem.lower() == pth_file.stem.lower():
            return idx

    if len(index_files) == 1:
        return index_files[0]

    def _normalize_name(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _tokenize_name(text: str) -> List[str]:
        return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 2]

    model_norm = _normalize_name(pth_file.stem)
    model_tokens = set(_tokenize_name(pth_file.stem))

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
        shared_tokens = len(model_tokens & idx_tokens)
        score += shared_tokens * 40
        if "added" in idx.stem.lower():
            score += 10
        if score > best_score:
            best_score = score
            best_match = idx

    if best_match is not None and best_score > 0:
        return best_match
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

    for key in ("version_label", "version_note", "variant"):
        value = _normalize_note(str(data.get(key) or ""))
        if value:
            return value

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
        for candidate in (LOCAL_MODEL_INFO_FILENAME, "metadata.json", "model_info.json", "info.json"):
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


def _load_local_model_info(char_dir: Path) -> Dict:
    info_path = char_dir / LOCAL_MODEL_INFO_FILENAME
    if not info_path.exists():
        return {}
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _build_source_page_url(info: Dict) -> Optional[str]:
    explicit = str(info.get("source_page_url") or "").strip()
    if explicit:
        return explicit
    repo_id = _get_registry_repo_id(info)
    if repo_id:
        return _build_repo_page_url(repo_id)
    if info.get("url"):
        return str(info.get("url")).strip()
    return _build_gdrive_view_url(info.get("gdrive_id"))


def _build_download_url(info: Dict) -> Optional[str]:
    explicit = str(info.get("download_url") or "").strip()
    if explicit:
        return explicit
    if info.get("url"):
        return str(info.get("url")).strip()
    if info.get("gdrive_id"):
        return _build_gdrive_view_url(info.get("gdrive_id"))
    repo_id = _get_registry_repo_id(info)
    file_name = info.get("file")
    if file_name:
        return _build_repo_file_url(repo_id, file_name)
    files = info.get("files") or []
    if files:
        return _build_repo_file_url(repo_id, files[0])
    return _build_repo_page_url(repo_id)


def _build_character_record(name: str, info: Dict) -> Dict:
    source = info.get("source", "未知")
    category = _get_character_category(info)
    base_display = _get_base_display_name(info, name)
    display = _get_display_name(info, name)
    repo_id = _get_registry_repo_id(info)
    version_note = _get_version_note(name, info)
    version_label = (
        str(info.get("version_label") or "").strip()
        or _build_variant_label(name, info)
        or version_note
        or ""
    )
    return {
        "name": name,
        "description": info.get("description", display),
        "base_display": base_display,
        "display": display,
        "source": source,
        "series": category,
        "category": category,
        "variant": str(info.get("variant") or "").strip(),
        "version_note": version_note,
        "version_label": version_label,
        "role": str(info.get("role") or "角色模型").strip() or "角色模型",
        "continuity": _infer_continuity(info) or "",
        "distribution": _infer_distribution(info),
        "repo": repo_id,
        "source_page_url": _build_source_page_url(info),
        "download_url": _build_download_url(info),
        "download_method": _infer_download_method(info),
        "file": info.get("file"),
        "files": info.get("files"),
        "url": info.get("url"),
        "gdrive_id": info.get("gdrive_id"),
        "zh_name": info.get("zh_name"),
        "en_name": info.get("en_name"),
        "jp_name": info.get("jp_name"),
    }


def _write_local_model_info(name: str, char_dir: Path, info: Dict):
    try:
        payload = _build_character_record(name, info)
        payload["registry_key"] = name
        payload["local_files"] = {
            "pth": sorted(p.name for p in char_dir.glob("*.pth")),
            "index": sorted(p.name for p in char_dir.glob("*.index")),
            "metadata": sorted(
                p.name for p in char_dir.glob("*.json")
                if p.name != LOCAL_MODEL_INFO_FILENAME
            ),
        }
        info_path = char_dir / LOCAL_MODEL_INFO_FILENAME
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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
        "source": "Love Live! Sunshine!!",
        "distribution": "社区直链",
        "source_page_url": "https://rentry.co/llrvc",
        "version_label": "RVC V2 · 211 epochs"
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

CHARACTER_MODELS.update({
    # ===== 原神：mrmocciai/genshin-impact 成对权重 =====
    "genshin_aether_mrmocciai": {
        "files": ["model/weights/aether-v2.pth", "model/aether-v2/added_IVF596_Flat_nprobe_1_aether-v2_v2.index"],
        "zh_name": "空",
        "en_name": "Aether",
        "jp_name": "空",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_amber_mrmocciai": {
        "files": ["model/weights/amber-v2.pth", "model/amber-v2/added_IVF837_Flat_nprobe_1_amber-v2_v2.index"],
        "zh_name": "安柏",
        "en_name": "Amber",
        "jp_name": "アンバー",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_arlecchino_mrmocciai": {
        "files": ["model/weights/arle_rmvpe.pth", "model/arle_rmvpe/added_IVF656_Flat_nprobe_1_arle_rmvpe_v2.index"],
        "zh_name": "阿蕾奇诺",
        "en_name": "Arlecchino",
        "jp_name": "アルレッキーノ",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_ayaka_mrmocciai": {
        "files": ["model/weights/ayaka-rmvpe.pth", "model/ayaka-rmvpe/added_IVF1228_Flat_nprobe_1_kamisato_ayaka_v2.index"],
        "zh_name": "神里绫华",
        "en_name": "Kamisato Ayaka",
        "jp_name": "神里綾華",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_beidou_mrmocciai": {
        "files": ["model/weights/beidou_rmvpe.pth", "model/beidou_rmvpe/added_IVF857_Flat_nprobe_1_beidou_rmvpe_v2.index"],
        "zh_name": "北斗",
        "en_name": "Beidou",
        "jp_name": "北斗",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_bennett_mrmocciai": {
        "files": ["model/weights/bennett-v2.pth", "model/bennett-v2/added_IVF900_Flat_nprobe_1_benett-v2_v2.index"],
        "zh_name": "班尼特",
        "en_name": "Bennett",
        "jp_name": "ベネット",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_candace_mrmocciai": {
        "files": ["model/weights/candez.pth", "model/candez-v2/added_IVF920_Flat_nprobe_1_candez_v2.index"],
        "zh_name": "坎蒂丝",
        "en_name": "Candace",
        "jp_name": "キャンディス",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_childe_mrmocciai": {
        "files": ["model/weights/childe-v2.pth", "model/childe-v2/added_IVF684_Flat_nprobe_1_childe-v2_v2.index"],
        "zh_name": "达达利亚",
        "en_name": "Childe",
        "jp_name": "タルタリヤ",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_chiori_mrmocciai": {
        "files": ["model/weights/chiori_rmvpe.pth", "model/chiori_rmvpe/added_IVF450_Flat_nprobe_1_chiori_rmvpe_v2.index"],
        "zh_name": "千织",
        "en_name": "Chiori",
        "jp_name": "千織",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_clorinde_mrmocciai": {
        "files": ["model/weights/clorinde.pth", "model/clorinde/added_IVF217_Flat_nprobe_1_clorinde_v2.index"],
        "zh_name": "克洛琳德",
        "en_name": "Clorinde",
        "jp_name": "クロリンデ",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_collei_mrmocciai": {
        "files": ["model/weights/collei-v2.pth", "model/collei-v2/added_IVF1127_Flat_nprobe_1_collei_v2.index"],
        "zh_name": "柯莱",
        "en_name": "Collei",
        "jp_name": "コレイ",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_dehya_mrmocciai": {
        "files": ["model/weights/dehya_rmvpe.pth", "model/dehya_rmvpe/added_IVF493_Flat_nprobe_1_dehya_rmvpe_v2.index"],
        "zh_name": "迪希雅",
        "en_name": "Dehya",
        "jp_name": "ディシア",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_ei_mrmocciai": {
        "files": ["model/weights/ei2_rmvpe.pth", "model/ei2_rmvpe/added_IVF480_Flat_nprobe_1_ei2_rmvpe_v2.index"],
        "zh_name": "雷电影",
        "en_name": "Raiden Ei",
        "jp_name": "雷電影",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_eula_mrmocciai": {
        "files": ["model/weights/eula_rmvpe.pth", "model/eula_rmvpe/added_IVF1185_Flat_nprobe_1_eula-v2_v2.index"],
        "zh_name": "优菈",
        "en_name": "Eula",
        "jp_name": "エウルア",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_faruzan_mrmocciai": {
        "files": ["model/weights/faruzan_rmvpe.pth", "model/faruzan_rmvpe/added_IVF1101_Flat_nprobe_1_faruzan_rmvpe_v2.index"],
        "zh_name": "珐露珊",
        "en_name": "Faruzan",
        "jp_name": "ファルザン",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_furina_mrmocciai": {
        "files": ["model/weights/furina_rmvpe.pth", "model/furina_rmvpe/added_IVF1203_Flat_nprobe_1_furina_rmvpe_v2.index"],
        "zh_name": "芙宁娜",
        "en_name": "Furina",
        "jp_name": "フリーナ",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_ganyu_mrmocciai": {
        "files": ["model/weights/ganyu-v2.pth", "model/ganyu-v2/added_IVF816_Flat_nprobe_1_ganyu_v2.index"],
        "zh_name": "甘雨",
        "en_name": "Ganyu",
        "jp_name": "甘雨",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_hutao_mrmocciai": {
        "files": ["model/weights/hutao-v2.pth", "model/hutao-v2/added_IVF601_Flat_nprobe_1_hutao_v2.index"],
        "zh_name": "胡桃",
        "en_name": "Hu Tao",
        "jp_name": "胡桃",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_jean_mrmocciai": {
        "files": ["model/weights/jean-v2.pth", "model/jean-v2/added_IVF675_Flat_nprobe_1_jean-v2_v2.index"],
        "zh_name": "琴",
        "en_name": "Jean",
        "jp_name": "ジン",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_kaveh_mrmocciai": {
        "files": ["model/weights/kaveh_v2.pth", "model/kaveh-v2/added_IVF613_Flat_nprobe_1_kaveh_v2_v2.index"],
        "zh_name": "卡维",
        "en_name": "Kaveh",
        "jp_name": "カーヴェ",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_kazuha_mrmocciai": {
        "files": ["model/weights/kazuha-v2.pth", "model/kazuha-v2/added_IVF860_Flat_nprobe_1_kazuha_v2.index"],
        "zh_name": "枫原万叶",
        "en_name": "Kaedehara Kazuha",
        "jp_name": "楓原万葉",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_keqing_mrmocciai": {
        "files": ["model/weights/keqing-v2.pth", "model/keqing-v2/added_IVF1430_Flat_nprobe_1_keqing-v2_v2.index"],
        "zh_name": "刻晴",
        "en_name": "Keqing",
        "jp_name": "刻晴",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_kirara_mrmocciai": {
        "files": ["model/weights/kirara_rmvpe.pth", "model/kirara_rmvpe/added_IVF443_Flat_nprobe_1_kirara_rmvpe_v2.index"],
        "zh_name": "绮良良",
        "en_name": "Kirara",
        "jp_name": "綺良々",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_kokomi_mrmocciai": {
        "files": ["model/weights/kokomi-v2.pth", "model/kokomi-v2/added_IVF934_Flat_nprobe_1_kokomi_v2.index"],
        "zh_name": "珊瑚宫心海",
        "en_name": "Sangonomiya Kokomi",
        "jp_name": "珊瑚宮心海",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_kujou_sara_mrmocciai": {
        "files": ["model/weights/kujou_sara.pth", "model/kujou_sara/added_IVF398_Flat_nprobe_1_kujou_sara_v2.index"],
        "zh_name": "九条裟罗",
        "en_name": "Kujou Sara",
        "jp_name": "九条裟羅",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_layla_mrmocciai": {
        "files": ["model/weights/layla-v2.pth", "model/layla-v2/added_IVF1099_Flat_nprobe_1_layla-v2_v2.index"],
        "zh_name": "莱依拉",
        "en_name": "Layla",
        "jp_name": "レイラ",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_lisa_mrmocciai": {
        "files": ["model/weights/lisa-v2.pth", "model/lisa-v2/added_IVF758_Flat_nprobe_1_lisa_v2.index"],
        "zh_name": "丽莎",
        "en_name": "Lisa",
        "jp_name": "リサ",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_lumine_mrmocciai": {
        "files": ["model/weights/lumine_rmvpe.pth", "model/lumine_rmvpe/added_IVF1329_Flat_nprobe_1_lumine-rmvpe_v2.index"],
        "zh_name": "荧",
        "en_name": "Lumine",
        "jp_name": "蛍",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_lynette_mrmocciai": {
        "files": ["model/weights/lynette_rmvpe.pth", "model/lynette_rmvpe/added_IVF386_Flat_nprobe_1_lynette_rmvpe_v2.index"],
        "zh_name": "琳妮特",
        "en_name": "Lynette",
        "jp_name": "リネット",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_lyney_mrmocciai": {
        "files": ["model/weights/lyney_rmvpe.pth", "model/lyney_rmvpe/added_IVF467_Flat_nprobe_1_lyney_rmvpe_v2.index"],
        "zh_name": "林尼",
        "en_name": "Lyney",
        "jp_name": "リネ",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_mavuika_mrmocciai": {
        "files": ["model/weights/mavuika-rmvpe.pth", "model/mavuika-rmvpe/added_IVF406_Flat_nprobe_1_mavuika_v2.index"],
        "zh_name": "玛薇卡",
        "en_name": "Mavuika",
        "jp_name": "マーヴィカ",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_mualani_mrmocciai": {
        "files": ["model/weights/mualani-jp.pth", "model/mualani- jp/added_IVF126_Flat_nprobe_1_mualani-jp_v2.index"],
        "zh_name": "玛拉妮",
        "en_name": "Mualani",
        "jp_name": "ムアラニ",
        "variant": "日语",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_navia_mrmocciai": {
        "files": ["model/weights/navia_rmvpe.pth", "model/navia_rmvpe/added_IVF453_Flat_nprobe_1_navia_rmvpe_v2.index"],
        "zh_name": "娜维娅",
        "en_name": "Navia",
        "jp_name": "ナヴィア",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_neuvillette_mrmocciai": {
        "files": ["model/weights/neuvillette.pth", "model/neuvillette/added_IVF353_Flat_nprobe_1_neuvillette_v2.index"],
        "zh_name": "那维莱特",
        "en_name": "Neuvillette",
        "jp_name": "ヌヴィレット",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_paimon_mrmocciai": {
        "files": ["model/weights/paimon_rmvpe.pth", "model/paimon_rmvpe/added_IVF1480_Flat_nprobe_1_paimon_rmvpe_v2.index"],
        "zh_name": "派蒙",
        "en_name": "Paimon",
        "jp_name": "パイモン",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_raiden_puppet_mrmocciai": {
        "files": ["model/weights/raiden-puppet-rmvpe.pth", "model/raiden-puppet-rmvpe/added_IVF645_Flat_nprobe_1_raiden-puppet_rmvpe_v2.index"],
        "zh_name": "雷电将军",
        "en_name": "Raiden Shogun",
        "jp_name": "雷電将軍",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_shenhe_mrmocciai": {
        "files": ["model/weights/shenhe_rmvpe.pth", "model/shenhe_rmvpe/added_IVF453_Flat_nprobe_1_shenhe_rmvpe_v2.index"],
        "zh_name": "申鹤",
        "en_name": "Shenhe",
        "jp_name": "申鶴",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_venti_mrmocciai": {
        "files": ["model/weights/venti_rmvpe.pth", "model/venti_rmvpe/added_IVF463_Flat_nprobe_1_venti_rmvpe_v2.index"],
        "zh_name": "温迪",
        "en_name": "Venti",
        "jp_name": "ウェンティ",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_wanderer_mrmocciai": {
        "files": ["model/weights/warderer-v2.pth", "model/warderer-v2/added_IVF953_Flat_nprobe_1_wanderer-v2_v2.index"],
        "zh_name": "流浪者",
        "en_name": "Wanderer",
        "jp_name": "放浪者",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_wriothesley_mrmocciai": {
        "files": ["model/weights/wriothesley.pth", "model/wriothesley/added_IVF408_Flat_nprobe_1_wriothesley_v2.index"],
        "zh_name": "莱欧斯利",
        "en_name": "Wriothesley",
        "jp_name": "リオセスリ",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_xiangling_mrmocciai": {
        "files": ["model/weights/xiangling-v2.pth", "model/xiangling-v2/added_IVF814_Flat_nprobe_1_xianling-v2_v2.index"],
        "zh_name": "香菱",
        "en_name": "Xiangling",
        "jp_name": "香菱",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_xianyun_mrmocciai": {
        "files": ["model/weights/xianyun-jp.pth", "model/Xianyun-jp/added_IVF145_Flat_nprobe_1_xianyun-jp_v2.index"],
        "zh_name": "闲云",
        "en_name": "Xianyun",
        "jp_name": "閑雲",
        "variant": "日语",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_xiao_mrmocciai": {
        "files": ["model/weights/xiao-v2.pth", "model/xiao-v2/added_IVF647_Flat_nprobe_1_xiao_v2.index"],
        "zh_name": "魈",
        "en_name": "Xiao",
        "jp_name": "魈",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_xinyan_mrmocciai": {
        "files": ["model/weights/xinyan-v2.pth", "model/xinyan-v2/added_IVF971_Flat_nprobe_1_xinyan_v2.index"],
        "zh_name": "辛焱",
        "en_name": "Xinyan",
        "jp_name": "辛炎",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_yae_miko_mrmocciai": {
        "files": ["model/weights/yae-v2.pth", "model/yae-v2/added_IVF1097_Flat_nprobe_1_yae-v2_v2.index"],
        "zh_name": "八重神子",
        "en_name": "Yae Miko",
        "jp_name": "八重神子",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_yanfei_mrmocciai": {
        "files": ["model/weights/yanfei.pth", "model/yanfei/added_IVF1271_Flat_nprobe_1_yanfei-v2_v2.index"],
        "zh_name": "烟绯",
        "en_name": "Yanfei",
        "jp_name": "煙緋",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_yelan_mrmocciai": {
        "files": ["model/weights/yelan-v2.pth", "model/yelan-v2/added_IVF1017_Flat_nprobe_1_yelan-v2_v2.index"],
        "zh_name": "夜兰",
        "en_name": "Yelan",
        "jp_name": "夜蘭",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_yoimiya_mrmocciai": {
        "files": ["model/weights/yoimiya-v2.pth", "model/yoimiya-v2/added_IVF871_Flat_nprobe_1_yoimiya-v2_v2.index"],
        "zh_name": "宵宫",
        "en_name": "Yoimiya",
        "jp_name": "宵宮",
        "variant": "v2",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    "genshin_zhongli_mrmocciai": {
        "files": ["model/weights/zhongli_rmvpe.pth", "model/zhongli_rmvpe/added_IVF1059_Flat_nprobe_1_zhongli_rmvpe_v2.index"],
        "zh_name": "钟离",
        "en_name": "Zhongli",
        "jp_name": "鍾離",
        "variant": "RMVPE",
        "source": "原神",
        "repo": "mrmocciai/genshin-impact",
    },
    # ===== Hololive / VTuber：Kit-Lemonfoot zip 模型 =====
    "azki_kit_hybrid": {
        "file": "AZKi (Hybrid).zip",
        "zh_name": "AZKi",
        "en_name": "AZKi",
        "jp_name": "AZKi",
        "variant": "Hybrid",
        "role": "混合 RVC 模型",
        "source": "Hololive Japan",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "azki_black_kit_singing": {
        "file": "AZKi BLaCK (Singing)(KitLemonfoot).zip",
        "zh_name": "AZKi BLaCK",
        "en_name": "AZKi BLaCK",
        "jp_name": "AZKi BLaCK",
        "variant": "Singing",
        "role": "歌唱 RVC 模型",
        "source": "Hololive Japan",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "airani_iofifteen_kit": {
        "file": "Airani Iofifteen (Speaking).zip",
        "zh_name": "艾拉妮·伊欧菲芙汀",
        "en_name": "Airani Iofifteen",
        "jp_name": "アイラニ・イオフィフティーン",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Hololive Indonesia",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "cecilia_immergreen_kit": {
        "file": "Cecilia Immergreen (Singing)(KitLemonfoot).zip",
        "zh_name": "塞西莉亚·伊默格林",
        "en_name": "Cecilia Immergreen",
        "jp_name": "セシリア・イマーグリーン",
        "variant": "Singing",
        "role": "歌唱 RVC 模型",
        "source": "Hololive English",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "ichijou_ririka_kit": {
        "file": "Ichijou Ririka (Speaking)(KitLemonfoot).zip",
        "zh_name": "一条莉莉华",
        "en_name": "Ichijou Ririka",
        "jp_name": "一条莉々華",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Hololive Japan",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "kanade_izuru_kit": {
        "file": "Kanade Izuru (Singing).zip",
        "zh_name": "奏手一弦",
        "en_name": "Kanade Izuru",
        "jp_name": "奏手イヅル",
        "variant": "Singing",
        "role": "歌唱 RVC 模型",
        "source": "Holostars Japan",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "shiranui_flare_kit": {
        "file": "Shiranui Flare (Speaking)(KitLemonfoot).zip",
        "zh_name": "不知火芙蕾雅",
        "en_name": "Shiranui Flare",
        "jp_name": "不知火フレア",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Hololive Japan",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "shirogane_noel_kit": {
        "file": "Shirogane Noel (Speaking).zip",
        "zh_name": "白银诺艾尔",
        "en_name": "Shirogane Noel",
        "jp_name": "白銀ノエル",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Hololive Japan",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "tsukumo_sana_kit": {
        "file": "Tsukumo Sana (Singing).zip",
        "zh_name": "九十九佐命",
        "en_name": "Tsukumo Sana",
        "jp_name": "九十九佐命",
        "variant": "Singing",
        "role": "歌唱 RVC 模型",
        "source": "Hololive English",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "regis_altare_kit": {
        "file": "Regis Altare (Speaking)(KitLemonfoot).zip",
        "zh_name": "雷吉斯·阿尔泰尔",
        "en_name": "Regis Altare",
        "jp_name": "リージス・アルテア",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Holostars English",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "axel_syrios_kit": {
        "file": "Axel Syrios (Speaking)(KitLemonfoot).zip",
        "zh_name": "阿克塞尔·西里奥斯",
        "en_name": "Axel Syrios",
        "jp_name": "アクセル・シリオス",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Holostars English",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "octavio_kit": {
        "file": "Octavio (Speaking)(KitLemonfoot).zip",
        "zh_name": "奥克塔维奥",
        "en_name": "Octavio",
        "jp_name": "オクタビオ",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Holostars English",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "crimzon_ruze_kit": {
        "file": "Crimzon Ruze (Speaking)(KitLemonfoot).zip",
        "zh_name": "克里姆森·鲁兹",
        "en_name": "Crimzon Ruze",
        "jp_name": "クリムゾン・ルーズ",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "Holostars English",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "finana_ryugu_kit": {
        "file": "Finana Ryugu (Hybrid)(KitLemonfoot).zip",
        "zh_name": "菲娜娜·龙宫",
        "en_name": "Finana Ryugu",
        "jp_name": "フィナーナ竜宮",
        "variant": "Hybrid",
        "role": "混合 RVC 模型",
        "source": "NIJISANJI English",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
    "mika_melatika_kit": {
        "file": "Mika Melatika (Speaking)(KitLemonfoot).zip",
        "zh_name": "米卡·梅拉提卡",
        "en_name": "Mika Melatika",
        "jp_name": "ミカ・メラティカ",
        "variant": "Speaking",
        "role": "说话 RVC 模型",
        "source": "NIJISANJI Indonesia",
        "repo": "Kit-Lemonfoot/kitlemonfoot_rvc_models",
    },
})


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def get_character_models_dir() -> Path:
    """获取角色模型目录"""
    return get_project_root() / "assets" / "weights" / "characters"


def _uploaded_file_path(file_value: Any) -> Optional[Path]:
    if not file_value:
        return None
    if isinstance(file_value, (str, os.PathLike)):
        return Path(file_value)
    name = getattr(file_value, "name", None)
    if name:
        return Path(name)
    raise TypeError(f"无法识别上传文件对象: {type(file_value).__name__}")


def _sanitize_custom_model_key(text: str) -> str:
    key = re.sub(r"[^\w.-]+", "_", str(text or ""), flags=re.UNICODE).strip("._-")
    key = re.sub(r"_+", "_", key)
    if not key:
        raise ValueError("自定义模型名称不能为空")
    return key[:80]


def _assert_under_directory(path: Path, directory: Path):
    resolved_path = path.resolve()
    resolved_dir = directory.resolve()
    if not resolved_path.is_relative_to(resolved_dir):
        raise ValueError(f"目标路径越界: {resolved_path}")


def _copy_custom_file(source: Path, dest_dir: Path, allowed_suffixes: set) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"上传文件不存在: {source}")
    suffix = source.suffix.lower()
    if suffix not in allowed_suffixes:
        expected = ", ".join(sorted(allowed_suffixes))
        raise ValueError(f"不支持的文件类型: {source.name}，只支持 {expected}")
    target = dest_dir / source.name
    _assert_under_directory(target, dest_dir)
    if target.exists():
        raise FileExistsError(f"目标文件已存在: {target.name}")
    shutil.copy2(source, target)
    return target


def _extract_custom_model_zip(source: Path, dest_dir: Path) -> List[Path]:
    if not source.exists():
        raise FileNotFoundError(f"上传文件不存在: {source}")
    if source.suffix.lower() != ".zip":
        raise ValueError(f"不支持的压缩包类型: {source.name}，只支持 .zip")

    extracted: List[Path] = []
    with zipfile.ZipFile(source, "r") as zip_ref:
        members = [
            member for member in zip_ref.infolist()
            if not member.is_dir()
        ]
        pth_members = [
            member for member in members
            if Path(member.filename).suffix.lower() == ".pth"
        ]
        if not pth_members:
            raise ValueError("压缩包中没有 .pth 权重文件")
        if len(pth_members) > 1:
            names = ", ".join(Path(member.filename).name for member in pth_members)
            raise ValueError(f"压缩包中包含多个 .pth 文件，请拆分后上传: {names}")

        allowed_suffixes = {".pth", ".index", ".json"}
        for member in members:
            suffix = Path(member.filename).suffix.lower()
            if suffix not in allowed_suffixes:
                continue
            target = dest_dir / Path(member.filename).name
            _assert_under_directory(target, dest_dir)
            if target.exists():
                raise FileExistsError(f"目标文件已存在: {target.name}")
            with zip_ref.open(member, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)

    return extracted


def import_custom_character_model(
    model_file: Any,
    index_file: Any = None,
    display_name: str = "",
    source: str = "自定义模型",
    category: str = "自定义 / 手动上传",
) -> Dict:
    """
    导入用户手动上传的 RVC 模型。

    支持单个 .pth、.pth + .index，或包含一个 .pth 的 .zip。
    """
    model_path = _uploaded_file_path(model_file)
    index_path = _uploaded_file_path(index_file)
    if model_path is None:
        raise ValueError("请上传 .pth 权重文件或包含一个 .pth 的 .zip")

    stem = model_path.stem
    model_name = str(display_name or stem).strip()
    key = _sanitize_custom_model_key(model_name)
    models_dir = get_character_models_dir()
    char_dir = models_dir / key
    _assert_under_directory(char_dir, models_dir)
    created_dir = not char_dir.exists()
    if char_dir.exists() and list(char_dir.glob("*.pth")):
        raise FileExistsError(f"自定义模型已存在: {key}")

    char_dir.mkdir(parents=True, exist_ok=True)
    try:
        if model_path.suffix.lower() == ".zip":
            _extract_custom_model_zip(model_path, char_dir)
        else:
            _copy_custom_file(model_path, char_dir, {".pth"})

        if index_path is not None:
            _copy_custom_file(index_path, char_dir, {".index"})

        pth_files = sorted(char_dir.glob("*.pth"))
        if len(pth_files) != 1:
            raise ValueError(f"导入后必须只有一个 .pth 权重文件，当前数量: {len(pth_files)}")

        info = {
            "zh_name": model_name,
            "source": str(source or "自定义模型").strip() or "自定义模型",
            "category": str(category or "自定义 / 手动上传").strip() or "自定义 / 手动上传",
            "distribution": "手动上传",
            "role": "自定义 RVC 模型",
        }
        _write_local_model_info(key, char_dir, info)
        record = _build_character_record(key, info)
        index = _find_index_file(pth_files[0])
        record.update({
            "model_path": str(pth_files[0]),
            "index_path": str(index) if index else None,
        })
        return record
    except Exception:
        if created_dir and char_dir.exists():
            shutil.rmtree(char_dir)
        raise


def list_available_characters() -> List[Dict]:
    """
    列出可用的角色模型

    Returns:
        list: 角色信息列表
    """
    result = []
    for name, info in CHARACTER_MODELS.items():
        result.append(_build_character_record(name, info))
    result.sort(
        key=lambda item: (
            str(item.get("series", "")),
            str(item.get("base_display", item.get("display", ""))),
            str(item.get("continuity", "")),
            str(item.get("version_label", "")),
            str(item.get("name", "")),
        )
    )
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

        info = dict(CHARACTER_MODELS.get(char_name, {}))
        local_info = _load_local_model_info(models_dir / char_name)
        if local_info:
            info.update({
                key: value for key, value in local_info.items()
                if value not in (None, "", [], {})
            })
        record = _build_character_record(char_name, info)
        index_file = _find_index_file(pth_file)
        record.update({
            "model_path": str(pth_file),
            "index_path": str(index_file) if index_file else None,
        })
        downloaded.append(record)

    downloaded.sort(
        key=lambda item: (
            str(item.get("series", "")),
            str(item.get("base_display", item.get("display", ""))),
            str(item.get("continuity", "")),
            str(item.get("version_label", "")),
            str(item.get("name", "")),
        )
    )
    return downloaded


def get_character_info(name: str, downloaded_only: bool = False) -> Optional[Dict]:
    if not name:
        return None
    if downloaded_only:
        pool = list_downloaded_characters()
    else:
        pool = list_downloaded_characters() + list_available_characters()
    seen = set()
    for item in pool:
        item_name = item.get("name")
        if item_name in seen:
            continue
        seen.add(item_name)
        if item_name == name:
            return item
    return None


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
        _write_local_model_info(name, char_dir, char_info)
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

        _write_local_model_info(name, char_dir, char_info)
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
        series_set.add(_get_character_category(info))
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
