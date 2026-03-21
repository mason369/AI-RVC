#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回声处理配置验证脚本

检查当前配置是否正确启用了激进的去回声处理
"""

import json
import os
from pathlib import Path

def check_deecho_models():
    """检查 DeEcho 模型是否存在"""
    print("=" * 60)
    print("检查 DeEcho 模型")
    print("=" * 60)

    models_dir = Path("assets/uvr5_weights")
    required_models = [
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoNormal.pth",
        "onnx_dereverb_By_FoxJoy"
    ]

    all_found = True
    for model in required_models:
        model_path = models_dir / model
        exists = model_path.exists()
        status = "[OK]" if exists else "[NO]"
        size = ""
        if exists:
            if model_path.is_file():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                size = f" ({size_mb:.1f} MB)"
            else:
                size = " (目录)"

        print(f"{status} {model}{size}")
        if not exists:
            all_found = False

    print()
    if all_found:
        print("[OK] All DeEcho models ready")
    else:
        print("[NO] Missing models, run: python tools/download_models.py")

    return all_found

def check_config():
    """检查配置文件"""
    print("\n" + "=" * 60)
    print("检查配置文件")
    print("=" * 60)

    config_path = Path("configs/config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    cover_config = config.get("cover", {})

    # 检查关键配置
    checks = [
        ("VC 预处理模式", "vc_preprocess_mode", "auto", cover_config.get("vc_preprocess_mode")),
        ("源约束模式", "source_constraint_mode", "auto", cover_config.get("source_constraint_mode")),
        ("Karaoke 分离", "karaoke_separation", True, cover_config.get("karaoke_separation")),
        ("索引率", "index_rate", 0.50, cover_config.get("index_rate")),
        ("保护系数", "protect", 0.33, cover_config.get("protect")),
    ]

    all_correct = True
    for name, key, expected, actual in checks:
        match = actual == expected
        status = "[OK]" if match else "[NO]"
        print(f"{status} {name} ({key}): {actual} {'==' if match else '!='} {expected}")
        if not match:
            all_correct = False

    print()
    if all_correct:
        print("[OK] Config optimized for aggressive deecho")
    else:
        print("[WARN] Some configs not optimized")

    return all_correct

def print_recommendations():
    """打印使用建议"""
    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)

    print("""
1. 当前配置使用自动模式：
   - 优先使用 UVR DeEcho 模型，缺模型时回退到算法去混响
   - DeEcho 质量好时跳过 blend 直接使用，避免混回原始回音
   - 源约束仅在去过回音的预处理下自动启用

2. 如果回声仍然明显，可以尝试：
   - 在 UI 中调整"索引率"（降低到 0.2-0.3）
   - 在 UI 中调整"保护系数"（降低到 0.2-0.25）
   - 使用更高质量的输入音频

3. 处理流程：
   原始音频 → Karaoke 分离 → UVR DeEcho(或算法去混响) → RVC 转换 → 输出

4. 测试建议：
   - 选择一首有明显回声的歌曲
   - 查看日志中 DeEcho quality 指标
   - 对比处理前后的回声强度
""")

def main():
    print("\n回声处理配置验证\n")

    models_ok = check_deecho_models()
    config_ok = check_config()
    print_recommendations()

    print("\n" + "=" * 60)
    if models_ok and config_ok:
        print("[OK] System ready for testing")
    else:
        print("[WARN] Please fix issues above")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
