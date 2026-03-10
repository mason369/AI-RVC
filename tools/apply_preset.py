#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置预设应用工具
快速切换不同的音质优化配置
"""
import json
import shutil
from pathlib import Path

PRESETS_DIR = Path("configs/presets")
CONFIG_FILE = Path("configs/config.json")
BACKUP_FILE = Path("configs/config.backup.json")

PRESETS = {
    "1": "balanced.json",
    "2": "clarity_priority.json",
    "3": "timbre_priority.json"
}

def load_json(path: Path) -> dict:
    """加载JSON文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: dict):
    """保存JSON文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def backup_config():
    """备份当前配置"""
    if CONFIG_FILE.exists():
        shutil.copy(CONFIG_FILE, BACKUP_FILE)
        print(f"✓ 已备份当前配置到: {BACKUP_FILE}")

def restore_config():
    """恢复备份配置"""
    if BACKUP_FILE.exists():
        shutil.copy(BACKUP_FILE, CONFIG_FILE)
        print(f"✓ 已恢复配置从: {BACKUP_FILE}")
    else:
        print("✗ 未找到备份文件")

def apply_preset(preset_name: str):
    """应用预设配置"""
    preset_path = PRESETS_DIR / preset_name

    if not preset_path.exists():
        print(f"✗ 预设文件不存在: {preset_path}")
        return

    # 备份当前配置
    backup_config()

    # 加载预设和当前配置
    preset = load_json(preset_path)
    config = load_json(CONFIG_FILE)

    # 合并配置 (只更新 cover 部分)
    if "cover" in preset:
        config["cover"].update(preset["cover"])

    # 保存
    save_json(CONFIG_FILE, config)

    print(f"\n✓ 已应用预设: {preset.get('name', preset_name)}")
    print(f"  描述: {preset.get('description', 'N/A')}")
    print(f"\n主要参数:")
    print(f"  - index_rate: {config['cover']['index_rate']}")
    print(f"  - protect: {config['cover']['protect']}")
    print(f"  - rms_mix_rate: {config['cover']['rms_mix_rate']}")
    print(f"  - filter_radius: {config['cover']['filter_radius']}")
    print(f"  - f0_stabilize: {config['cover']['f0_stabilize']}")

def show_menu():
    """显示菜单"""
    print("\n" + "="*60)
    print("RVC 音质优化配置预设")
    print("="*60)
    print("\n可用预设:")
    print("  1. 平衡型配置 (推荐)")
    print("     - 适合大多数歌曲")
    print("     - 在音色转换和清晰度之间平衡")
    print()
    print("  2. 清晰度优先配置")
    print("     - 减少伪影和失真")
    print("     - 适合复杂歌曲和高音多的情况")
    print("     - 保留更多源音频特征")
    print()
    print("  3. 音色优先配置")
    print("     - 彻底的音色转换")
    print("     - 适合音色特征明显的角色")
    print("     - 可能有轻微口齿模糊")
    print()
    print("  r. 恢复备份配置")
    print("  q. 退出")
    print("="*60)

def main():
    """主函数"""
    while True:
        show_menu()
        choice = input("\n请选择 (1-3/r/q): ").strip().lower()

        if choice == "q":
            print("\n再见!")
            break
        elif choice == "r":
            restore_config()
        elif choice in PRESETS:
            apply_preset(PRESETS[choice])
        else:
            print("\n✗ 无效选择，请重试")

        input("\n按回车继续...")

if __name__ == "__main__":
    main()
