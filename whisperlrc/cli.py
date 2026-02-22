from __future__ import annotations

import os
from pathlib import Path

from whisperlrc.config import load_config
from whisperlrc.logging import setup_logging


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _read_menu_key() -> str:
    try:
        import msvcrt

        while True:
            ch = msvcrt.getch()
            if not ch:
                continue
            try:
                k = ch.decode("utf-8").lower()
            except UnicodeDecodeError:
                continue
            if k in {"1", "2", "3", "4", "q"}:
                return k
    except Exception:
        return input("请选择 [1/2/3/4/q]：").strip().lower()


def _prompt_path(label: str, default: str | None = None) -> Path:
    suffix = f" ({default})" if default else ""
    v = input(f"{label}{suffix}：").strip()
    if not v and default:
        v = default
    return Path(v)


def _run_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    config: Path,
) -> int:
    from whisperlrc.pipeline import process_batch

    setup_logging("INFO")
    cfg = load_config(config)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[错误] 输入目录不存在：{input_dir}")
        return 2
    return process_batch(input_dir, output_dir, cfg)


def _show_config_summary(config: Path) -> None:
    cfg = load_config(config)
    print("\n当前配置摘要")
    print("------------")
    print(f"配置文件：{config}")
    print(f"ASR 后端：{cfg.asr.backend}")
    print(f"ASR 模型：{cfg.asr.model}")
    print(f"识别语言：{cfg.asr.language}")
    print(f"运行设备：{cfg.asr.device}")
    print(f"翻译后端：{cfg.translation.backend}")
    print(f"翻译目标：{cfg.translation.target}")
    print(f"支持输入格式：{', '.join(cfg.pipeline.input_formats)}")
    print(f"ASR 重试次数：{cfg.pipeline.retry_asr}")
    print(f"翻译重试次数：{cfg.pipeline.retry_translate}")
    print()


def _check_translation_backend(config: Path) -> None:
    from whisperlrc.translate.factory import build_translator

    cfg = load_config(config)
    print("\n翻译后端检查")
    print("------------")
    print(f"配置文件：{config}")
    print(f"翻译后端：{cfg.translation.backend}")
    try:
        translator = build_translator(cfg.translation)
        translator.translate_batch(["测试"], src="ja", tgt=cfg.translation.target)
        print("翻译功能可用。")
    except NotImplementedError as e:
        print(f"翻译功能尚未实现：{e}")
    except Exception as e:
        print(f"翻译后端检查失败：{e}")
    print()


def run_interactive_menu() -> int:
    setup_logging("INFO")
    while True:
        _clear_screen()
        print("WhisperLRC 交互式菜单")
        print("====================")
        print("1) 使用当前配置开始批量处理")
        print("2) 设置本次输入/输出路径并开始")
        print("3) 查看当前配置摘要")
        print("4) 翻译后端状态检查")
        print("q) 退出")
        print()
        print("按键选择：", end="", flush=True)
        key = _read_menu_key()
        print(key)

        if key == "q":
            return 0
        if key == "3":
            _show_config_summary(Path("settings.toml"))
            input("按回车继续...")
            continue
        if key == "4":
            config = _prompt_path("配置文件路径", "settings.toml")
            _check_translation_backend(config)
            input("按回车继续...")
            continue
        if key == "1":
            input_dir = _prompt_path("输入目录", "input_audio")
            output_dir = _prompt_path("输出目录", "output")
            rc = _run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                config=Path("settings.toml"),
            )
            print(f"\n处理完成，退出码：{rc}")
            input("按回车继续...")
            continue
        if key == "2":
            input_dir = _prompt_path("输入目录", "input_audio")
            output_dir = _prompt_path("输出目录", "output")
            config = _prompt_path("配置文件路径", "settings.toml")
            rc = _run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
            )
            print(f"\n处理完成，退出码：{rc}")
            input("按回车继续...")
            continue
