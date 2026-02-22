from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from whisperlrc.config import load_config
from whisperlrc.logging import setup_logging


class Page(Enum):
    MAIN = auto()
    BATCH = auto()
    CONFIG = auto()
    CHECK = auto()
    HELP = auto()


@dataclass
class SessionState:
    config_path: Path = Path("settings.toml")
    input_dir: Path = Path("input_audio")
    output_dir: Path = Path("output")


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _read_single_key(valid_keys: set[str], *, allow_esc: bool = True) -> str:
    normalized = {k.lower() for k in valid_keys}
    try:
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if not ch:
                continue
            if ch in {"\x00", "\xe0"}:
                msvcrt.getwch()
                continue
            if allow_esc and ch == "\x1b":
                return "esc"
            key = ch.lower()
            if key in normalized:
                return key
    except Exception:
        text = input().strip().lower()
        if allow_esc and text == "\x1b":
            return "esc"
        return text if text in normalized else ""


def _read_line_with_cancel(label: str, default: str | None = None) -> str | None:
    suffix = f" (当前: {default})" if default else ""
    prompt = f"{label}{suffix}（回车取消，Esc 取消）："

    if os.name == "nt":
        try:
            import msvcrt

            print(prompt, end="", flush=True)
            chars: list[str] = []
            while True:
                ch = msvcrt.getwch()
                if ch in {"\r", "\n"}:
                    print()
                    text = "".join(chars).strip()
                    return text or None
                if ch == "\x1b":
                    print()
                    return None
                if ch in {"\x00", "\xe0"}:
                    msvcrt.getwch()
                    continue
                if ch in {"\b", "\x7f"}:
                    if chars:
                        chars.pop()
                        print("\b \b", end="", flush=True)
                    continue
                chars.append(ch)
                print(ch, end="", flush=True)
        except Exception:
            pass

    text = input(prompt).strip()
    if not text:
        return None
    if text == "\x1b":
        return None
    return text


def _pause(message: str = "按任意键继续...") -> None:
    print(message, end="", flush=True)
    try:
        import msvcrt

        msvcrt.getch()
        print()
    except Exception:
        input()


def _confirm_yes_no(prompt: str) -> bool:
    print(f"{prompt} [y/n]（默认 n）：", end="", flush=True)
    key = _read_single_key({"y", "n", "q"}, allow_esc=True)
    if key:
        print(key if key != "esc" else "esc")
    return key == "y"


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


def _render_main_page() -> Page | None:
    print("WhisperLRC 主菜单")
    print("===============")
    print("1) 批处理")
    print("2) 配置")
    print("3) 检查")
    print("4) 帮助")
    print("q) 退出程序")
    print()
    print("按键选择（Esc 退出）：", end="", flush=True)
    key = _read_single_key({"1", "2", "3", "4", "q"}, allow_esc=True)
    print(key if key != "esc" else "esc")

    if key in {"q", "esc"}:
        return None
    if key == "1":
        return Page.BATCH
    if key == "2":
        return Page.CONFIG
    if key == "3":
        return Page.CHECK
    if key == "4":
        return Page.HELP
    return Page.MAIN


def _show_runtime_preview(state: SessionState) -> None:
    print("\n当前运行参数")
    print("------------")
    print(f"输入目录：{state.input_dir}")
    print(f"输出目录：{state.output_dir}")
    print(f"配置文件：{state.config_path}")
    print()


def _run_batch_with_confirm(input_dir: Path, output_dir: Path, config: Path) -> None:
    print("\n即将执行批处理：")
    print(f"- 输入目录：{input_dir}")
    print(f"- 输出目录：{output_dir}")
    print(f"- 配置文件：{config}")
    if not _confirm_yes_no("确认开始执行？"):
        print("已取消执行。")
        _pause()
        return

    rc = _run_batch(input_dir=input_dir, output_dir=output_dir, config=config)
    print(f"\n处理完成，退出码：{rc}")
    _pause()


def _render_batch_page(state: SessionState) -> Page:
    print("批处理页面")
    print("==========")
    print("1) 使用当前会话默认参数执行")
    print("2) 自定义本次路径并执行")
    print("3) 查看当前运行参数")
    print("q) 返回主菜单")
    print()
    print("按键选择（Esc 返回）：", end="", flush=True)
    key = _read_single_key({"1", "2", "3", "q"}, allow_esc=True)
    print(key if key != "esc" else "esc")

    if key in {"q", "esc"}:
        return Page.MAIN
    if key == "1":
        _run_batch_with_confirm(state.input_dir, state.output_dir, state.config_path)
        return Page.BATCH
    if key == "2":
        input_text = _read_line_with_cancel("输入目录", str(state.input_dir))
        if input_text is None:
            print("已取消。")
            _pause()
            return Page.BATCH
        output_text = _read_line_with_cancel("输出目录", str(state.output_dir))
        if output_text is None:
            print("已取消。")
            _pause()
            return Page.BATCH
        config_text = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_text is None:
            print("已取消。")
            _pause()
            return Page.BATCH
        _run_batch_with_confirm(Path(input_text), Path(output_text), Path(config_text))
        return Page.BATCH
    if key == "3":
        _show_runtime_preview(state)
        _pause()
        return Page.BATCH
    return Page.BATCH


def _render_config_page(state: SessionState) -> Page:
    print("配置页面")
    print("========")
    print("1) 查看当前配置摘要")
    print("2) 切换会话配置文件路径")
    print("3) 重置会话配置为 settings.toml")
    print("q) 返回主菜单")
    print()
    print("按键选择（Esc 返回）：", end="", flush=True)
    key = _read_single_key({"1", "2", "3", "q"}, allow_esc=True)
    print(key if key != "esc" else "esc")

    if key in {"q", "esc"}:
        return Page.MAIN
    if key == "1":
        _show_config_summary(state.config_path)
        _pause()
        return Page.CONFIG
    if key == "2":
        config_text = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_text is None:
            print("已取消。")
            _pause()
            return Page.CONFIG
        state.config_path = Path(config_text)
        print(f"已更新会话配置：{state.config_path}")
        _pause()
        return Page.CONFIG
    if key == "3":
        state.config_path = Path("settings.toml")
        print("已重置会话配置为 settings.toml")
        _pause()
        return Page.CONFIG
    return Page.CONFIG


def _render_check_page(state: SessionState) -> Page:
    print("检查页面")
    print("========")
    print("1) 检查当前会话配置的翻译后端")
    print("2) 使用自定义配置文件检查")
    print("q) 返回主菜单")
    print()
    print("按键选择（Esc 返回）：", end="", flush=True)
    key = _read_single_key({"1", "2", "q"}, allow_esc=True)
    print(key if key != "esc" else "esc")

    if key in {"q", "esc"}:
        return Page.MAIN
    if key == "1":
        _check_translation_backend(state.config_path)
        _pause()
        return Page.CHECK
    if key == "2":
        config_text = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_text is None:
            print("已取消。")
            _pause()
            return Page.CHECK
        _check_translation_backend(Path(config_text))
        _pause()
        return Page.CHECK
    return Page.CHECK


def _render_help_page() -> Page:
    print("帮助页面")
    print("========")
    print("- 数字键：即时执行当前页面选项")
    print("- q：返回上一级页面")
    print("- Esc：取消输入或返回上一级")
    print("- 批处理等关键动作会要求 y/n 二次确认")
    print("- 路径输入为空会取消当前操作")
    print()
    print("按 q 或 Esc 返回主菜单：", end="", flush=True)
    key = _read_single_key({"q"}, allow_esc=True)
    print(key if key != "esc" else "esc")
    return Page.MAIN


def run_interactive_menu() -> int:
    setup_logging("INFO")
    state = SessionState()
    page: Page | None = Page.MAIN

    while page is not None:
        _clear_screen()
        if page == Page.MAIN:
            page = _render_main_page()
            continue
        if page == Page.BATCH:
            page = _render_batch_page(state)
            continue
        if page == Page.CONFIG:
            page = _render_config_page(state)
            continue
        if page == Page.CHECK:
            page = _render_check_page(state)
            continue
        if page == Page.HELP:
            page = _render_help_page()
            continue

    return 0
