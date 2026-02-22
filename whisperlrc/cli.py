from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    INFO = auto()


@dataclass
class SessionState:
    config_path: Path = Path("settings.toml")
    input_dir: Path = Path("input_audio")
    output_dir: Path = Path("output")
    info_title: str = ""
    info_lines: list[str] = field(default_factory=list)
    info_back: Page = Page.MAIN
    info_path: str = "主菜单"


@dataclass
class LineInputResult:
    kind: str
    value: str = ""


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


def _read_line_with_cancel(label: str, default: str | None = None) -> LineInputResult:
    suffix = f" (当前: {default})" if default else ""
    prompt = f"{label}{suffix}（回车取消，q 主菜单，Esc 返回）："

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
                    if not text:
                        return LineInputResult("cancel")
                    if text.lower() == "q":
                        return LineInputResult("main")
                    return LineInputResult("value", text)
                if ch == "\x1b":
                    print()
                    return LineInputResult("cancel")
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
        return LineInputResult("cancel")
    if text == "\x1b":
        return LineInputResult("cancel")
    if text.lower() == "q":
        return LineInputResult("main")
    return LineInputResult("value", text)


def _confirm_action(prompt: str) -> str:
    print(f"{prompt} [y/n]（q 主菜单，Esc 返回，默认 n）")
    key = _read_single_key({"y", "n", "q"}, allow_esc=True)
    return key or "n"


def _print_path_bar(path: str) -> None:
    print(f"[{path}]")
    print()


def _show_info(
    state: SessionState,
    title: str,
    lines: list[str],
    back_page: Page,
    path: str,
) -> Page:
    state.info_title = title
    state.info_lines = lines
    state.info_back = back_page
    state.info_path = path
    return Page.INFO


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
        raise ValueError(f"输入目录不存在：{input_dir}")
    return process_batch(input_dir, output_dir, cfg)


def _build_config_summary(config: Path) -> list[str]:
    cfg = load_config(config)
    return [
        f"配置文件：{config}",
        f"ASR 后端：{cfg.asr.backend}",
        f"ASR 模型：{cfg.asr.model}",
        f"识别语言：{cfg.asr.language}",
        f"运行设备：{cfg.asr.device}",
        f"翻译后端：{cfg.translation.backend}",
        f"翻译目标：{cfg.translation.target}",
        f"支持输入格式：{', '.join(cfg.pipeline.input_formats)}",
        f"ASR 重试次数：{cfg.pipeline.retry_asr}",
        f"翻译重试次数：{cfg.pipeline.retry_translate}",
    ]


def _build_translation_check(config: Path) -> list[str]:
    from whisperlrc.translate.factory import build_translator

    cfg = load_config(config)
    lines = [
        f"配置文件：{config}",
        f"翻译后端：{cfg.translation.backend}",
    ]
    try:
        translator = build_translator(cfg.translation)
        translator.translate_batch(["测试"], src="ja", tgt=cfg.translation.target)
        lines.append("结果：翻译功能可用")
    except NotImplementedError as e:
        lines.append(f"结果：翻译功能尚未实现：{e}")
    except Exception as e:
        lines.append(f"结果：翻译后端检查失败：{e}")
    return lines


def _render_main_page() -> Page | None:
    _print_path_bar("主菜单")
    print("WhisperLRC 主菜单")
    print("===============")
    print("1) 批处理")
    print("2) 配置")
    print("3) 检查")
    print("4) 帮助")
    print()
    print("q 返回主菜单（当前位置），Esc 退出")

    key = _read_single_key({"1", "2", "3", "4", "q"}, allow_esc=True)
    if key == "esc":
        return None
    if key == "q":
        return Page.MAIN
    if key == "1":
        return Page.BATCH
    if key == "2":
        return Page.CONFIG
    if key == "3":
        return Page.CHECK
    if key == "4":
        return Page.HELP
    return Page.MAIN


def _render_batch_page(state: SessionState) -> Page:
    _print_path_bar("主菜单->批处理")
    print("批处理页面")
    print("==========")
    print("1) 使用当前会话默认参数执行")
    print("2) 自定义本次路径并执行")
    print("3) 查看当前运行参数")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"1", "2", "3", "q"}, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return Page.MAIN

    if key == "1":
        confirm = _confirm_action("确认开始执行？")
        if confirm == "q":
            return Page.MAIN
        if confirm != "y":
            return _show_info(state, "执行结果", ["已取消执行。"], Page.BATCH, "主菜单->批处理->执行结果")
        try:
            rc = _run_batch(
                input_dir=state.input_dir,
                output_dir=state.output_dir,
                config=state.config_path,
            )
            return _show_info(state, "执行结果", [f"处理完成，退出码：{rc}"], Page.BATCH, "主菜单->批处理->执行结果")
        except Exception as e:
            return _show_info(state, "执行结果", [f"处理失败：{e}"], Page.BATCH, "主菜单->批处理->执行结果")

    if key == "2":
        input_res = _read_line_with_cancel("输入目录", str(state.input_dir))
        if input_res.kind == "main":
            return Page.MAIN
        if input_res.kind != "value":
            return _show_info(state, "执行结果", ["已取消本次自定义执行。"], Page.BATCH, "主菜单->批处理->执行结果")

        output_res = _read_line_with_cancel("输出目录", str(state.output_dir))
        if output_res.kind == "main":
            return Page.MAIN
        if output_res.kind != "value":
            return _show_info(state, "执行结果", ["已取消本次自定义执行。"], Page.BATCH, "主菜单->批处理->执行结果")

        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "执行结果", ["已取消本次自定义执行。"], Page.BATCH, "主菜单->批处理->执行结果")

        confirm = _confirm_action("确认开始执行？")
        if confirm == "q":
            return Page.MAIN
        if confirm != "y":
            return _show_info(state, "执行结果", ["已取消执行。"], Page.BATCH, "主菜单->批处理->执行结果")
        try:
            rc = _run_batch(
                input_dir=Path(input_res.value),
                output_dir=Path(output_res.value),
                config=Path(config_res.value),
            )
            return _show_info(state, "执行结果", [f"处理完成，退出码：{rc}"], Page.BATCH, "主菜单->批处理->执行结果")
        except Exception as e:
            return _show_info(state, "执行结果", [f"处理失败：{e}"], Page.BATCH, "主菜单->批处理->执行结果")

    if key == "3":
        lines = [
            f"输入目录：{state.input_dir}",
            f"输出目录：{state.output_dir}",
            f"配置文件：{state.config_path}",
        ]
        return _show_info(state, "当前运行参数", lines, Page.BATCH, "主菜单->批处理->当前运行参数")

    return Page.BATCH


def _render_config_page(state: SessionState) -> Page:
    _print_path_bar("主菜单->配置")
    print("配置页面")
    print("========")
    print("1) 查看当前配置摘要")
    print("2) 切换会话配置文件路径")
    print("3) 重置会话配置为 settings.toml")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"1", "2", "3", "q"}, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return Page.MAIN

    if key == "1":
        try:
            return _show_info(state, "当前配置摘要", _build_config_summary(state.config_path), Page.CONFIG, "主菜单->配置->当前配置摘要")
        except Exception as e:
            return _show_info(state, "当前配置摘要", [f"读取失败：{e}"], Page.CONFIG, "主菜单->配置->当前配置摘要")

    if key == "2":
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "配置结果", ["已取消修改配置路径。"], Page.CONFIG, "主菜单->配置->配置结果")
        state.config_path = Path(config_res.value)
        return _show_info(state, "配置结果", [f"已更新会话配置：{state.config_path}"], Page.CONFIG, "主菜单->配置->配置结果")

    if key == "3":
        state.config_path = Path("settings.toml")
        return _show_info(state, "配置结果", ["已重置会话配置为 settings.toml"], Page.CONFIG, "主菜单->配置->配置结果")

    return Page.CONFIG


def _render_check_page(state: SessionState) -> Page:
    _print_path_bar("主菜单->检查")
    print("检查页面")
    print("========")
    print("1) 检查当前会话配置的翻译后端")
    print("2) 使用自定义配置文件检查")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"1", "2", "q"}, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return Page.MAIN

    if key == "1":
        try:
            return _show_info(state, "翻译后端检查", _build_translation_check(state.config_path), Page.CHECK, "主菜单->检查->翻译后端检查")
        except Exception as e:
            return _show_info(state, "翻译后端检查", [f"检查失败：{e}"], Page.CHECK, "主菜单->检查->翻译后端检查")

    if key == "2":
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "翻译后端检查", ["已取消本次检查。"], Page.CHECK, "主菜单->检查->翻译后端检查")
        try:
            return _show_info(state, "翻译后端检查", _build_translation_check(Path(config_res.value)), Page.CHECK, "主菜单->检查->翻译后端检查")
        except Exception as e:
            return _show_info(state, "翻译后端检查", [f"检查失败：{e}"], Page.CHECK, "主菜单->检查->翻译后端检查")

    return Page.CHECK


def _render_help_page() -> Page:
    _print_path_bar("主菜单->帮助")
    print("帮助页面")
    print("========")
    print("- 数字键：即时执行当前页面选项")
    print("- q：返回主菜单")
    print("- Esc：全局返回（在主菜单中 Esc 退出）")
    print("- 关键动作会要求 y/n 二次确认")
    print("- 信息结果以单独页面展示，Esc 关闭")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"q"}, allow_esc=True)
    if key in {"q", "esc"}:
        return Page.MAIN
    return Page.HELP


def _render_info_page(state: SessionState) -> Page:
    _print_path_bar(state.info_path)
    print(state.info_title)
    print("=" * len(state.info_title))
    for line in state.info_lines:
        print(line)
    print()
    print("Esc 返回，q 主菜单")

    key = _read_single_key({"q"}, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return state.info_back
    return Page.INFO


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
        if page == Page.INFO:
            page = _render_info_page(state)
            continue

    return 0
