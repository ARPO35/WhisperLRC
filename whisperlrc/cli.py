from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from whisperlrc.config import AppConfig, load_config, save_config
from whisperlrc.logging import setup_logging


class Page(Enum):
    MAIN = auto()
    BATCH = auto()
    CONFIG = auto()
    CHECK = auto()
    HELP = auto()
    INFO = auto()
    CONFIG_EDIT_MENU = auto()
    CONFIG_EDIT_FIELDS = auto()


@dataclass
class SessionState:
    config_path: Path = Path("settings.toml")
    input_dir: Path = Path("input_audio")
    output_dir: Path = Path("output")
    info_title: str = ""
    info_lines: list[str] = field(default_factory=list)
    info_back: Page = Page.MAIN
    info_path: str = "主菜单"
    edit_cfg: AppConfig | None = None
    edit_section: str = "asr"
    edit_offset: int = 0


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
        f"提示词文件：{cfg.translation.llm_prompt_file}",
        f"偏好文件：{cfg.translation.llm_preferences_file}",
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


SECTION_TITLES: dict[str, str] = {
    "asr": "ASR",
    "pipeline": "管线",
    "translation": "翻译",
    "output": "输出",
    "schema": "Schema",
}


def _get_section_obj(cfg: AppConfig, section: str) -> Any:
    return getattr(cfg, section)


def _format_value_for_menu(value: Any) -> str:
    if isinstance(value, str):
        if not value:
            return '""'
        return value if len(value) <= 36 else value[:33] + "..."
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return f"{len(value)} 项"
        return ", ".join(str(v) for v in value) if value else "[]"
    return str(value)


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    if text not in mapping:
        raise ValueError("布尔值只接受 true/false/1/0/yes/no")
    return mapping[text]


def _parse_field_value(section: str, field_name: str, current: Any, text: str) -> Any:
    if isinstance(current, bool):
        return _parse_bool(text)
    if isinstance(current, int):
        return int(text)
    if isinstance(current, str):
        return "" if text == '""' else text
    if isinstance(current, list):
        if section == "pipeline" and field_name == "input_formats":
            if text.strip() == "[]":
                return []
            values = [v.strip() for v in text.split(",")]
            values = [v for v in values if v]
            if not values:
                raise ValueError("input_formats 不能为空，请输入如 mp3,wav")
            return values
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("列表字段需要 JSON 数组")
        return parsed
    raise ValueError("暂不支持该字段类型")


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
    print("4) 修改配置项（可保存到文件）")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"1", "2", "3", "4", "q"}, allow_esc=True)
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
    if key == "4":
        try:
            state.edit_cfg = load_config(state.config_path)
            state.edit_section = "asr"
            state.edit_offset = 0
            return Page.CONFIG_EDIT_MENU
        except Exception as e:
            return _show_info(state, "配置结果", [f"加载配置失败：{e}"], Page.CONFIG, "主菜单->配置->配置结果")

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


def _render_config_edit_menu_page(state: SessionState) -> Page:
    _print_path_bar("主菜单->配置->修改配置项")
    print("配置分区")
    print("========")
    print("1) ASR")
    print("2) 管线")
    print("3) 翻译")
    print("4) 输出")
    print("5) Schema")
    print("6) 保存并写入当前配置文件")
    print("7) 放弃本次修改")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"1", "2", "3", "4", "5", "6", "7", "q"}, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return Page.CONFIG
    if state.edit_cfg is None:
        return _show_info(state, "配置结果", ["未加载配置，请重新进入修改页面。"], Page.CONFIG, "主菜单->配置->配置结果")

    if key in {"1", "2", "3", "4", "5"}:
        mapping = {"1": "asr", "2": "pipeline", "3": "translation", "4": "output", "5": "schema"}
        state.edit_section = mapping[key]
        state.edit_offset = 0
        return Page.CONFIG_EDIT_FIELDS

    if key == "6":
        confirm = _confirm_action(f"确认写入配置文件 {state.config_path}？")
        if confirm == "q":
            return Page.MAIN
        if confirm != "y":
            return _show_info(state, "配置结果", ["已取消保存。"], Page.CONFIG_EDIT_MENU, "主菜单->配置->修改配置项->配置结果")
        try:
            save_config(state.config_path, state.edit_cfg)
            return _show_info(state, "配置结果", [f"已保存到：{state.config_path}"], Page.CONFIG_EDIT_MENU, "主菜单->配置->修改配置项->配置结果")
        except Exception as e:
            return _show_info(state, "配置结果", [f"保存失败：{e}"], Page.CONFIG_EDIT_MENU, "主菜单->配置->修改配置项->配置结果")

    if key == "7":
        confirm = _confirm_action("确认放弃本次未保存修改？")
        if confirm == "q":
            return Page.MAIN
        if confirm != "y":
            return Page.CONFIG_EDIT_MENU
        state.edit_cfg = None
        state.edit_section = "asr"
        state.edit_offset = 0
        return _show_info(state, "配置结果", ["已放弃本次修改。"], Page.CONFIG, "主菜单->配置->配置结果")

    return Page.CONFIG_EDIT_MENU


def _render_config_edit_fields_page(state: SessionState) -> Page:
    if state.edit_cfg is None:
        return _show_info(state, "配置结果", ["未加载配置，请重新进入修改页面。"], Page.CONFIG, "主菜单->配置->配置结果")

    section = state.edit_section
    section_obj = _get_section_obj(state.edit_cfg, section)
    items: list[tuple[str, Any]] = list(section_obj.__dict__.items())
    page_size = 7
    offset = state.edit_offset
    visible = items[offset : offset + page_size]
    has_prev = offset > 0
    has_next = offset + page_size < len(items)

    _print_path_bar(f"主菜单->配置->修改配置项->{SECTION_TITLES.get(section, section)}")
    print(f"{SECTION_TITLES.get(section, section)} 字段编辑")
    print("==============")
    for idx, (name, value) in enumerate(visible, start=1):
        print(f"{idx}) {name} = {_format_value_for_menu(value)}")
    if has_prev:
        print("8) 上一页")
    if has_next:
        print("9) 下一页")
    print()
    print("q 主菜单，Esc 返回")

    valid_keys = {str(i) for i in range(1, len(visible) + 1)} | {"q"}
    if has_prev:
        valid_keys.add("8")
    if has_next:
        valid_keys.add("9")
    key = _read_single_key(valid_keys, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return Page.CONFIG_EDIT_MENU
    if key == "8" and has_prev:
        state.edit_offset = max(0, offset - page_size)
        return Page.CONFIG_EDIT_FIELDS
    if key == "9" and has_next:
        state.edit_offset = offset + page_size
        return Page.CONFIG_EDIT_FIELDS

    if key.isdigit():
        selected = int(key)
        if 1 <= selected <= len(visible):
            field_name, current = visible[selected - 1]
            value_res = _read_line_with_cancel(f"新值 {field_name}", str(current))
            if value_res.kind == "main":
                return Page.MAIN
            if value_res.kind != "value":
                return _show_info(
                    state,
                    "配置结果",
                    [f"已取消修改 {field_name}。"],
                    Page.CONFIG_EDIT_FIELDS,
                    f"主菜单->配置->修改配置项->{SECTION_TITLES.get(section, section)}->配置结果",
                )
            try:
                new_value = _parse_field_value(section, field_name, current, value_res.value)
                setattr(section_obj, field_name, new_value)
                return _show_info(
                    state,
                    "配置结果",
                    [f"已更新 {field_name} = {_format_value_for_menu(new_value)}", "提示：当前仅在内存，需在“保存并写入”后落盘。"],
                    Page.CONFIG_EDIT_FIELDS,
                    f"主菜单->配置->修改配置项->{SECTION_TITLES.get(section, section)}->配置结果",
                )
            except Exception as e:
                return _show_info(
                    state,
                    "配置结果",
                    [f"更新失败：{e}"],
                    Page.CONFIG_EDIT_FIELDS,
                    f"主菜单->配置->修改配置项->{SECTION_TITLES.get(section, section)}->配置结果",
                )
    return Page.CONFIG_EDIT_FIELDS


def _render_help_page() -> Page:
    _print_path_bar("主菜单->帮助")
    print("帮助页面")
    print("========")
    print("- 数字键：即时执行当前页面选项")
    print("- q：返回主菜单")
    print("- Esc：全局返回（在主菜单中 Esc 退出）")
    print("- 关键动作会要求 y/n 二次确认")
    print("- 信息结果以单独页面展示，Esc 关闭")
    print("- 提示词和翻译偏好文件路径可在配置中设置")
    print("- 在 prompt.txt 中可使用 {perf} 插入偏好字典")
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
        if page == Page.CONFIG_EDIT_MENU:
            page = _render_config_edit_menu_page(state)
            continue
        if page == Page.CONFIG_EDIT_FIELDS:
            page = _render_config_edit_fields_page(state)
            continue
        if page == Page.HELP:
            page = _render_help_page()
            continue
        if page == Page.INFO:
            page = _render_info_page(state)
            continue

    return 0
