from __future__ import annotations

import os
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from whisperlrc.config import AppConfig, load_config, save_config
from whisperlrc.logging import setup_logging


class Page(Enum):
    MAIN = auto()
    BATCH = auto()
    PROCESSING = auto()
    CONFIG = auto()
    CHECK = auto()
    HELP = auto()
    INFO = auto()
    API_TEST = auto()
    CONFIG_EDIT_MENU = auto()
    CONFIG_EDIT_FIELDS = auto()
    CONFIG_EDIT_LLM = auto()


@dataclass
class SessionState:
    config_path: Path = Path("settings.toml")
    input_dir: Path = Path()
    output_dir: Path = Path()
    info_title: str = ""
    info_lines: list[str] = field(default_factory=list)
    info_back: Page = Page.MAIN
    info_path: str = "主菜单"
    edit_cfg: AppConfig | None = None
    edit_section: str = "asr"
    edit_offset: int = 0
    edit_translation_llm_only: bool = False
    api_test_config: Path = Path()
    api_test_running: bool = False
    api_test_result_lines: list[str] = field(default_factory=list)
    api_test_queue: queue.Queue[tuple[bool, list[str]]] | None = None
    batch_running: bool = False
    batch_finished: bool = False
    batch_requested_cancel: bool = False
    batch_result_rc: int = 0
    batch_error: str = ""
    batch_summary_lines: list[str] = field(default_factory=list)
    batch_task_queue: queue.Queue[dict[str, Any]] | None = None
    batch_worker: threading.Thread | None = None
    batch_cancel_token: Any | None = None
    batch_total_files: int = 0
    batch_current_file: str = ""
    batch_current_file_index: int = 0
    batch_group_index: int = 0
    batch_group_total: int = 0
    batch_last_request_json: str = ""
    batch_last_response_json: str = ""
    batch_running_path: str = ""


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
    print("!!! 关键操作确认 !!!")
    print(prompt)
    print("按 y 继续执行，按 n 取消，q 返回主菜单，Esc 返回（默认 n）")
    key = _read_single_key({"y", "n", "q"}, allow_esc=True)
    return key or "n"


def _read_key_nonblocking() -> str:
    if os.name != "nt":
        return ""
    try:
        import msvcrt

        if not msvcrt.kbhit():
            return ""
        ch = msvcrt.getwch()
        if not ch:
            return ""
        if ch in {"\x00", "\xe0"}:
            msvcrt.getwch()
            return ""
        if ch == "\x1b":
            return "esc"
        return ch.lower()
    except Exception:
        return ""


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


def _start_batch_task(
    state: SessionState,
    *,
    input_dir: Path,
    output_dir: Path,
    config: Path,
) -> None:
    from whisperlrc.pipeline import CancelToken, process_batch

    task_q: queue.Queue[dict[str, Any]] = queue.Queue()
    cancel_token = CancelToken()
    state.batch_task_queue = task_q
    state.batch_cancel_token = cancel_token
    state.batch_running = True
    state.batch_finished = False
    state.batch_requested_cancel = False
    state.batch_result_rc = 0
    state.batch_error = ""
    state.batch_summary_lines = []
    state.batch_total_files = 0
    state.batch_current_file = ""
    state.batch_current_file_index = 0
    state.batch_group_index = 0
    state.batch_group_total = 0
    state.batch_last_request_json = ""
    state.batch_last_response_json = ""
    state.batch_running_path = f"输入={input_dir} | 输出={output_dir} | 配置={config}"

    def emit(event: dict[str, Any]) -> None:
        task_q.put(event)

    def worker() -> None:
        try:
            setup_logging("INFO")
            cfg = load_config(config)
            if not input_dir.exists() or not input_dir.is_dir():
                raise ValueError(f"输入目录不存在：{input_dir}")
            rc = process_batch(
                input_dir,
                output_dir,
                cfg,
                event_cb=emit,
                cancel_token=cancel_token,
            )
            task_q.put({"type": "worker_done", "rc": rc})
        except Exception as e:
            task_q.put({"type": "worker_error", "error": str(e)})

    t = threading.Thread(target=worker, daemon=True)
    state.batch_worker = t
    t.start()


def _consume_batch_events(state: SessionState) -> None:
    if state.batch_task_queue is None:
        return
    while True:
        try:
            event = state.batch_task_queue.get_nowait()
        except queue.Empty:
            return

        etype = str(event.get("type", "")).strip()
        if etype == "batch_start":
            state.batch_total_files = int(event.get("total_files", 0))
            continue
        if etype == "file_start":
            state.batch_current_file = str(event.get("file", ""))
            state.batch_current_file_index = int(event.get("file_index", 0))
            state.batch_group_index = 0
            state.batch_group_total = 0
            continue
        if etype == "translation_group_start":
            state.batch_group_index = int(event.get("group_index", 0))
            state.batch_group_total = int(event.get("total_groups", 0))
            continue
        if etype == "llm_request":
            state.batch_last_request_json = str(event.get("json", ""))
            continue
        if etype == "llm_response":
            state.batch_last_response_json = str(event.get("json", ""))
            continue
        if etype == "worker_done":
            state.batch_running = False
            state.batch_finished = True
            state.batch_result_rc = int(event.get("rc", 2))
            status = "已取消" if state.batch_requested_cancel else "已完成"
            state.batch_summary_lines = [
                f"状态：{status}",
                f"退出码：{state.batch_result_rc}",
                f"总文件数：{state.batch_total_files}",
                f"最后处理文件：{state.batch_current_file or '（无）'}",
            ]
            continue
        if etype == "worker_error":
            state.batch_running = False
            state.batch_finished = True
            state.batch_result_rc = 2
            state.batch_error = str(event.get("error", "未知错误"))
            state.batch_summary_lines = [
                "状态：失败",
                f"错误：{state.batch_error}",
                f"总文件数：{state.batch_total_files}",
                f"最后处理文件：{state.batch_current_file or '（无）'}",
            ]
            continue


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
        f"默认输入目录：{cfg.output.default_input_dir}",
        f"默认输出目录：{cfg.output.default_output_dir}",
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


def _build_api_hello_check(config: Path) -> list[str]:
    from whisperlrc.translate.factory import build_translator

    cfg = load_config(config)
    lines = [
        f"配置文件：{config}",
        f"翻译后端：{cfg.translation.backend}",
        "测试请求：hello",
    ]
    translator = build_translator(cfg.translation)
    try:
        if hasattr(translator, "test_api_hello"):
            reply = str(getattr(translator, "test_api_hello")()).strip()
        else:
            reply_list = translator.translate_batch(["hello"], src="en", tgt=cfg.translation.target, retry=0)
            reply = reply_list[0].strip() if reply_list else ""
        if not reply:
            raise RuntimeError("API 回复为空")
        lines.append(f"结果：连通成功，回复：{reply}")
    except Exception as e:
        lines.append(f"结果：API 测试失败：{e}")
    return lines


def _start_api_test(state: SessionState, config: Path) -> None:
    q: queue.Queue[tuple[bool, list[str]]] = queue.Queue()
    state.api_test_queue = q
    state.api_test_config = config
    state.api_test_running = True
    state.api_test_result_lines = []

    def _worker() -> None:
        try:
            lines = _build_api_hello_check(config)
            q.put((True, lines))
        except Exception as e:
            q.put((False, [f"测试失败：{e}"]))

    threading.Thread(target=_worker, daemon=True).start()


def _consume_api_test_result(state: SessionState) -> None:
    if state.api_test_queue is None:
        return
    try:
        _ok, lines = state.api_test_queue.get_nowait()
    except queue.Empty:
        return
    state.api_test_running = False
    state.api_test_result_lines = lines


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
    _consume_batch_events(state)
    if state.batch_running:
        return Page.PROCESSING

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
        _start_batch_task(
            state,
            input_dir=state.input_dir,
            output_dir=state.output_dir,
            config=state.config_path,
        )
        return Page.PROCESSING

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
        _start_batch_task(
            state,
            input_dir=Path(input_res.value),
            output_dir=Path(output_res.value),
            config=Path(config_res.value),
        )
        return Page.PROCESSING

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
    print("2) 修改配置项")
    print("3) 切换会话配置文件路径")
    print("4) 重置会话配置为 settings.toml")
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
        try:
            state.edit_cfg = load_config(state.config_path)
            state.edit_section = "asr"
            state.edit_offset = 0
            state.edit_translation_llm_only = False
            return Page.CONFIG_EDIT_MENU
        except Exception as e:
            return _show_info(state, "配置结果", [f"加载配置失败：{e}"], Page.CONFIG, "主菜单->配置->配置结果")

    if key == "3":
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "配置结果", ["已取消修改配置路径。"], Page.CONFIG, "主菜单->配置->配置结果")
        state.config_path = Path(config_res.value)
        try:
            cfg = load_config(state.config_path)
            state.input_dir = Path(cfg.output.default_input_dir)
            state.output_dir = Path(cfg.output.default_output_dir)
            return _show_info(
                state,
                "配置结果",
                [
                    f"已更新会话配置：{state.config_path}",
                    f"默认输入目录：{state.input_dir}",
                    f"默认输出目录：{state.output_dir}",
                ],
                Page.CONFIG,
                "主菜单->配置->配置结果",
            )
        except Exception as e:
            return _show_info(state, "配置结果", [f"已更新会话配置：{state.config_path}", f"加载默认路径失败：{e}"], Page.CONFIG, "主菜单->配置->配置结果")

    if key == "4":
        state.config_path = Path("settings.toml")
        try:
            cfg = load_config(state.config_path)
            state.input_dir = Path(cfg.output.default_input_dir)
            state.output_dir = Path(cfg.output.default_output_dir)
            return _show_info(
                state,
                "配置结果",
                [
                    "已重置会话配置为 settings.toml",
                    f"默认输入目录：{state.input_dir}",
                    f"默认输出目录：{state.output_dir}",
                ],
                Page.CONFIG,
                "主菜单->配置->配置结果",
            )
        except Exception as e:
            return _show_info(state, "配置结果", ["已重置会话配置为 settings.toml", f"加载默认路径失败：{e}"], Page.CONFIG, "主菜单->配置->配置结果")
    return Page.CONFIG


def _render_check_page(state: SessionState) -> Page:
    _print_path_bar("主菜单->检查")
    print("检查页面")
    print("========")
    print("1) 检查当前会话配置的翻译后端")
    print("2) API 测试")
    print("3) 使用自定义配置文件检查")
    print("4) 使用自定义配置做 API 测试")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"1", "2", "3", "4", "q"}, allow_esc=True)
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
        _start_api_test(state, state.config_path)
        return Page.API_TEST

    if key == "3":
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "翻译后端检查", ["已取消本次检查。"], Page.CHECK, "主菜单->检查->翻译后端检查")
        try:
            return _show_info(state, "翻译后端检查", _build_translation_check(Path(config_res.value)), Page.CHECK, "主菜单->检查->翻译后端检查")
        except Exception as e:
            return _show_info(state, "翻译后端检查", [f"检查失败：{e}"], Page.CHECK, "主菜单->检查->翻译后端检查")

    if key == "4":
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path))
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "API 测试", ["已取消本次测试。"], Page.CHECK, "主菜单->检查->API测试")
        _start_api_test(state, Path(config_res.value))
        return Page.API_TEST

    return Page.CHECK


def _render_config_edit_menu_page(state: SessionState) -> Page:
    _print_path_bar("主菜单->配置->修改配置项")
    print("配置分区")
    print("========")
    print("1) ASR")
    print("2) 管线")
    print("3) 翻译基础")
    print("4) LLM 配置")
    print("5) 输出")
    print("6) Schema")
    print("7) 保存并写入当前配置文件")
    print("8) 放弃本次修改")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({"1", "2", "3", "4", "5", "6", "7", "8", "q"}, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return Page.CONFIG
    if state.edit_cfg is None:
        return _show_info(state, "配置结果", ["未加载配置，请重新进入修改页面。"], Page.CONFIG, "主菜单->配置->配置结果")

    if key in {"1", "2", "3", "5", "6"}:
        mapping = {"1": "asr", "2": "pipeline", "3": "translation", "5": "output", "6": "schema"}
        state.edit_section = mapping[key]
        state.edit_offset = 0
        state.edit_translation_llm_only = False
        return Page.CONFIG_EDIT_FIELDS

    if key == "4":
        state.edit_section = "translation"
        state.edit_offset = 0
        state.edit_translation_llm_only = True
        return Page.CONFIG_EDIT_LLM

    if key == "7":
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

    if key == "8":
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
    all_items: list[tuple[str, Any]] = list(section_obj.__dict__.items())
    if section == "translation" and not state.edit_translation_llm_only:
        items = [(k, v) for k, v in all_items if (not k.startswith("llm_")) and k != "timeout_sec"]
    else:
        items = all_items
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
                if section == "output" and field_name == "default_input_dir":
                    state.input_dir = Path(str(new_value))
                if section == "output" and field_name == "default_output_dir":
                    state.output_dir = Path(str(new_value))
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


def _render_config_edit_llm_page(state: SessionState) -> Page:
    if state.edit_cfg is None:
        return _show_info(state, "配置结果", ["未加载配置，请重新进入修改页面。"], Page.CONFIG, "主菜单->配置->配置结果")

    section_obj = _get_section_obj(state.edit_cfg, "translation")
    llm_keys = [
        "llm_provider",
        "llm_model",
        "llm_base_url",
        "llm_api_key",
        "llm_prompt_file",
        "llm_preferences_file",
        "llm_batch_size",
        "llm_context_window",
        "timeout_sec",
    ]
    items: list[tuple[str, Any]] = [(k, getattr(section_obj, k)) for k in llm_keys]

    _print_path_bar("主菜单->配置->修改配置项->LLM配置")
    print("LLM 配置")
    print("==============")
    for idx, (name, value) in enumerate(items, start=1):
        print(f"{idx}) {name} = {_format_value_for_menu(value)}")
    print()
    print("q 主菜单，Esc 返回")

    key = _read_single_key({str(i) for i in range(1, len(items) + 1)} | {"q"}, allow_esc=True)
    if key == "q":
        return Page.MAIN
    if key == "esc":
        return Page.CONFIG_EDIT_MENU

    if key.isdigit():
        selected = int(key)
        if 1 <= selected <= len(items):
            field_name, current = items[selected - 1]
            value_res = _read_line_with_cancel(f"新值 {field_name}", str(current))
            if value_res.kind == "main":
                return Page.MAIN
            if value_res.kind != "value":
                return _show_info(
                    state,
                    "配置结果",
                    [f"已取消修改 {field_name}。"],
                    Page.CONFIG_EDIT_LLM,
                    "主菜单->配置->修改配置项->LLM配置->配置结果",
                )
            try:
                new_value = _parse_field_value("translation", field_name, current, value_res.value)
                setattr(section_obj, field_name, new_value)
                return _show_info(
                    state,
                    "配置结果",
                    [f"已更新 {field_name} = {_format_value_for_menu(new_value)}", "提示：当前仅在内存，需在“保存并写入”后落盘。"],
                    Page.CONFIG_EDIT_LLM,
                    "主菜单->配置->修改配置项->LLM配置->配置结果",
                )
            except Exception as e:
                return _show_info(
                    state,
                    "配置结果",
                    [f"更新失败：{e}"],
                    Page.CONFIG_EDIT_LLM,
                    "主菜单->配置->修改配置项->LLM配置->配置结果",
                )
    return Page.CONFIG_EDIT_LLM


def _render_processing_page(state: SessionState) -> Page:
    while True:
        _clear_screen()
        _print_path_bar("主菜单->批处理->处理中")
        _consume_batch_events(state)

        print("处理页面")
        print("========")
        print(state.batch_running_path or "（未设置）")
        print()
        if state.batch_running:
            status = "运行中"
            if state.batch_requested_cancel:
                status = "已请求取消，等待当前步骤结束"
            print(f"状态：{status}")
        else:
            print("状态：已结束")

        total = state.batch_total_files
        current_idx = state.batch_current_file_index
        if total > 0:
            print(f"文件进度：{current_idx}/{total}")
        else:
            print("文件进度：0/0")
        print(f"当前文件：{state.batch_current_file or '（无）'}")

        if state.batch_group_total > 0:
            print(f"翻译分组：{state.batch_group_index}/{state.batch_group_total}")
        else:
            print("翻译分组：0/0")
        print()

        print("最近一次请求 JSON：")
        print("------------------")
        print(state.batch_last_request_json or "（暂无）")
        print()
        print("最近一次响应 JSON：")
        print("------------------")
        print(state.batch_last_response_json or "（暂无）")
        print()

        if state.batch_running:
            print("Esc 取消并返回")
            key = _read_key_nonblocking()
            if key == "esc":
                if not state.batch_requested_cancel and state.batch_cancel_token is not None:
                    state.batch_requested_cancel = True
                    try:
                        cancel = getattr(state.batch_cancel_token, "cancel", None)
                        if callable(cancel):
                            cancel()
                    except Exception:
                        pass
                return Page.BATCH
            if key == "q":
                return Page.MAIN
            time.sleep(0.08)
            continue

        lines = state.batch_summary_lines or [
            f"状态：{'已取消' if state.batch_requested_cancel else '已完成'}",
            f"退出码：{state.batch_result_rc}",
        ]
        state.batch_finished = False
        return _show_info(state, "执行结果", lines, Page.BATCH, "主菜单->批处理->执行结果")


def _render_api_test_page(state: SessionState) -> Page:
    while True:
        _clear_screen()
        _print_path_bar("主菜单->检查->API测试")
        _consume_api_test_result(state)
        if state.api_test_running:
            print("API 测试")
            print("========")
            print(f"配置文件：{state.api_test_config}")
            print("正在测试，请稍候...")
            print("Esc 返回上级，q 主菜单")
            key = _read_key_nonblocking()
            if key == "esc":
                return Page.CHECK
            if key == "q":
                return Page.MAIN
            time.sleep(0.08)
            continue

        print("API 测试")
        print("========")
        for line in state.api_test_result_lines or ["测试已结束。"]:
            print(line)
        print()
        print("Esc 返回上级，q 主菜单")
        key = _read_single_key({"q"}, allow_esc=True)
        if key == "q":
            return Page.MAIN
        if key == "esc":
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
    print("- 提示词和翻译偏好文件路径可在配置中设置")
    print("- 在 prompt.txt 中可使用 {perf} 插入偏好字典")
    print("- 检查页面支持 API 测试")
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
    try:
        cfg = load_config(state.config_path)
        state.input_dir = Path(cfg.output.default_input_dir)
        state.output_dir = Path(cfg.output.default_output_dir)
    except Exception:
        state.input_dir = Path("input_audio")
        state.output_dir = Path("output")
    page: Page | None = Page.MAIN

    while page is not None:
        _clear_screen()
        if page == Page.MAIN:
            page = _render_main_page()
            continue
        if page == Page.BATCH:
            page = _render_batch_page(state)
            continue
        if page == Page.PROCESSING:
            page = _render_processing_page(state)
            continue
        if page == Page.CONFIG:
            page = _render_config_page(state)
            continue
        if page == Page.CHECK:
            page = _render_check_page(state)
            continue
        if page == Page.API_TEST:
            page = _render_api_test_page(state)
            continue
        if page == Page.CONFIG_EDIT_MENU:
            page = _render_config_edit_menu_page(state)
            continue
        if page == Page.CONFIG_EDIT_FIELDS:
            page = _render_config_edit_fields_page(state)
            continue
        if page == Page.CONFIG_EDIT_LLM:
            page = _render_config_edit_llm_page(state)
            continue
        if page == Page.HELP:
            page = _render_help_page()
            continue
        if page == Page.INFO:
            page = _render_info_page(state)
            continue

    return 0
