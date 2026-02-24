from __future__ import annotations

import os
import json
import queue
import signal
import sys
import threading
import time
import traceback
import faulthandler
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, TextIO

from whisperlrc.config import AppConfig, load_config, save_config
from whisperlrc.logging import setup_logging


class Page(Enum):
    MAIN = auto()
    BATCH = auto()
    PROCESSING = auto()
    CONFIG = auto()
    CHECK = auto()
    REVIEW_WEB = auto()
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
    edit_dirty: bool = False
    edit_menu_cursor: int = 0
    edit_field_cursor: int = 0
    edit_llm_cursor: int = 0
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
    batch_logs: list[str] = field(default_factory=list)
    batch_log_cursor: int = 0
    batch_processing_header_printed: bool = False
    batch_output_dir: Path = Path()
    batch_current_log_path: Path | None = None
    batch_current_log_fp: TextIO | None = None
    batch_current_prompt_tokens: int = 0
    batch_current_completion_tokens: int = 0
    batch_current_total_tokens: int = 0
    batch_current_prompt_cache_hit_tokens: int = 0
    batch_current_prompt_cache_miss_tokens: int = 0
    batch_current_reasoning_tokens: int = 0
    batch_current_cached_tokens: int = 0
    batch_current_usage_missing_count: int = 0
    batch_last_usage_has_usage: bool | None = None
    batch_current_input_chars: int = 0
    batch_current_output_chars: int = 0
    batch_current_tool_calls: int = 0
    batch_last_file_stats_lines: list[str] = field(default_factory=list)
    op_log_path: Path | None = None
    op_log_fp: TextIO | None = None
    crash_log_path: Path | None = None
    crash_log_fp: TextIO | None = None


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


def _read_nav_key() -> str:
    if os.name == "nt":
        try:
            import msvcrt

            while True:
                ch = msvcrt.getwch()
                if not ch:
                    continue
                if ch in {"\x00", "\xe0"}:
                    code = msvcrt.getwch()
                    mapping = {
                        "H": "up",
                        "P": "down",
                        "K": "left",
                        "M": "right",
                    }
                    key = mapping.get(code.upper(), "")
                    if key:
                        return key
                    continue
                if ch in {"\r", "\n"}:
                    return "enter"
                if ch == "\x1b":
                    return "esc"
                if ch.lower() == "q":
                    return "q"
        except Exception:
            pass

    text = input().strip().lower()
    fallback_map = {
        "w": "up",
        "s": "down",
        "a": "left",
        "d": "right",
        "": "enter",
        "q": "q",
        "esc": "esc",
    }
    return fallback_map.get(text, "")


def _read_line_with_cancel(
    label: str,
    default: str | None = None,
    *,
    empty_as_default: bool = False,
) -> LineInputResult:
    suffix = f" (当前: {default})" if default else ""
    enter_hint = "回车保留默认" if empty_as_default and default is not None else "回车取消"
    prompt = f"{label}{suffix}（{enter_hint}，q 主菜单，Esc 返回）："

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
                        if empty_as_default and default is not None:
                            return LineInputResult("value", default)
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
        if empty_as_default and default is not None:
            return LineInputResult("value", default)
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


def _run_arrow_menu(
    *,
    path: str,
    title: str,
    options: list[str],
    cursor: int = 0,
    subtitle_lines: list[str] | None = None,
) -> tuple[str, int]:
    if not options:
        return ("esc", 0)
    idx = min(max(cursor, 0), len(options) - 1)
    while True:
        _clear_screen()
        _print_path_bar(path)
        print(title)
        print("=" * len(title))
        if subtitle_lines:
            for line in subtitle_lines:
                print(line)
            print()
        for i, text in enumerate(options):
            marker = ">" if i == idx else " "
            print(f"{marker} {text}")

        key = _read_nav_key()
        if key == "up":
            idx = (idx - 1) % len(options)
            continue
        if key == "down":
            idx = (idx + 1) % len(options)
            continue
        if key == "enter":
            return ("enter", idx)
        if key == "q":
            return ("q", idx)
        if key == "esc":
            return ("esc", idx)


def _safe_log_stem(file_name: str) -> str:
    stem = Path(file_name).stem.strip()
    return stem or "unnamed"


def _ensure_unique_session_log_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"session_{ts}.log"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = output_dir / f"session_{ts}_{i}.log"
        if not candidate.exists():
            return candidate
        i += 1


def _open_operation_log_file(state: SessionState) -> None:
    if state.op_log_fp is not None:
        return
    try:
        output_dir = state.output_dir if str(state.output_dir).strip() else Path("output")
        log_path = _ensure_unique_session_log_path(output_dir)
        state.op_log_fp = log_path.open("w", encoding="utf-8")
        state.op_log_path = log_path
    except Exception:
        state.op_log_fp = None
        state.op_log_path = None


def _close_operation_log_file(state: SessionState) -> None:
    if state.op_log_fp is not None:
        try:
            state.op_log_fp.flush()
            state.op_log_fp.close()
        except Exception:
            pass
    state.op_log_fp = None
    state.op_log_path = None


def _append_operation_log(state: SessionState, event: str, payload: dict[str, Any] | None = None) -> None:
    fp = state.op_log_fp
    if fp is None:
        return
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "payload": payload or {},
    }
    try:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fp.flush()
    except Exception:
        pass


_CRASH_HOOKS_INSTALLED = False
_CRASH_LOG_FP: TextIO | None = None
_ORIG_SYS_EXCEPTHOOK = sys.excepthook
_ORIG_THREAD_EXCEPTHOOK = getattr(threading, "excepthook", None)


def _build_crash_log_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"crash_{ts}.log"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = output_dir / f"crash_{ts}_{i}.log"
        if not candidate.exists():
            return candidate
        i += 1


def _append_crash_log_line(text: str) -> None:
    fp = _CRASH_LOG_FP
    if fp is None:
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        fp.write(f"{ts} | {text}\n")
        fp.flush()
    except Exception:
        pass


def _install_crash_logging(state: SessionState) -> None:
    global _CRASH_HOOKS_INSTALLED, _CRASH_LOG_FP
    if _CRASH_HOOKS_INSTALLED:
        return
    try:
        output_dir = state.output_dir if str(state.output_dir).strip() else Path("output")
        crash_path = _build_crash_log_path(output_dir)
        crash_fp = crash_path.open("a", encoding="utf-8")
    except Exception:
        return

    state.crash_log_path = crash_path
    state.crash_log_fp = crash_fp
    _CRASH_LOG_FP = crash_fp
    _append_crash_log_line("crash logger initialized")

    def _sys_excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
        _append_crash_log_line("UNCAUGHT EXCEPTION")
        _append_crash_log_line("".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip())
        try:
            _ORIG_SYS_EXCEPTHOOK(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    def _thread_excepthook(args: Any) -> None:
        _append_crash_log_line(f"UNCAUGHT THREAD EXCEPTION in {getattr(args, 'thread', None)}")
        _append_crash_log_line(
            "".join(
                traceback.format_exception(
                    getattr(args, "exc_type", Exception),
                    getattr(args, "exc_value", Exception("unknown thread exception")),
                    getattr(args, "exc_traceback", None),
                )
            ).rstrip()
        )
        try:
            if _ORIG_THREAD_EXCEPTHOOK is not None:
                _ORIG_THREAD_EXCEPTHOOK(args)
        except Exception:
            pass

    try:
        sys.excepthook = _sys_excepthook
    except Exception:
        pass
    try:
        if hasattr(threading, "excepthook"):
            threading.excepthook = _thread_excepthook  # type: ignore[assignment]
    except Exception:
        pass

    try:
        faulthandler.enable(file=crash_fp, all_threads=True)
    except Exception:
        pass
    try:
        sig = getattr(signal, "SIGTERM", None)
        if sig is not None:
            faulthandler.register(sig, file=crash_fp, all_threads=True, chain=True)
    except Exception:
        pass

    _CRASH_HOOKS_INSTALLED = True


def _close_crash_log_file(state: SessionState) -> None:
    global _CRASH_LOG_FP
    fp = state.crash_log_fp
    if fp is None:
        return
    try:
        _append_crash_log_line("crash logger closed")
        fp.flush()
        fp.close()
    except Exception:
        pass
    state.crash_log_fp = None
    _CRASH_LOG_FP = None


def _state_snapshot(state: SessionState) -> dict[str, Any]:
    return {
        "config_path": str(state.config_path),
        "input_dir": str(state.input_dir),
        "output_dir": str(state.output_dir),
        "batch_running": state.batch_running,
        "batch_finished": state.batch_finished,
        "batch_requested_cancel": state.batch_requested_cancel,
        "batch_total_files": state.batch_total_files,
        "batch_current_file": state.batch_current_file,
        "batch_current_file_index": state.batch_current_file_index,
        "batch_group_index": state.batch_group_index,
        "batch_group_total": state.batch_group_total,
        "batch_result_rc": state.batch_result_rc,
    }


def _build_log_path(output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}.log"


def _close_current_log_file(state: SessionState) -> None:
    if state.batch_current_log_fp is not None:
        try:
            state.batch_current_log_fp.flush()
            state.batch_current_log_fp.close()
        except Exception:
            pass
    state.batch_current_log_fp = None
    state.batch_current_log_path = None


def _open_current_log_file(state: SessionState, file_name: str) -> None:
    _close_current_log_file(state)
    try:
        output_dir = state.batch_output_dir or state.output_dir
        log_path = _build_log_path(output_dir, _safe_log_stem(file_name))
        fp = log_path.open("w", encoding="utf-8")
        state.batch_current_log_path = log_path
        state.batch_current_log_fp = fp
    except Exception as e:
        state.batch_current_log_path = None
        state.batch_current_log_fp = None
        state.batch_logs.append(f"[日志错误] 无法创建日志文件：{e}")


def _append_batch_log(state: SessionState, line: str) -> None:
    state.batch_logs.append(line)
    fp = state.batch_current_log_fp
    if fp is None:
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        fp.write(f"{ts} | {line}\n")
        fp.flush()
    except Exception as e:
        state.batch_logs.append(f"[日志错误] 写入日志失败：{e}")
        _close_current_log_file(state)


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
    state.batch_output_dir = output_dir
    state.batch_current_prompt_tokens = 0
    state.batch_current_completion_tokens = 0
    state.batch_current_total_tokens = 0
    state.batch_current_prompt_cache_hit_tokens = 0
    state.batch_current_prompt_cache_miss_tokens = 0
    state.batch_current_reasoning_tokens = 0
    state.batch_current_cached_tokens = 0
    state.batch_current_usage_missing_count = 0
    state.batch_last_usage_has_usage = None
    state.batch_current_input_chars = 0
    state.batch_current_output_chars = 0
    state.batch_current_tool_calls = 0
    state.batch_last_file_stats_lines = []
    _close_current_log_file(state)
    state.batch_running_path = f"输入={input_dir} | 输出={output_dir} | 配置={config}"
    state.batch_logs = [f"[启动] {state.batch_running_path}"]
    state.batch_log_cursor = 0
    state.batch_processing_header_printed = False
    _append_operation_log(
        state,
        "batch_start_requested",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "config": str(config),
            "state": _state_snapshot(state),
        },
    )

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
        except BaseException as e:
            task_q.put(
                {
                    "type": "worker_error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

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
        _append_operation_log(
            state,
            "batch_event",
            {
                "type": etype,
                "event": event,
                "state_before": _state_snapshot(state),
            },
        )
        if etype == "batch_start":
            state.batch_total_files = int(event.get("total_files", 0))
            _append_batch_log(state, f"[批处理] 总文件数={state.batch_total_files}")
            continue
        if etype == "file_start":
            state.batch_current_file = str(event.get("file", ""))
            state.batch_current_file_index = int(event.get("file_index", 0))
            state.batch_current_prompt_tokens = 0
            state.batch_current_completion_tokens = 0
            state.batch_current_total_tokens = 0
            state.batch_current_prompt_cache_hit_tokens = 0
            state.batch_current_prompt_cache_miss_tokens = 0
            state.batch_current_reasoning_tokens = 0
            state.batch_current_cached_tokens = 0
            state.batch_current_usage_missing_count = 0
            state.batch_last_usage_has_usage = None
            state.batch_current_input_chars = 0
            state.batch_current_output_chars = 0
            state.batch_current_tool_calls = 0
            _open_current_log_file(state, state.batch_current_file)
            _append_batch_log(
                state,
                f"[文件开始] {state.batch_current_file_index}/{int(event.get('total_files', 0))} {state.batch_current_file}",
            )
            if state.batch_current_log_path is not None:
                _append_batch_log(state, f"[日志文件] {state.batch_current_log_path}")
            continue
        if etype == "asr_output":
            _append_batch_log(state, "[ASR输出]")
            _append_batch_log(state, str(event.get("text", "")).strip() or "（空）")
            continue
        if etype == "json_cache_resume_loaded":
            _append_batch_log(state, "[缓存恢复]")
            _append_batch_log(
                state,
                f"已加载 {event.get('path', '')} | 已完成句子 {event.get('completed_sentences', 0)}/{event.get('total_sentences', 0)}",
            )
            continue
        if etype == "json_cache_resume_mismatch":
            _append_batch_log(state, "[缓存恢复]")
            _append_batch_log(state, f"跳过缓存：{event.get('reason', '未知原因')}")
            continue
        if etype == "json_cache_init_written":
            _append_batch_log(state, "[JSON缓存]")
            _append_batch_log(state, f"ASR首写：{event.get('path', '')}")
            continue
        if etype == "json_cache_group_written":
            _append_batch_log(state, "[JSON缓存]")
            _append_batch_log(
                state,
                f"分组回写：{event.get('path', '')} | 进度 {event.get('translated_sentences', 0)}/{event.get('total_sentences', 0)}",
            )
            continue
        if etype == "json_cache_final_written":
            _append_batch_log(state, "[JSON缓存]")
            _append_batch_log(
                state,
                f"最终写入：{event.get('path', '')} | 状态={event.get('status', '')}",
            )
            continue
        if etype == "translation_group_start":
            state.batch_group_index = int(event.get("group_index", 0))
            state.batch_group_total = int(event.get("total_groups", 0))
            _append_batch_log(state, f"[翻译分组] {state.batch_group_index}/{state.batch_group_total} 开始")
            continue
        if etype == "llm_request":
            state.batch_last_request_json = str(event.get("json", ""))
            _append_batch_log(state, "[LLM请求]")
            _append_batch_log(state, state.batch_last_request_json or "（空）")
            continue
        if etype == "llm_response":
            state.batch_last_response_json = str(event.get("json", ""))
            _append_batch_log(state, "[LLM响应]")
            _append_batch_log(state, state.batch_last_response_json or "（空）")
            continue
        if etype == "llm_usage_tokens":
            has_usage = bool(event.get("has_usage", False))
            state.batch_last_usage_has_usage = has_usage
            if not has_usage:
                state.batch_current_usage_missing_count += 1
                _append_batch_log(state, "[LLM统计]")
                _append_batch_log(state, "usage 缺失，回退使用字符统计")
            else:
                state.batch_current_prompt_tokens += int(event.get("prompt_tokens", 0) or 0)
                state.batch_current_completion_tokens += int(event.get("completion_tokens", 0) or 0)
                state.batch_current_total_tokens += int(event.get("total_tokens", 0) or 0)
                state.batch_current_prompt_cache_hit_tokens += int(event.get("prompt_cache_hit_tokens", 0) or 0)
                state.batch_current_prompt_cache_miss_tokens += int(event.get("prompt_cache_miss_tokens", 0) or 0)
                state.batch_current_reasoning_tokens += int(event.get("reasoning_tokens", 0) or 0)
                state.batch_current_cached_tokens += int(event.get("cached_tokens", 0) or 0)
            continue
        if etype == "llm_usage_chars":
            if state.batch_last_usage_has_usage in {False, None}:
                state.batch_current_input_chars += int(event.get("input_chars", 0) or 0)
                state.batch_current_output_chars += int(event.get("output_chars", 0) or 0)
            state.batch_last_usage_has_usage = None
            continue
        if etype == "llm_tool_call":
            state.batch_current_tool_calls += 1
            _append_batch_log(state, "[LLM工具调用]")
            name = str(event.get("name", "")).strip()
            args = str(event.get("arguments", "")).strip()
            _append_batch_log(state, f"name={name}" if name else "（无工具名）")
            _append_batch_log(state, args or "（无参数）")
            continue
        if etype == "llm_tool_result":
            _append_batch_log(state, "[LLM工具结果]")
            _append_batch_log(state, str(event.get("json", "")).strip() or "（空）")
            continue
        if etype == "llm_tool_error":
            _append_batch_log(state, "[LLM工具错误]")
            text = str(event.get("text", "")).strip()
            if text:
                _append_batch_log(state, text)
            continue
        if etype == "llm_cot":
            _append_batch_log(state, "[LLM思考]")
            _append_batch_log(state, str(event.get("text", "")).strip() or "（空）")
            continue
        if etype == "llm_final_output":
            _append_batch_log(state, "[LLM最终输出]")
            _append_batch_log(state, str(event.get("text", "")).strip() or "（空）")
            continue
        if etype == "file_stats":
            asr_sec = float(event.get("asr_sec", 0.0) or 0.0)
            tr_sec = float(event.get("translate_sec", 0.0) or 0.0)
            write_sec = float(event.get("write_sec", 0.0) or 0.0)
            total_sec = float(event.get("total_sec", 0.0) or 0.0)
            status = str(event.get("status", "")).strip() or "unknown"
            lines = [
                f"文件：{event.get('file', '')}",
                f"状态：{status}",
                f"Prompt Tokens：{state.batch_current_prompt_tokens}",
                f"Completion Tokens：{state.batch_current_completion_tokens}",
                f"Total Tokens：{state.batch_current_total_tokens}",
                f"Reasoning Tokens：{state.batch_current_reasoning_tokens}",
                f"Prompt Cache Hit Tokens：{state.batch_current_prompt_cache_hit_tokens}",
                f"Prompt Cache Miss Tokens：{state.batch_current_prompt_cache_miss_tokens}",
                f"Prompt Cached Tokens：{state.batch_current_cached_tokens}",
                f"usage 缺失次数：{state.batch_current_usage_missing_count}",
                f"工具调用次数：{state.batch_current_tool_calls}",
                f"耗时：ASR {asr_sec:.2f}s | 翻译 {tr_sec:.2f}s | 写出 {write_sec:.2f}s | 总计 {total_sec:.2f}s",
            ]
            if state.batch_current_usage_missing_count > 0:
                lines.insert(8, f"输出字符数(回退)：{state.batch_current_output_chars}")
                lines.insert(8, f"输入字符数(回退)：{state.batch_current_input_chars}")
            state.batch_last_file_stats_lines = lines
            _append_batch_log(state, "[统计]")
            for line in lines:
                _append_batch_log(state, line)
            continue
        if etype == "file_end":
            _append_batch_log(
                state,
                f"[文件结束] {event.get('file', '')} | 状态={event.get('status', '')} | 输出={event.get('output', '')}"
            )
            lrc_output = str(event.get("lrc_output", "")).strip()
            if lrc_output:
                _append_batch_log(state, f"[LRC输出] {lrc_output}")
            if event.get("error"):
                _append_batch_log(state, f"[文件错误] {event.get('error')}")
            _close_current_log_file(state)
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
            if state.batch_last_file_stats_lines:
                state.batch_summary_lines.append("当前文件统计：")
                state.batch_summary_lines.extend(state.batch_last_file_stats_lines)
            _append_batch_log(state, f"[结束] 状态={status} 退出码={state.batch_result_rc}")
            _close_current_log_file(state)
            _append_operation_log(
                state,
                "batch_worker_done",
                {
                    "status": status,
                    "rc": state.batch_result_rc,
                    "summary_lines": state.batch_summary_lines,
                    "state_after": _state_snapshot(state),
                },
            )
            continue
        if etype == "worker_error":
            state.batch_running = False
            state.batch_finished = True
            state.batch_result_rc = 2
            state.batch_error = str(event.get("error", "未知错误"))
            tb_text = str(event.get("traceback", "")).strip()
            state.batch_summary_lines = [
                "状态：失败",
                f"错误：{state.batch_error}",
                f"总文件数：{state.batch_total_files}",
                f"最后处理文件：{state.batch_current_file or '（无）'}",
            ]
            _append_batch_log(state, f"[异常] {state.batch_error}")
            if tb_text:
                _append_batch_log(state, "[异常栈]")
                for line in tb_text.splitlines():
                    _append_batch_log(state, line)
                _append_crash_log_line("BATCH WORKER ERROR")
                _append_crash_log_line(tb_text)
            _close_current_log_file(state)
            _append_operation_log(
                state,
                "batch_worker_error",
                {
                    "error": state.batch_error,
                    "traceback": tb_text,
                    "state_after": _state_snapshot(state),
                },
            )
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


def _build_export_lrc_result(source_path: Path, output_dir: Path | None = None) -> list[str]:
    from whisperlrc.output.json_lrc_exporter import export_lrc_from_json_path

    out_paths = export_lrc_from_json_path(source_path, output_dir=output_dir)
    lines = [
        f"输入路径：{source_path}",
        f"导出数量：{len(out_paths)}",
    ]
    for p in out_paths:
        lines.append(f"LRC：{p}")
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


FIELD_PRESETS: dict[str, list[Any]] = {
    "asr.backend": ["faster_whisper"],
    "asr.device": ["cuda", "cpu", "auto", "mps"],
    "asr.compute_type": ["float16", "int8_float16", "int8", "float32", "bfloat16"],
    "translation.backend": ["llm"],
    "translation.llm_provider": ["openai_compatible"],
    "output.json_ext": [".json"],
}


def _apply_field_update(state: SessionState, section: str, section_obj: Any, field_name: str, value: Any) -> None:
    setattr(section_obj, field_name, value)
    state.edit_dirty = True
    if section == "output" and field_name == "default_input_dir":
        state.input_dir = Path(str(value))
    if section == "output" and field_name == "default_output_dir":
        state.output_dir = Path(str(value))


def _reset_edit_session(state: SessionState) -> None:
    state.edit_cfg = None
    state.edit_section = "asr"
    state.edit_offset = 0
    state.edit_translation_llm_only = False
    state.edit_dirty = False
    state.edit_menu_cursor = 0
    state.edit_field_cursor = 0
    state.edit_llm_cursor = 0


def _get_field_presets(section: str, field_name: str, value: Any) -> list[Any] | None:
    if isinstance(value, bool):
        return [False, True]
    return FIELD_PRESETS.get(f"{section}.{field_name}")


def _cycle_preset(values: list[Any], current: Any, delta: int) -> Any:
    if not values:
        return current
    try:
        idx = values.index(current)
    except ValueError:
        idx = 0
    return values[(idx + delta) % len(values)]


def _adjust_field_by_arrow(
    *,
    section: str,
    section_obj: Any,
    field_name: str,
    direction: int,
) -> tuple[bool, str]:
    current = getattr(section_obj, field_name)
    if isinstance(current, int):
        new_value = max(0, current + direction)
        setattr(section_obj, field_name, new_value)
        return (True, f"{field_name} = {new_value}")

    presets = _get_field_presets(section, field_name, current)
    if presets is None:
        return (False, f"{field_name} 不支持左右调整")
    new_value = _cycle_preset(presets, current, direction)
    setattr(section_obj, field_name, new_value)
    return (True, f"{field_name} = {_format_value_for_menu(new_value)}")


def _confirm_exit_config_edit(state: SessionState, *, exit_page: Page, stay_page: Page) -> Page:
    if state.edit_cfg is None:
        return exit_page
    if not state.edit_dirty:
        _reset_edit_session(state)
        return exit_page

    _clear_screen()
    _print_path_bar("主菜单->配置->修改配置项->退出确认")
    confirm = _confirm_action(f"检测到未保存修改，是否写入 {state.config_path}？")
    if confirm == "q":
        _reset_edit_session(state)
        return Page.MAIN
    if confirm == "esc":
        return stay_page
    if confirm == "y":
        try:
            save_config(state.config_path, state.edit_cfg)
            _reset_edit_session(state)
            return exit_page
        except Exception as e:
            return _show_info(
                state,
                "配置结果",
                [f"保存失败：{e}"],
                stay_page,
                "主菜单->配置->修改配置项->配置结果",
            )
    _reset_edit_session(state)
    return exit_page


def _render_main_page(state: SessionState) -> Page | None:
    options = ["批处理", "配置", "检查", "人工校对WebUI", "帮助"]
    action, idx = _run_arrow_menu(path="主菜单", title="WhisperLRC 主菜单", options=options)
    _append_operation_log(
        state,
        "main_menu_action",
        {"action": action, "index": idx, "option": options[idx] if 0 <= idx < len(options) else ""},
    )
    if action == "esc":
        return None
    if action == "q":
        return Page.MAIN
    mapping = {
        0: Page.BATCH,
        1: Page.CONFIG,
        2: Page.CHECK,
        3: Page.REVIEW_WEB,
        4: Page.HELP,
    }
    return mapping.get(idx, Page.MAIN)


def _render_batch_page(state: SessionState) -> Page:
    _consume_batch_events(state)
    if state.batch_running:
        return Page.PROCESSING

    options = [
        "使用当前会话默认参数执行",
        "自定义本次路径并执行",
        "查看当前运行参数",
    ]
    action, idx = _run_arrow_menu(path="主菜单->批处理", title="批处理页面", options=options)
    _append_operation_log(
        state,
        "batch_menu_action",
        {"action": action, "index": idx, "option": options[idx] if 0 <= idx < len(options) else ""},
    )
    if action == "q":
        return Page.MAIN
    if action == "esc":
        return Page.MAIN

    if idx == 0:
        confirm = _confirm_action("确认开始执行？")
        _append_operation_log(state, "batch_confirm_default_run", {"confirm": confirm})
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

    if idx == 1:
        input_res = _read_line_with_cancel("输入目录", str(state.input_dir), empty_as_default=True)
        if input_res.kind == "main":
            return Page.MAIN
        if input_res.kind != "value":
            return _show_info(state, "执行结果", ["已取消本次自定义执行。"], Page.BATCH, "主菜单->批处理->执行结果")

        output_res = _read_line_with_cancel("输出目录", str(state.output_dir), empty_as_default=True)
        if output_res.kind == "main":
            return Page.MAIN
        if output_res.kind != "value":
            return _show_info(state, "执行结果", ["已取消本次自定义执行。"], Page.BATCH, "主菜单->批处理->执行结果")

        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path), empty_as_default=True)
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "执行结果", ["已取消本次自定义执行。"], Page.BATCH, "主菜单->批处理->执行结果")

        confirm = _confirm_action("确认开始执行？")
        _append_operation_log(
            state,
            "batch_confirm_custom_run",
            {
                "confirm": confirm,
                "input_dir": input_res.value if input_res.kind == "value" else "",
                "output_dir": output_res.value if output_res.kind == "value" else "",
                "config": config_res.value if config_res.kind == "value" else "",
            },
        )
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

    if idx == 2:
        lines = [
            f"输入目录：{state.input_dir}",
            f"输出目录：{state.output_dir}",
            f"配置文件：{state.config_path}",
        ]
        return _show_info(state, "当前运行参数", lines, Page.BATCH, "主菜单->批处理->当前运行参数")

    return Page.BATCH


def _render_config_page(state: SessionState) -> Page:
    options = [
        "查看当前配置摘要",
        "修改配置项",
        "切换会话配置文件路径",
        "重置会话配置为 settings.toml",
    ]
    action, idx = _run_arrow_menu(path="主菜单->配置", title="配置页面", options=options)
    _append_operation_log(
        state,
        "config_menu_action",
        {"action": action, "index": idx, "option": options[idx] if 0 <= idx < len(options) else ""},
    )
    if action == "q":
        return Page.MAIN
    if action == "esc":
        return Page.MAIN

    if idx == 0:
        try:
            return _show_info(state, "当前配置摘要", _build_config_summary(state.config_path), Page.CONFIG, "主菜单->配置->当前配置摘要")
        except Exception as e:
            return _show_info(state, "当前配置摘要", [f"读取失败：{e}"], Page.CONFIG, "主菜单->配置->当前配置摘要")

    if idx == 1:
        try:
            state.edit_cfg = load_config(state.config_path)
            state.edit_section = "asr"
            state.edit_offset = 0
            state.edit_translation_llm_only = False
            state.edit_dirty = False
            state.edit_menu_cursor = 0
            state.edit_field_cursor = 0
            state.edit_llm_cursor = 0
            return Page.CONFIG_EDIT_MENU
        except Exception as e:
            return _show_info(state, "配置结果", [f"加载配置失败：{e}"], Page.CONFIG, "主菜单->配置->配置结果")

    if idx == 2:
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path), empty_as_default=True)
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

    if idx == 3:
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
    options = [
        "检查当前会话配置的翻译后端",
        "API 测试",
        "使用自定义配置文件检查",
        "使用自定义配置做 API 测试",
        "从默认输出目录导出 LRC",
        "从 JSON 路径导出 LRC",
    ]
    action, idx = _run_arrow_menu(path="主菜单->检查", title="检查页面", options=options)
    _append_operation_log(
        state,
        "check_menu_action",
        {"action": action, "index": idx, "option": options[idx] if 0 <= idx < len(options) else ""},
    )
    if action == "q":
        return Page.MAIN
    if action == "esc":
        return Page.MAIN

    if idx == 0:
        try:
            return _show_info(state, "翻译后端检查", _build_translation_check(state.config_path), Page.CHECK, "主菜单->检查->翻译后端检查")
        except Exception as e:
            return _show_info(state, "翻译后端检查", [f"检查失败：{e}"], Page.CHECK, "主菜单->检查->翻译后端检查")

    if idx == 1:
        _start_api_test(state, state.config_path)
        return Page.API_TEST

    if idx == 2:
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path), empty_as_default=True)
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "翻译后端检查", ["已取消本次检查。"], Page.CHECK, "主菜单->检查->翻译后端检查")
        try:
            return _show_info(state, "翻译后端检查", _build_translation_check(Path(config_res.value)), Page.CHECK, "主菜单->检查->翻译后端检查")
        except Exception as e:
            return _show_info(state, "翻译后端检查", [f"检查失败：{e}"], Page.CHECK, "主菜单->检查->翻译后端检查")

    if idx == 3:
        config_res = _read_line_with_cancel("配置文件路径", str(state.config_path), empty_as_default=True)
        if config_res.kind == "main":
            return Page.MAIN
        if config_res.kind != "value":
            return _show_info(state, "API 测试", ["已取消本次测试。"], Page.CHECK, "主菜单->检查->API测试")
        _start_api_test(state, Path(config_res.value))
        return Page.API_TEST

    if idx == 4:
        try:
            source_path = state.output_dir
            lines = _build_export_lrc_result(source_path, output_dir=source_path)
            return _show_info(state, "LRC 导出", lines, Page.CHECK, "主菜单->检查->LRC导出")
        except Exception as e:
            return _show_info(state, "LRC 导出", [f"导出失败：{e}"], Page.CHECK, "主菜单->检查->LRC导出")

    if idx == 5:
        path_res = _read_line_with_cancel("JSON 文件或目录路径", str(state.output_dir), empty_as_default=True)
        if path_res.kind == "main":
            return Page.MAIN
        if path_res.kind != "value":
            return _show_info(state, "LRC 导出", ["已取消本次导出。"], Page.CHECK, "主菜单->检查->LRC导出")
        out_res = _read_line_with_cancel("LRC 输出目录（可留空取消）", str(state.output_dir))
        if out_res.kind == "main":
            return Page.MAIN
        if out_res.kind != "value":
            return _show_info(state, "LRC 导出", ["已取消本次导出。"], Page.CHECK, "主菜单->检查->LRC导出")
        try:
            lines = _build_export_lrc_result(Path(path_res.value), output_dir=Path(out_res.value))
            return _show_info(state, "LRC 导出", lines, Page.CHECK, "主菜单->检查->LRC导出")
        except Exception as e:
            return _show_info(state, "LRC 导出", [f"导出失败：{e}"], Page.CHECK, "主菜单->检查->LRC导出")

    return Page.CHECK


def _render_config_edit_menu_page(state: SessionState) -> Page:
    if state.edit_cfg is None:
        return _show_info(state, "配置结果", ["未加载配置，请重新进入修改页面。"], Page.CONFIG, "主菜单->配置->配置结果")

    options = ["ASR", "管线", "翻译基础", "LLM 配置", "输出", "Schema", "返回上级"]
    action, idx = _run_arrow_menu(
        path="主菜单->配置->修改配置项",
        title="配置分区",
        options=options,
        cursor=state.edit_menu_cursor,
    )
    _append_operation_log(
        state,
        "config_edit_menu_action",
        {"action": action, "index": idx, "option": options[idx] if 0 <= idx < len(options) else ""},
    )
    state.edit_menu_cursor = idx
    if action == "q":
        return _confirm_exit_config_edit(state, exit_page=Page.MAIN, stay_page=Page.CONFIG_EDIT_MENU)
    if action == "esc":
        return _confirm_exit_config_edit(state, exit_page=Page.CONFIG, stay_page=Page.CONFIG_EDIT_MENU)
    if idx == 6:
        return _confirm_exit_config_edit(state, exit_page=Page.CONFIG, stay_page=Page.CONFIG_EDIT_MENU)

    if idx in {0, 1, 2, 4, 5}:
        mapping = {0: "asr", 1: "pipeline", 2: "translation", 4: "output", 5: "schema"}
        state.edit_section = mapping[idx]
        state.edit_offset = 0
        state.edit_translation_llm_only = False
        state.edit_field_cursor = 0
        return Page.CONFIG_EDIT_FIELDS

    if idx == 3:
        state.edit_section = "translation"
        state.edit_offset = 0
        state.edit_translation_llm_only = True
        state.edit_llm_cursor = 0
        return Page.CONFIG_EDIT_LLM

    return Page.CONFIG_EDIT_MENU


def _render_config_edit_fields_page(state: SessionState) -> Page:
    if state.edit_cfg is None:
        return _show_info(state, "配置结果", ["未加载配置，请重新进入修改页面。"], Page.CONFIG, "主菜单->配置->配置结果")

    notice = ""
    while True:
        section = state.edit_section
        section_obj = _get_section_obj(state.edit_cfg, section)
        all_items: list[tuple[str, Any]] = list(section_obj.__dict__.items())
        if section == "translation" and not state.edit_translation_llm_only:
            items = [(k, v) for k, v in all_items if (not k.startswith("llm_")) and k != "timeout_sec"]
        else:
            items = all_items
        if not items:
            return _show_info(
                state,
                "配置结果",
                ["当前分区没有可编辑字段。"],
                Page.CONFIG_EDIT_MENU,
                "主菜单->配置->修改配置项->配置结果",
            )

        state.edit_field_cursor = min(max(0, state.edit_field_cursor), len(items) - 1)
        _clear_screen()
        _print_path_bar(f"主菜单->配置->修改配置项->{SECTION_TITLES.get(section, section)}")
        print(f"{SECTION_TITLES.get(section, section)} 字段编辑")
        print("==============")
        if notice:
            print(notice)
            print()
        for i, (name, value) in enumerate(items):
            marker = ">" if i == state.edit_field_cursor else " "
            print(f"{marker} {name} = {_format_value_for_menu(value)}")

        key = _read_nav_key()
        if key == "up":
            state.edit_field_cursor = (state.edit_field_cursor - 1) % len(items)
            continue
        if key == "down":
            state.edit_field_cursor = (state.edit_field_cursor + 1) % len(items)
            continue
        if key in {"left", "right"}:
            field_name, _current = items[state.edit_field_cursor]
            _append_operation_log(
                state,
                "config_field_adjust",
                {
                    "section": section,
                    "field": field_name,
                    "direction": key,
                },
            )
            delta = -1 if key == "left" else 1
            ok, msg = _adjust_field_by_arrow(
                section=section,
                section_obj=section_obj,
                field_name=field_name,
                direction=delta,
            )
            if ok:
                new_value = getattr(section_obj, field_name)
                if isinstance(new_value, int):
                    new_value = max(0, new_value)
                    _apply_field_update(state, section, section_obj, field_name, new_value)
                else:
                    _apply_field_update(state, section, section_obj, field_name, new_value)
                notice = f"已更新 {msg}"
            else:
                notice = msg
            continue
        if key == "enter":
            field_name, current = items[state.edit_field_cursor]
            _append_operation_log(
                state,
                "config_field_enter_edit",
                {"section": section, "field": field_name, "current": str(current)},
            )
            value_res = _read_line_with_cancel(f"新值 {field_name}", str(current), empty_as_default=True)
            if value_res.kind == "main":
                return _confirm_exit_config_edit(state, exit_page=Page.MAIN, stay_page=Page.CONFIG_EDIT_FIELDS)
            if value_res.kind != "value":
                notice = f"已取消修改 {field_name}"
                continue
            try:
                new_value = _parse_field_value(section, field_name, current, value_res.value)
                _apply_field_update(state, section, section_obj, field_name, new_value)
                _append_operation_log(
                    state,
                    "config_field_updated",
                    {"section": section, "field": field_name, "new_value": str(new_value)},
                )
                notice = f"已更新 {field_name} = {_format_value_for_menu(new_value)}"
            except Exception as e:
                _append_operation_log(
                    state,
                    "config_field_update_error",
                    {"section": section, "field": field_name, "error": str(e)},
                )
                notice = f"更新失败：{e}"
            continue
        if key == "q":
            return _confirm_exit_config_edit(state, exit_page=Page.MAIN, stay_page=Page.CONFIG_EDIT_FIELDS)
        if key == "esc":
            return Page.CONFIG_EDIT_MENU


def _render_config_edit_llm_page(state: SessionState) -> Page:
    if state.edit_cfg is None:
        return _show_info(state, "配置结果", ["未加载配置，请重新进入修改页面。"], Page.CONFIG, "主菜单->配置->配置结果")

    notice = ""
    llm_keys = [
        "llm_provider",
        "llm_enable_thinking",
        "llm_model",
        "llm_base_url",
        "llm_api_key",
        "llm_prompt_file",
        "llm_preferences_file",
        "llm_batch_size",
        "llm_context_window",
        "timeout_sec",
    ]
    while True:
        section_obj = _get_section_obj(state.edit_cfg, "translation")
        items: list[tuple[str, Any]] = [(k, getattr(section_obj, k)) for k in llm_keys]
        state.edit_llm_cursor = min(max(0, state.edit_llm_cursor), len(items) - 1)

        _clear_screen()
        _print_path_bar("主菜单->配置->修改配置项->LLM配置")
        print("LLM 配置")
        print("==============")
        if notice:
            print(notice)
            print()
        for i, (name, value) in enumerate(items):
            marker = ">" if i == state.edit_llm_cursor else " "
            print(f"{marker} {name} = {_format_value_for_menu(value)}")

        key = _read_nav_key()
        if key == "up":
            state.edit_llm_cursor = (state.edit_llm_cursor - 1) % len(items)
            continue
        if key == "down":
            state.edit_llm_cursor = (state.edit_llm_cursor + 1) % len(items)
            continue
        if key in {"left", "right"}:
            field_name, _current = items[state.edit_llm_cursor]
            _append_operation_log(
                state,
                "llm_config_adjust",
                {"field": field_name, "direction": key},
            )
            delta = -1 if key == "left" else 1
            ok, msg = _adjust_field_by_arrow(
                section="translation",
                section_obj=section_obj,
                field_name=field_name,
                direction=delta,
            )
            if ok:
                new_value = getattr(section_obj, field_name)
                if isinstance(new_value, int):
                    new_value = max(0, new_value)
                    _apply_field_update(state, "translation", section_obj, field_name, new_value)
                else:
                    _apply_field_update(state, "translation", section_obj, field_name, new_value)
                notice = f"已更新 {msg}"
            else:
                notice = msg
            continue
        if key == "enter":
            field_name, current = items[state.edit_llm_cursor]
            _append_operation_log(
                state,
                "llm_config_enter_edit",
                {"field": field_name, "current": str(current)},
            )
            value_res = _read_line_with_cancel(f"新值 {field_name}", str(current), empty_as_default=True)
            if value_res.kind == "main":
                return _confirm_exit_config_edit(state, exit_page=Page.MAIN, stay_page=Page.CONFIG_EDIT_LLM)
            if value_res.kind != "value":
                notice = f"已取消修改 {field_name}"
                continue
            try:
                new_value = _parse_field_value("translation", field_name, current, value_res.value)
                _apply_field_update(state, "translation", section_obj, field_name, new_value)
                _append_operation_log(
                    state,
                    "llm_config_updated",
                    {"field": field_name, "new_value": str(new_value)},
                )
                notice = f"已更新 {field_name} = {_format_value_for_menu(new_value)}"
            except Exception as e:
                _append_operation_log(
                    state,
                    "llm_config_update_error",
                    {"field": field_name, "error": str(e)},
                )
                notice = f"更新失败：{e}"
            continue
        if key == "q":
            return _confirm_exit_config_edit(state, exit_page=Page.MAIN, stay_page=Page.CONFIG_EDIT_LLM)
        if key == "esc":
            return Page.CONFIG_EDIT_MENU


def _render_processing_page(state: SessionState) -> Page:
    if not state.batch_processing_header_printed:
        _print_path_bar("主菜单->批处理->处理中")
        print("处理页面")
        print("========")
        print(state.batch_running_path or "（未设置）")
        print("日志正在滚动输出")
        print()
        state.batch_processing_header_printed = True
        _append_operation_log(
            state,
            "processing_page_enter",
            {"running_path": state.batch_running_path, "state": _state_snapshot(state)},
        )

    while True:
        _consume_batch_events(state)

        if state.batch_log_cursor < len(state.batch_logs):
            for line in state.batch_logs[state.batch_log_cursor :]:
                print(line)
            state.batch_log_cursor = len(state.batch_logs)

        if state.batch_running:
            key = _read_key_nonblocking()
            if key == "esc":
                _append_operation_log(state, "processing_key", {"key": "esc", "state": _state_snapshot(state)})
                if not state.batch_requested_cancel and state.batch_cancel_token is not None:
                    state.batch_requested_cancel = True
                    _append_batch_log(state, "[操作] 已请求取消，等待当前步骤结束")
                    try:
                        cancel = getattr(state.batch_cancel_token, "cancel", None)
                        if callable(cancel):
                            cancel()
                    except Exception:
                        pass
                return Page.BATCH
            if key == "q":
                _append_operation_log(state, "processing_key", {"key": "q", "state": _state_snapshot(state)})
                return Page.MAIN
            time.sleep(0.12)
            continue

        lines = state.batch_summary_lines or [
            f"状态：{'已取消' if state.batch_requested_cancel else '已完成'}",
            f"退出码：{state.batch_result_rc}",
        ]
        _append_operation_log(
            state,
            "processing_finished",
            {"summary": lines, "state": _state_snapshot(state)},
        )
        state.batch_finished = False
        state.batch_processing_header_printed = False
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
        key = _read_single_key({"q"}, allow_esc=True)
        if key == "q":
            return Page.MAIN
        if key == "esc":
            return Page.CHECK


def _resolve_review_output_dir(state: SessionState) -> Path:
    try:
        cfg = load_config(state.config_path)
        out = Path(cfg.output.default_output_dir)
        return out if str(out).strip() else state.output_dir
    except Exception:
        return state.output_dir


def _run_review_web_server(state: SessionState, *, host: str, port: int, output_dir: Path) -> tuple[bool, str]:
    try:
        from whisperlrc.review_server import run_review_server
    except ModuleNotFoundError as e:
        return (False, f"缺少依赖：{e}。请安装 fastapi uvicorn pydantic")
    except Exception as e:
        return (False, f"加载 WebUI 模块失败：{e}")

    _clear_screen()
    _print_path_bar("主菜单->人工校对WebUI->服务中")
    print("人工校对 WebUI 服务已启动（独占模式）")
    print("================================")
    print(f"任务目录：{output_dir}")
    print(f"访问地址：http://{host}:{port}")
    print("按 Ctrl+C 停止服务并返回主菜单。")
    _append_operation_log(
        state,
        "review_web_server_start",
        {"host": host, "port": port, "output_dir": str(output_dir)},
    )
    try:
        run_review_server(output_dir=output_dir, host=host, port=port)
        _append_operation_log(
            state,
            "review_web_server_stop",
            {"reason": "normal_exit", "host": host, "port": port},
        )
        return (True, "WebUI 服务已停止。")
    except KeyboardInterrupt:
        _append_operation_log(
            state,
            "review_web_server_stop",
            {"reason": "keyboard_interrupt", "host": host, "port": port},
        )
        return (True, "WebUI 服务已停止。")
    except Exception as e:
        _append_operation_log(
            state,
            "review_web_server_error",
            {"host": host, "port": port, "error": str(e)},
        )
        return (False, f"WebUI 服务异常：{e}")


def _render_review_web_page(state: SessionState) -> Page:
    host = "127.0.0.1"
    port = 8765
    output_dir = _resolve_review_output_dir(state)
    options = ["启动服务（独占模式）", "返回主菜单"]
    action, idx = _run_arrow_menu(path="主菜单->人工校对WebUI", title="人工校对 WebUI", options=options)
    _append_operation_log(
        state,
        "review_web_menu_action",
        {
            "action": action,
            "index": idx,
            "option": options[idx] if 0 <= idx < len(options) else "",
            "output_dir": str(output_dir),
        },
    )
    if action in {"q", "esc"}:
        return Page.MAIN
    if idx == 1:
        return Page.MAIN

    confirm = _confirm_action(
        f"将以独占模式启动 WebUI 服务。\n地址：http://{host}:{port}\n任务目录：{output_dir}\n确认启动？"
    )
    _append_operation_log(
        state,
        "review_web_confirm_start",
        {"confirm": confirm, "host": host, "port": port, "output_dir": str(output_dir)},
    )
    if confirm == "q":
        return Page.MAIN
    if confirm != "y":
        return _show_info(
            state,
            "WebUI 结果",
            ["已取消启动 WebUI 服务。"],
            Page.MAIN,
            "主菜单->人工校对WebUI->结果",
        )

    ok, msg = _run_review_web_server(state, host=host, port=port, output_dir=output_dir)
    return _show_info(
        state,
        "WebUI 结果",
        [msg],
        Page.MAIN,
        "主菜单->人工校对WebUI->结果",
    )


def _render_help_page() -> Page:
    _print_path_bar("主菜单->帮助")
    print("帮助页面")
    print("========")
    print("- 菜单：方向键上下移动，回车确认")
    print("- q：返回主菜单")
    print("- Esc：全局返回（在主菜单中 Esc 退出）")
    print("- 配置页面退出时会要求 y/n 确认是否写入")
    print("- 信息结果以单独页面展示，Esc 关闭")
    print("- 提示词和翻译偏好文件路径可在配置中设置")
    print("- 在 prompt.txt 中可使用 {perf} 插入偏好字典")
    print("- role=user 只发送 {input}，prompt.txt 不再强制包含 {input}")
    print("- 检查页面支持 API 测试")
    print("- 检查页面支持从 JSON 导出 LRC")
    print("- 人工校对WebUI：主菜单可启动本地服务（Ctrl+C 停止）")

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
    _install_crash_logging(state)
    _open_operation_log_file(state)
    if state.op_log_path is not None:
        print(f"会话日志：{state.op_log_path}")
    if state.crash_log_path is not None:
        print(f"崩溃日志：{state.crash_log_path}")
    _append_operation_log(
        state,
        "app_start",
        {
            "state": _state_snapshot(state),
            "op_log_path": str(state.op_log_path) if state.op_log_path else "",
            "crash_log_path": str(state.crash_log_path) if state.crash_log_path else "",
        },
    )
    page: Page | None = Page.MAIN
    prev_page: Page | None = None

    try:
        while page is not None:
            if page != prev_page:
                _append_operation_log(
                    state,
                    "page_transition",
                    {
                        "from": prev_page.name if prev_page else None,
                        "to": page.name,
                        "state": _state_snapshot(state),
                    },
                )
                prev_page = page
            _clear_screen()
            if page == Page.MAIN:
                page = _render_main_page(state)
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
            if page == Page.REVIEW_WEB:
                page = _render_review_web_page(state)
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
    finally:
        _append_operation_log(
            state,
            "app_exit",
            {
                "state": _state_snapshot(state),
                "next_page": page.name if isinstance(page, Page) else None,
            },
        )
        _close_current_log_file(state)
        _close_operation_log_file(state)
        _close_crash_log_file(state)

    return 0

