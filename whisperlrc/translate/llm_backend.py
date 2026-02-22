from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from whisperlrc.config import TranslationConfig
from whisperlrc.translate.base import Translator
from whisperlrc.translate.relisten_tool import WhisperRelistenExecutor
from whisperlrc.translate.tooling import TranslationToolContext


class LLMTranslator(Translator):
    """
    OpenAI compatible LLM translator.
    """

    TOOL_NAME_RELISTEN = "asr_relisten_sentence"
    TOOL_MAX_CALLS_PER_GROUP = 5
    TOOL_MAX_ROUNDS = 8
    TOOL_MAX_CALLS_PER_INDEX = 1

    def __init__(self, cfg: TranslationConfig) -> None:
        self.cfg = cfg
        self._term_memory: dict[str, dict[str, str]] = {}
        self._preference_notes: list[str] = []
        self._prompt_template = self._load_prompt_template()
        self._load_preferences()

    def translate_batch(
        self,
        texts: list[str],
        src: str = "ja",
        tgt: str = "zh-Hans",
        retry: int = 0,
        event_cb: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: Any | None = None,
        tool_ctx: TranslationToolContext | None = None,
    ) -> list[str]:
        if not texts:
            return []
        self._validate_runtime_config()

        batch_size = max(1, int(self.cfg.llm_batch_size))
        total_groups = (len(texts) + batch_size - 1) // batch_size
        out: list[str] = []
        relisten_executor = WhisperRelistenExecutor(tool_ctx) if tool_ctx is not None else None

        for group_start in range(0, len(texts), batch_size):
            if self._is_cancelled(cancel_token):
                raise RuntimeError("用户取消处理")

            group_texts = texts[group_start : group_start + batch_size]
            group_index = (group_start // batch_size) + 1
            self._emit_event(
                event_cb,
                {
                    "type": "translation_group_start",
                    "group_index": group_index,
                    "total_groups": total_groups,
                    "group_start": group_start,
                    "group_size": len(group_texts),
                },
            )

            translated = self._translate_group_with_retry(
                all_texts=texts,
                group_texts=group_texts,
                group_start=group_start,
                src=src,
                tgt=tgt,
                retry=retry,
                event_cb=event_cb,
                cancel_token=cancel_token,
                group_index=group_index,
                total_groups=total_groups,
                relisten_executor=relisten_executor,
            )
            out.extend(translated)
            self._emit_event(
                event_cb,
                {
                    "type": "translation_group_end",
                    "group_index": group_index,
                    "total_groups": total_groups,
                    "group_start": group_start,
                    "group_size": len(group_texts),
                    "status": "ok",
                },
            )
        return out

    def test_api_hello(self) -> str:
        self._validate_runtime_config()
        content = self._request_chat_completion(
            "你是连通性检查助手。请用一句中文简短回复。",
            "hello",
        )
        text = content.strip()
        if not text:
            raise RuntimeError("API 返回为空")
        return text

    def _translate_group_with_retry(
        self,
        *,
        all_texts: list[str],
        group_texts: list[str],
        group_start: int,
        src: str,
        tgt: str,
        retry: int,
        event_cb: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: Any | None = None,
        group_index: int,
        total_groups: int,
        relisten_executor: WhisperRelistenExecutor | None = None,
    ) -> list[str]:
        max_attempts = max(1, retry + 1)
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            if self._is_cancelled(cancel_token):
                raise RuntimeError("用户取消处理")
            try:
                input_payload = self._build_input_payload(
                    all_texts=all_texts,
                    group_texts=group_texts,
                    group_start=group_start,
                    src=src,
                    tgt=tgt,
                )
                system_prompt = self._build_system_prompt(
                    src=src,
                    tgt=tgt,
                )
                request_meta = {
                    "group_index": group_index,
                    "total_groups": total_groups,
                    "group_start": group_start,
                    "group_size": len(group_texts),
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                }
                raw = ""
                try:
                    raw = self._request_chat_completion(
                        system_prompt,
                        input_payload,
                        event_cb=event_cb,
                        request_meta={**request_meta, "phase": "direct"},
                    )
                    parsed = self._parse_json_response(raw, expected_count=len(group_texts))
                except Exception as direct_err:
                    if relisten_executor is None:
                        raise direct_err
                    self._emit_event(
                        event_cb,
                        {
                            "type": "llm_tool_error",
                            "text": f"直出失败，进入工具会话：{direct_err}",
                            "meta": {**request_meta, "phase": "direct"},
                        },
                    )
                    raw = self._request_chat_completion_with_tools(
                        system_prompt,
                        input_payload,
                        event_cb=event_cb,
                        request_meta={**request_meta, "phase": "tool_session"},
                        group_start=group_start,
                        group_size=len(group_texts),
                        relisten_executor=relisten_executor,
                    )
                    parsed = self._parse_json_response(raw, expected_count=len(group_texts))
                if self._is_cancelled(cancel_token):
                    raise RuntimeError("用户取消处理")
                translations, terms, cot = parsed
                self._merge_terms(terms)
                if cot:
                    self._emit_event(
                        event_cb,
                        {
                            "type": "llm_cot",
                            "text": cot,
                            "meta": {
                                "group_index": group_index,
                                "total_groups": total_groups,
                                "group_start": group_start,
                                "group_size": len(group_texts),
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                            },
                        },
                    )
                return translations
            except Exception as e:
                last_err = e
                logging.warning(
                    "LLM 翻译分组失败（第 %d/%d 次，分组起始索引=%d）：%s",
                    attempt,
                    max_attempts,
                    group_start,
                    e,
                )
        raise RuntimeError(f"LLM 翻译分组失败：{last_err}")

    def _validate_runtime_config(self) -> None:
        provider = self.cfg.llm_provider.strip().lower()
        if provider not in {"openai_compatible", "openai-compatible", "openai"}:
            raise ValueError(f"不支持的 llm_provider：{self.cfg.llm_provider}")
        if not self.cfg.llm_model.strip():
            raise ValueError("缺少配置：translation.llm_model")
        if not self.cfg.llm_base_url.strip():
            raise ValueError("缺少配置：translation.llm_base_url")
        if not self.cfg.llm_api_key.strip():
            raise ValueError("缺少配置：translation.llm_api_key")

    def _normalize_chat_url(self) -> str:
        base = self.cfg.llm_base_url.strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _perform_chat_request(
        self,
        req_payload: dict[str, Any],
        *,
        event_cb: Callable[[dict[str, Any]], None] | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._normalize_chat_url()
        req_body = json.dumps(req_payload, ensure_ascii=False)
        self._emit_event(
            event_cb,
            {
                "type": "llm_request",
                "url": url,
                "json": self._safe_json_dump(req_payload),
                "meta": request_meta or {},
            },
        )

        req = request.Request(
            url=url,
            data=req_body.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.llm_api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=max(1, int(self.cfg.timeout_sec))) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"LLM HTTP 错误 {e.code}：{detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"LLM 网络请求失败：{e}") from e

        self._emit_event(
            event_cb,
            {
                "type": "llm_response",
                "url": url,
                "json": self._safe_json_dump(body),
                "meta": request_meta or {},
            },
        )
        self._emit_event(
            event_cb,
            {
                "type": "llm_usage_chars",
                "input_chars": len(req_body),
                "output_chars": len(body),
                "meta": request_meta or {},
            },
        )

        try:
            data = json.loads(body)
        except Exception as e:
            raise RuntimeError(f"LLM 返回不是合法 JSON：{e}") from e
        return data

    def _request_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        event_cb: Callable[[dict[str, Any]], None] | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> str:
        req_payload = {
            "model": self.cfg.llm_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self._perform_chat_request(req_payload, event_cb=event_cb, request_meta=request_meta)
        try:
            message = data["choices"][0]["message"]
        except Exception as e:
            raise RuntimeError(f"LLM 返回 JSON 结构异常：{e}") from e

        content = self._message_content_to_text(message.get("content"))
        if not content:
            raise RuntimeError("LLM 返回内容为空")
        return content

    def _request_chat_completion_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        event_cb: Callable[[dict[str, Any]], None] | None = None,
        request_meta: dict[str, Any] | None = None,
        group_start: int,
        group_size: int,
        relisten_executor: WhisperRelistenExecutor | None,
    ) -> str:
        if relisten_executor is None:
            return self._request_chat_completion(
                system_prompt,
                user_prompt,
                event_cb=event_cb,
                request_meta=request_meta,
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tools = [self._build_relisten_tool_spec()]
        used_calls = 0
        per_index_calls: dict[int, int] = {}
        tools_enabled = True

        for round_idx in range(1, self.TOOL_MAX_ROUNDS + 1):
            req_payload: dict[str, Any] = {
                "model": self.cfg.llm_model,
                "temperature": 0,
                "messages": messages,
            }
            if tools_enabled:
                req_payload["tools"] = tools
                req_payload["tool_choice"] = "auto"

            round_meta = dict(request_meta or {})
            round_meta.update({"tool_round": round_idx, "tools_enabled": tools_enabled})
            try:
                data = self._perform_chat_request(req_payload, event_cb=event_cb, request_meta=round_meta)
            except Exception as e:
                if tools_enabled and self._is_tools_unsupported_error(e):
                    self._emit_event(
                        event_cb,
                        {
                            "type": "llm_tool_error",
                            "text": f"模型端不支持 tools，降级为普通请求：{e}",
                            "meta": round_meta,
                        },
                    )
                    tools_enabled = False
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    continue
                raise

            try:
                message = data["choices"][0]["message"]
            except Exception as e:
                raise RuntimeError(f"LLM 返回 JSON 结构异常：{e}") from e

            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            if tools_enabled and isinstance(tool_calls, list) and tool_calls:
                force_finalize = False
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._message_content_to_text(message.get("content")),
                        "tool_calls": tool_calls,
                    }
                )
                for call in tool_calls:
                    used_calls += 1
                    call_meta = dict(round_meta)
                    call_meta.update({"tool_calls_used": used_calls, "tool_calls_budget": self.TOOL_MAX_CALLS_PER_GROUP})
                    call_name = str(call.get("function", {}).get("name", "")).strip()
                    call_args = str(call.get("function", {}).get("arguments", "")).strip()
                    call_index = self._extract_tool_index(call_args)
                    self._emit_event(
                        event_cb,
                        {
                            "type": "llm_tool_call",
                            "name": call_name,
                            "arguments": call_args,
                            "meta": call_meta,
                        },
                    )

                    if used_calls > self.TOOL_MAX_CALLS_PER_GROUP:
                        result_obj = {
                            "ok": False,
                            "error": f"工具调用已达上限 {self.TOOL_MAX_CALLS_PER_GROUP}",
                        }
                        force_finalize = True
                    elif (
                        call_index is not None
                        and per_index_calls.get(call_index, 0) >= self.TOOL_MAX_CALLS_PER_INDEX
                    ):
                        result_obj = {
                            "ok": False,
                            "error": f"index={call_index} 已调用过，禁止重复重听",
                            "index": call_index,
                        }
                        force_finalize = True
                    else:
                        result_obj = self._execute_tool_call(
                            call_name=call_name,
                            arguments=call_args,
                            group_start=group_start,
                            group_size=group_size,
                            relisten_executor=relisten_executor,
                        )
                        if call_index is not None:
                            per_index_calls[call_index] = per_index_calls.get(call_index, 0) + 1

                    result_json = self._safe_json_dump(result_obj)
                    self._emit_event(
                        event_cb,
                        {
                            "type": "llm_tool_result",
                            "name": call_name,
                            "json": result_json,
                            "meta": call_meta,
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.get("id", "")),
                            "name": call_name,
                            "content": result_json,
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": self._build_tool_session_user_message(
                            used_calls=used_calls,
                            per_index_calls=per_index_calls,
                            force_finalize=force_finalize,
                        ),
                    }
                )
                if force_finalize:
                    tools_enabled = False
                continue

            content = self._message_content_to_text(message.get("content"))
            if content:
                return content
            raise RuntimeError("LLM 返回为空，且没有 tool_calls")

        raise RuntimeError("LLM tool 调用轮数超限，终止本组")

    def _build_relisten_tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.TOOL_NAME_RELISTEN,
                "description": "针对当前翻译分组中的某一句执行 Whisper 重听，返回 3 条识别候选文本。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "当前分组内句子索引（0-based）。",
                        }
                    },
                    "required": ["index"],
                },
            },
        }

    def _execute_tool_call(
        self,
        *,
        call_name: str,
        arguments: str,
        group_start: int,
        group_size: int,
        relisten_executor: WhisperRelistenExecutor,
    ) -> dict[str, Any]:
        if call_name != self.TOOL_NAME_RELISTEN:
            return {"ok": False, "error": f"未知工具：{call_name}"}

        try:
            arg_obj = json.loads(arguments) if arguments else {}
        except Exception as e:
            return {"ok": False, "error": f"工具参数不是合法 JSON：{e}", "arguments": arguments}

        if not isinstance(arg_obj, dict):
            return {"ok": False, "error": "工具参数必须是对象"}

        raw_idx = arg_obj.get("index")
        try:
            local_idx = int(raw_idx)
        except Exception:
            return {"ok": False, "error": f"index 非法：{raw_idx}"}

        if local_idx < 0 or local_idx >= group_size:
            return {"ok": False, "error": f"index 越界：{local_idx}", "group_size": group_size}

        global_idx = group_start + local_idx
        return relisten_executor.relisten_by_global_index(global_idx)

    def _extract_tool_index(self, arguments: str) -> int | None:
        if not arguments:
            return None
        try:
            obj = json.loads(arguments)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        raw_idx = obj.get("index")
        try:
            return int(raw_idx)
        except Exception:
            return None

    def _build_tool_session_user_message(
        self,
        *,
        used_calls: int,
        per_index_calls: dict[int, int],
        force_finalize: bool,
    ) -> str:
        summary = json.dumps(
            {
                "used_calls": used_calls,
                "call_limit": self.TOOL_MAX_CALLS_PER_GROUP,
                "per_index_calls": per_index_calls,
                "per_index_limit": self.TOOL_MAX_CALLS_PER_INDEX,
            },
            ensure_ascii=False,
        )
        if force_finalize:
            return (
                "工具调用已收敛，请不要继续调用工具。"
                "请基于已有上下文与工具结果，直接输出最终 JSON。"
                f"\n会话摘要：{summary}"
            )
        return (
            "请先检查当前会话摘要，避免重复调用同一句。"
            "若仍有明显识别问题再调用工具；否则直接输出最终 JSON。"
            f"\n会话摘要：{summary}"
        )

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    def _is_tools_unsupported_error(self, err_obj: Exception) -> bool:
        text = str(err_obj).lower()
        if "tool" not in text:
            return False
        hints = ["unsupported", "not support", "unknown", "invalid", "unrecognized"]
        return any(h in text for h in hints)

    def _emit_event(self, event_cb: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
        if event_cb is None:
            return
        try:
            event_cb(payload)
        except Exception:
            pass

    def _is_cancelled(self, cancel_token: Any | None) -> bool:
        if cancel_token is None:
            return False
        checker = getattr(cancel_token, "is_cancelled", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return bool(getattr(cancel_token, "cancelled", False))

    def _safe_json_dump(self, obj: Any) -> str:
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                return obj
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def _build_system_prompt(self, *, src: str, tgt: str) -> str:
        perf_text = self._render_perf_block()
        prompt = self._prompt_template
        if "{perf}" in prompt:
            prompt = prompt.replace("{perf}", perf_text)
        else:
            prompt = prompt.rstrip() + "\n\n当前翻译偏好：\n" + perf_text
        if "{input}" in prompt:
            prompt = prompt.replace("{input}", "[输入内容由 user 消息提供]")

        lines = [
            prompt,
            f"源语言：{src}",
            f"目标语言：{tgt}",
            "输出要求：最终必须只输出 JSON，不能包含 Markdown 代码块或解释文字。",
            "输出结构：",
            '{ "translations":[{"index":0,"text":"翻译结果"}], "terms":[{"src":"原文术语","tgt":"译文术语","note":"可选说明"}] }',
        ]
        return "\n".join(lines)

    def _build_input_payload(
        self,
        *,
        all_texts: list[str],
        group_texts: list[str],
        group_start: int,
        src: str,
        tgt: str,
    ) -> str:
        context_window = max(0, int(self.cfg.llm_context_window))
        items: list[dict[str, Any]] = []
        for local_idx, text in enumerate(group_texts):
            global_idx = group_start + local_idx
            prev_start = max(0, global_idx - context_window)
            next_end = min(len(all_texts), global_idx + context_window + 1)
            context_prev = all_texts[prev_start:global_idx]
            context_next = all_texts[global_idx + 1 : next_end]
            items.append(
                {
                    "index": local_idx,
                    "text": text,
                    "context_prev": context_prev,
                    "context_next": context_next,
                }
            )
        payload = json.dumps(items, ensure_ascii=False, indent=2)
        return (
            f"请将以下句子从 {src} 翻译到 {tgt}。\n"
            "上下文只用于理解语义，不需要翻译上下文本身。\n"
            "请按 index 返回每条翻译。\n"
            "输入数据（JSON）：\n"
            f"{payload}"
        )

    def _parse_json_response(
        self,
        content: str,
        *,
        expected_count: int,
    ) -> tuple[list[str], list[dict[str, str]], str]:
        think_cot, json_payload = self._extract_think_block(content)
        data = self._load_json_object_robust(json_payload)
        if not isinstance(data, dict):
            raise RuntimeError("JSON 顶层必须是对象")

        raw_translations = data.get("translations")
        if not isinstance(raw_translations, list):
            raise RuntimeError("JSON 缺少 translations 列表")

        result: list[str | None] = [None] * expected_count
        for item in raw_translations:
            if not isinstance(item, dict):
                raise RuntimeError("translations 项必须是对象")
            index = item.get("index")
            text = item.get("text")
            if not isinstance(index, int):
                raise RuntimeError("translations.index 必须是整数")
            if index < 0 or index >= expected_count:
                raise RuntimeError(f"translations.index 越界：{index}")
            if not isinstance(text, str):
                raise RuntimeError("translations.text 必须是字符串")
            result[index] = text

        if any(v is None for v in result):
            raise RuntimeError("translations 不完整，存在缺失索引")
        translations = [str(v) for v in result]

        terms: list[dict[str, str]] = []
        raw_terms = data.get("terms")
        if isinstance(raw_terms, list):
            for item in raw_terms:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("src", "")).strip()
                tgt = str(item.get("tgt", "")).strip()
                note = str(item.get("note", "")).strip()
                if not src or not tgt:
                    continue
                terms.append({"src": src, "tgt": tgt, "note": note})
        return translations, terms, think_cot

    def _strip_markdown_fence(self, text: str) -> str:
        s = text.strip()
        if not s.startswith("```"):
            return s
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _load_json_object_robust(self, content: str) -> dict[str, Any]:
        normalized = self._strip_markdown_fence(content).strip()
        try:
            obj = json.loads(normalized)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        decoder = json.JSONDecoder()
        for i, ch in enumerate(normalized):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(normalized, i)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        raise RuntimeError("JSON 解析失败：未找到有效 JSON 对象")

    def _extract_think_block(self, content: str) -> tuple[str, str]:
        text = content.strip()
        pattern = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
        match = pattern.search(text)
        if not match:
            return "", text
        think_text = match.group(1).strip()
        cleaned = pattern.sub("", text, count=1).strip()
        return think_text, cleaned

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_config_path(self, raw_path: str) -> Path:
        v = raw_path.strip()
        if not v:
            raise RuntimeError("配置路径不能为空")
        p = Path(v)
        if p.is_absolute():
            return p
        return self._project_root() / p

    def _load_prompt_template(self) -> str:
        prompt_path = self._resolve_config_path(self.cfg.llm_prompt_file)
        if not prompt_path.exists():
            raise RuntimeError(f"缺少提示词文件：{prompt_path}")
        text = prompt_path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"提示词文件为空：{prompt_path}")
        return text

    def _load_preferences(self) -> None:
        pref_path = self._resolve_config_path(self.cfg.llm_preferences_file)
        if not pref_path.exists():
            logging.warning("未找到偏好文件，将以空偏好继续：%s", pref_path)
            return
        lines = pref_path.read_text(encoding="utf-8").splitlines()
        for idx, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("note:"):
                note = line[5:].strip()
                if note:
                    self._preference_notes.append(note)
                continue
            if "=>" not in line:
                logging.warning("偏好文件第 %d 行无法解析，已跳过：%s", idx, raw)
                continue
            left, right = line.split("=>", 1)
            src = left.strip()
            right_part = right.strip()
            tgt = right_part
            note = ""
            if "|" in right_part:
                tgt_part, note_part = right_part.split("|", 1)
                tgt = tgt_part.strip()
                note = note_part.strip()
            if not src or not tgt:
                logging.warning("偏好文件第 %d 行缺少 src/tgt，已跳过：%s", idx, raw)
                continue
            self._add_term_from_obj({"src": src, "tgt": tgt, "note": note})

    def _render_perf_block(self) -> str:
        lines: list[str] = []
        if self._term_memory:
            for item in self._term_memory.values():
                note = item.get("note", "").strip()
                if note:
                    lines.append(f"- {item['src']} => {item['tgt']}（{note}）")
                else:
                    lines.append(f"- {item['src']} => {item['tgt']}")
        if self._preference_notes:
            lines.append("偏好说明：")
            for note in self._preference_notes:
                lines.append(f"- {note}")
        if not lines:
            return "- （无偏好）"
        return "\n".join(lines)

    def _add_term_from_obj(self, item: dict[str, Any]) -> None:
        src = str(item.get("src", "")).strip()
        tgt = str(item.get("tgt", "")).strip()
        note = str(item.get("note", "")).strip()
        if not src or not tgt:
            return
        self._term_memory[src] = {"src": src, "tgt": tgt, "note": note}

    def _merge_terms(self, terms: list[dict[str, str]]) -> None:
        for item in terms:
            self._add_term_from_obj(item)
