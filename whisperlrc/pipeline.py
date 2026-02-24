from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from whisperlrc.config import AppConfig
from whisperlrc.output.lrc_writer import write_lrc
from whisperlrc.output.review_json_writer import write_review_json
from whisperlrc.translate.factory import build_translator
from whisperlrc.translate.tooling import SentenceRef, TranslationToolContext
from whisperlrc.types import FileProcessResult, SentenceItem, WordItem


class CancelToken:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled


@dataclass
class CacheResumeState:
    cache_path: Path
    sentences: list[SentenceItem]
    duration_sec: float
    completed_prefix: int
    status: str


def _iter_audio_files(input_dir: Path, exts: list[str]) -> list[Path]:
    ext_set = {e.lower().lstrip(".") for e in exts}
    out: list[Path] = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in ext_set:
            out.append(p)
    return sorted(out)


def _run_asr_with_retry(audio_file: Path, asr_engine: Any, retry: int) -> tuple[Any | None, str | None]:
    max_attempts = retry + 1
    err: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return asr_engine.transcribe(str(audio_file)), None
        except Exception as e:
            err = f"ASR 第 {attempt}/{max_attempts} 次尝试失败：{e}"
            logging.error(err)
    return None, err


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_sentence_done(sentence: SentenceItem) -> bool:
    if sentence.translation_status != "ok":
        return False
    text = (sentence.zh_text or "").strip()
    return bool(text)


def _count_completed_prefix(sentences: list[SentenceItem]) -> int:
    count = 0
    for s in sentences:
        if not _is_sentence_done(s):
            break
        count += 1
    return count


def _mark_unfinished_failed(sentences: list[SentenceItem]) -> None:
    for s in sentences:
        if _is_sentence_done(s):
            continue
        s.zh_text = None
        s.translation_status = "failed"


def _total_groups(total_sentences: int, batch_size: int) -> int:
    if total_sentences <= 0:
        return 0
    return (total_sentences + batch_size - 1) // batch_size


def _completed_groups(completed_sentences: int, batch_size: int) -> int:
    if completed_sentences <= 0:
        return 0
    return (completed_sentences + batch_size - 1) // batch_size


def _build_source_meta(audio_file: Path) -> dict[str, Any]:
    try:
        st = audio_file.stat()
        return {
            "name": audio_file.name,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "path": str(audio_file),
        }
    except Exception:
        return {
            "name": audio_file.name,
            "path": str(audio_file),
        }


def _json_cache_path(output_dir: Path, base_name: str, cfg: AppConfig) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{base_name}{cfg.output.json_ext}"


def _sentence_from_cache_obj(obj: Any) -> SentenceItem:
    if not isinstance(obj, dict):
        raise ValueError("sentence 项必须是对象")

    words_raw = obj.get("word_items")
    words: list[WordItem] = []
    if isinstance(words_raw, list):
        for w in words_raw:
            if not isinstance(w, dict):
                continue
            words.append(
                WordItem(
                    word=str(w.get("word", "")),
                    start_sec=_safe_float(w.get("start_sec"), 0.0) if w.get("start_sec") is not None else None,
                    end_sec=_safe_float(w.get("end_sec"), 0.0) if w.get("end_sec") is not None else None,
                    confidence=_safe_float(w.get("confidence"), 0.0) if w.get("confidence") is not None else None,
                )
            )

    zh_text_raw = obj.get("zh_text")
    zh_text = str(zh_text_raw) if zh_text_raw is not None else None
    status = str(obj.get("translation_status", "")).strip()
    if not status:
        status = "ok" if (zh_text and zh_text.strip()) else "pending"

    return SentenceItem(
        sentence_id=str(obj.get("sentence_id", "")),
        start_sec=_safe_float(obj.get("start_sec"), 0.0),
        end_sec=_safe_float(obj.get("end_sec"), 0.0),
        ja_text=str(obj.get("ja_text", "")),
        zh_text=zh_text,
        translation_status=status,
        segment_confidence=_safe_float(obj.get("segment_confidence"), 0.0)
        if obj.get("segment_confidence") is not None
        else None,
        review_text=str(obj.get("review_text")) if obj.get("review_text") is not None else None,
        word_items=words,
        token_items=[],
    )


def _audio_meta_match(payload: dict[str, Any], audio_file: Path) -> tuple[bool, str]:
    source_meta = payload.get("source_meta")
    if isinstance(source_meta, dict):
        actual = _build_source_meta(audio_file)
        expected_name = str(source_meta.get("name", "")).strip()
        if expected_name and expected_name != audio_file.name:
            return False, f"缓存音频名不匹配：{expected_name} != {audio_file.name}"
        expected_size = source_meta.get("size")
        if expected_size is not None and _safe_int(expected_size, -1) != _safe_int(actual.get("size"), -2):
            return False, "缓存音频大小不匹配"
        expected_mtime = source_meta.get("mtime_ns")
        if expected_mtime is not None and _safe_int(expected_mtime, -1) != _safe_int(actual.get("mtime_ns"), -2):
            return False, "缓存音频修改时间不匹配"
        return True, ""

    audio_path = str(payload.get("audio_path", "")).strip()
    if audio_path and Path(audio_path).name != audio_file.name:
        return False, f"缓存 audio_path 不匹配：{audio_path}"
    return True, ""


def _load_resume_cache(cache_path: Path, audio_file: Path) -> tuple[CacheResumeState | None, str]:
    if not cache_path.exists():
        return None, "missing"
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"读取缓存失败：{e}"

    if not isinstance(payload, dict):
        return None, "缓存根对象不是 JSON object"

    ok, reason = _audio_meta_match(payload, audio_file)
    if not ok:
        return None, reason

    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, list):
        return None, "缓存缺少 sentences 列表"
    try:
        sentences = [_sentence_from_cache_obj(x) for x in raw_sentences]
    except Exception as e:
        return None, f"缓存句子解析失败：{e}"
    if not sentences:
        return None, "缓存句子为空"

    completed_prefix = _count_completed_prefix(sentences)
    duration_sec = _safe_float(payload.get("duration_sec"), 0.0)
    status = str(payload.get("status", "")).strip()
    return (
        CacheResumeState(
            cache_path=cache_path,
            sentences=sentences,
            duration_sec=duration_sec,
            completed_prefix=completed_prefix,
            status=status,
        ),
        "",
    )


def _build_asr_output_lines(sentences: list[SentenceItem]) -> str:
    lines: list[str] = []
    for sent in sentences:
        conf = "" if sent.segment_confidence is None else f" | conf={sent.segment_confidence:.4f}"
        lines.append(f"[{sent.start_sec:.2f}-{sent.end_sec:.2f}] {sent.ja_text}{conf}")
    return "\n".join(lines)


def _write_json_snapshot(
    *,
    output_dir: Path,
    base_name: str,
    audio_path: str,
    duration_sec: float,
    sentences: list[SentenceItem],
    status: str,
    error: str | None,
    cfg: AppConfig,
    source_meta: dict[str, Any],
    batch_size: int,
    asr_sec: float,
    translate_sec: float,
    write_sec: float,
) -> Path:
    total_sentences = len(sentences)
    translated_sentences = _count_completed_prefix(sentences)
    total_group_count = _total_groups(total_sentences, batch_size)
    completed_group_count = _completed_groups(translated_sentences, batch_size)
    progress = {
        "total_sentences": total_sentences,
        "translated_sentences": translated_sentences,
        "total_groups": total_group_count,
        "completed_groups": completed_group_count,
        "last_group_index": completed_group_count if completed_group_count > 0 else 0,
    }
    runtime = {
        "asr_sec": asr_sec,
        "translate_sec": translate_sec,
        "write_sec": write_sec,
    }
    result = FileProcessResult(status=status, sentences=sentences, error=error, logs=[])
    return write_review_json(
        output_dir=output_dir,
        base_name=base_name,
        audio_path=audio_path,
        duration_sec=duration_sec,
        result=result,
        cfg=cfg,
        progress=progress,
        runtime=runtime,
        source_meta=source_meta,
    )


def _remap_translation_event(
    event: dict[str, Any],
    *,
    sentence_offset: int,
    group_offset: int,
    total_groups_all: int,
) -> dict[str, Any]:
    mapped = dict(event)
    etype = str(mapped.get("type", "")).strip()

    if etype in {"translation_group_start", "translation_group_end", "translation_group_result"}:
        mapped["group_start"] = sentence_offset + _safe_int(mapped.get("group_start", 0), 0)
        mapped["group_index"] = group_offset + _safe_int(mapped.get("group_index", 0), 0)
        mapped["total_groups"] = total_groups_all

    meta_obj = mapped.get("meta")
    if isinstance(meta_obj, dict):
        meta = dict(meta_obj)
        if "group_start" in meta:
            meta["group_start"] = sentence_offset + _safe_int(meta.get("group_start", 0), 0)
        if "group_index" in meta:
            meta["group_index"] = group_offset + _safe_int(meta.get("group_index", 0), 0)
        if "total_groups" in meta:
            meta["total_groups"] = total_groups_all
        mapped["meta"] = meta
    return mapped


def process_batch(
    input_dir: Path,
    output_dir: Path,
    cfg: AppConfig,
    event_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> int:
    def emit(event: dict[str, Any]) -> None:
        if event_cb is None:
            return
        try:
            event_cb(event)
        except Exception:
            pass

    audio_files = _iter_audio_files(input_dir, cfg.pipeline.input_formats)
    emit(
        {
            "type": "batch_start",
            "total_files": len(audio_files),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        }
    )
    if not audio_files:
        logging.warning("在目录 %s 中未找到可处理的音频文件", input_dir)
        emit({"type": "batch_end", "ok": 0, "failed": 0, "total_files": 0, "cancelled": False, "rc": 0})
        return 0

    from whisperlrc.asr.faster_whisper_engine import FasterWhisperEngine

    asr_engine = FasterWhisperEngine(cfg.asr)
    ok = 0
    failed = 0
    total = len(audio_files)
    batch_size = max(1, int(cfg.translation.llm_batch_size))

    for idx, audio_file in enumerate(audio_files, start=1):
        file_started_at = time.perf_counter()
        asr_sec = 0.0
        translate_sec = 0.0
        write_sec = 0.0

        def emit_file_stats(status: str) -> None:
            total_sec = max(0.0, time.perf_counter() - file_started_at)
            emit(
                {
                    "type": "file_stats",
                    "file_index": idx,
                    "total_files": total,
                    "file": audio_file.name,
                    "status": status,
                    "asr_sec": asr_sec,
                    "translate_sec": translate_sec,
                    "write_sec": write_sec,
                    "total_sec": total_sec,
                }
            )

        if cancel_token is not None and cancel_token.is_cancelled():
            logging.warning("检测到取消请求，停止批处理。")
            emit(
                {
                    "type": "cancelled",
                    "stage": "before_file",
                    "current_file_index": idx,
                    "total_files": total,
                    "file": audio_file.name,
                }
            )
            emit(
                {
                    "type": "batch_end",
                    "ok": ok,
                    "failed": failed,
                    "total_files": total,
                    "cancelled": True,
                    "rc": 2,
                }
            )
            return 2

        emit(
            {
                "type": "file_start",
                "file_index": idx,
                "total_files": total,
                "file": audio_file.name,
            }
        )
        logging.info("开始处理：%s", audio_file.name)

        base_name = audio_file.stem
        json_path = _json_cache_path(output_dir, base_name, cfg)
        source_meta = _build_source_meta(audio_file)
        duration_sec = 0.0
        sentences: list[SentenceItem] = []
        resume_completed_prefix = 0

        resume_state, resume_reason = _load_resume_cache(json_path, audio_file)
        if resume_state is not None:
            sentences = resume_state.sentences
            duration_sec = resume_state.duration_sec
            resume_completed_prefix = resume_state.completed_prefix
            emit(
                {
                    "type": "json_cache_resume_loaded",
                    "file": audio_file.name,
                    "path": str(resume_state.cache_path),
                    "status": resume_state.status,
                    "completed_sentences": resume_completed_prefix,
                    "total_sentences": len(sentences),
                }
            )
            emit({"type": "asr_output", "file": audio_file.name, "text": _build_asr_output_lines(sentences)})
        else:
            if resume_reason != "missing":
                emit(
                    {
                        "type": "json_cache_resume_mismatch",
                        "file": audio_file.name,
                        "path": str(json_path),
                        "reason": resume_reason,
                    }
                )

            asr_started_at = time.perf_counter()
            run_output, asr_err = _run_asr_with_retry(audio_file, asr_engine, cfg.pipeline.retry_asr)
            asr_sec = max(0.0, time.perf_counter() - asr_started_at)

            if run_output is None:
                write_started_at = time.perf_counter()
                out = _write_json_snapshot(
                    output_dir=output_dir,
                    base_name=base_name,
                    audio_path=str(audio_file),
                    duration_sec=0.0,
                    sentences=[],
                    status="failed",
                    error=asr_err,
                    cfg=cfg,
                    source_meta=source_meta,
                    batch_size=batch_size,
                    asr_sec=asr_sec,
                    translate_sec=translate_sec,
                    write_sec=write_sec,
                )
                write_sec += max(0.0, time.perf_counter() - write_started_at)
                emit(
                    {
                        "type": "json_cache_final_written",
                        "file": audio_file.name,
                        "path": str(out),
                        "status": "failed",
                    }
                )
                logging.error("ASR 失败，已写入失败 JSON：%s", out)
                failed += 1
                emit_file_stats("failed")
                emit(
                    {
                        "type": "file_end",
                        "file_index": idx,
                        "total_files": total,
                        "file": audio_file.name,
                        "status": "failed",
                        "output": str(out),
                        "error": asr_err or "",
                    }
                )
                continue

            duration_sec = float(run_output.duration_sec)
            sentences = run_output.sentences
            emit({"type": "asr_output", "file": audio_file.name, "text": _build_asr_output_lines(sentences)})

            write_started_at = time.perf_counter()
            out = _write_json_snapshot(
                output_dir=output_dir,
                base_name=base_name,
                audio_path=str(audio_file),
                duration_sec=duration_sec,
                sentences=sentences,
                status="asr_done",
                error=None,
                cfg=cfg,
                source_meta=source_meta,
                batch_size=batch_size,
                asr_sec=asr_sec,
                translate_sec=translate_sec,
                write_sec=write_sec,
            )
            write_sec += max(0.0, time.perf_counter() - write_started_at)
            emit(
                {
                    "type": "json_cache_init_written",
                    "file": audio_file.name,
                    "path": str(out),
                }
            )

        if cancel_token is not None and cancel_token.is_cancelled():
            err_msg = "用户取消处理"
            _mark_unfinished_failed(sentences)
            write_started_at = time.perf_counter()
            out = _write_json_snapshot(
                output_dir=output_dir,
                base_name=base_name,
                audio_path=str(audio_file),
                duration_sec=duration_sec,
                sentences=sentences,
                status="cancelled",
                error=err_msg,
                cfg=cfg,
                source_meta=source_meta,
                batch_size=batch_size,
                asr_sec=asr_sec,
                translate_sec=translate_sec,
                write_sec=write_sec,
            )
            write_sec += max(0.0, time.perf_counter() - write_started_at)
            emit(
                {
                    "type": "json_cache_final_written",
                    "file": audio_file.name,
                    "path": str(out),
                    "status": "cancelled",
                }
            )
            failed += 1
            emit_file_stats("failed")
            emit(
                {
                    "type": "file_end",
                    "file_index": idx,
                    "total_files": total,
                    "file": audio_file.name,
                    "status": "failed",
                    "output": str(out),
                    "error": err_msg,
                }
            )
            emit(
                {
                    "type": "batch_end",
                    "ok": ok,
                    "failed": failed,
                    "total_files": total,
                    "cancelled": True,
                    "rc": 2,
                }
            )
            return 2

        remaining = sentences[resume_completed_prefix:]
        total_groups_all = _total_groups(len(sentences), batch_size)
        group_offset = resume_completed_prefix // batch_size

        try:
            if remaining:
                translator = build_translator(cfg.translation)
                tool_ctx = TranslationToolContext(
                    audio_path=str(audio_file),
                    asr_config=cfg.asr,
                    sentences=[
                        SentenceRef(
                            sentence_id=s.sentence_id,
                            start_sec=s.start_sec,
                            end_sec=s.end_sec,
                            ja_text=s.ja_text,
                        )
                        for s in remaining
                    ],
                    shared_model=getattr(asr_engine, "model", None),
                )

                def translate_event_proxy(raw_event: dict[str, Any]) -> None:
                    nonlocal write_sec
                    mapped = _remap_translation_event(
                        raw_event,
                        sentence_offset=resume_completed_prefix,
                        group_offset=group_offset,
                        total_groups_all=total_groups_all,
                    )
                    etype = str(mapped.get("type", "")).strip()
                    if etype == "translation_group_result":
                        translations = mapped.get("translations")
                        if isinstance(translations, list):
                            start_index = _safe_int(mapped.get("group_start", 0), 0)
                            for i, text in enumerate(translations):
                                global_idx = start_index + i
                                if global_idx < 0 or global_idx >= len(sentences):
                                    continue
                                sentences[global_idx].zh_text = str(text)
                                sentences[global_idx].translation_status = "ok"
                            write_started_at = time.perf_counter()
                            out_path = _write_json_snapshot(
                                output_dir=output_dir,
                                base_name=base_name,
                                audio_path=str(audio_file),
                                duration_sec=duration_sec,
                                sentences=sentences,
                                status="translating",
                                error=None,
                                cfg=cfg,
                                source_meta=source_meta,
                                batch_size=batch_size,
                                asr_sec=asr_sec,
                                translate_sec=translate_sec,
                                write_sec=write_sec,
                            )
                            write_sec += max(0.0, time.perf_counter() - write_started_at)
                            emit(
                                {
                                    "type": "json_cache_group_written",
                                    "file": audio_file.name,
                                    "path": str(out_path),
                                    "translated_sentences": _count_completed_prefix(sentences),
                                    "total_sentences": len(sentences),
                                }
                            )
                    emit(mapped)

                ja_lines = [s.ja_text for s in remaining]
                translate_started_at = time.perf_counter()
                zh_lines = translator.translate_batch(
                    ja_lines,
                    src="ja",
                    tgt=cfg.translation.target,
                    retry=cfg.pipeline.retry_translate,
                    event_cb=translate_event_proxy,
                    cancel_token=cancel_token,
                    tool_ctx=tool_ctx,
                )
                translate_sec += max(0.0, time.perf_counter() - translate_started_at)

                if len(zh_lines) != len(remaining):
                    raise RuntimeError("翻译结果数量与句子数量不一致")

                for i, zh in enumerate(zh_lines):
                    global_idx = resume_completed_prefix + i
                    sentences[global_idx].zh_text = zh
                    sentences[global_idx].translation_status = "ok"

            final_status = "ok"
            write_started_at = time.perf_counter()
            out = _write_json_snapshot(
                output_dir=output_dir,
                base_name=base_name,
                audio_path=str(audio_file),
                duration_sec=duration_sec,
                sentences=sentences,
                status=final_status,
                error=None,
                cfg=cfg,
                source_meta=source_meta,
                batch_size=batch_size,
                asr_sec=asr_sec,
                translate_sec=translate_sec,
                write_sec=write_sec,
            )
            write_sec += max(0.0, time.perf_counter() - write_started_at)
            emit(
                {
                    "type": "json_cache_final_written",
                    "file": audio_file.name,
                    "path": str(out),
                    "status": final_status,
                }
            )
        except Exception as e:
            err_msg = f"翻译失败，批处理终止：{e}"
            logging.error(err_msg)
            _mark_unfinished_failed(sentences)
            status = "cancelled" if bool(cancel_token and cancel_token.is_cancelled()) else "failed"
            write_started_at = time.perf_counter()
            out = _write_json_snapshot(
                output_dir=output_dir,
                base_name=base_name,
                audio_path=str(audio_file),
                duration_sec=duration_sec,
                sentences=sentences,
                status=status,
                error=err_msg,
                cfg=cfg,
                source_meta=source_meta,
                batch_size=batch_size,
                asr_sec=asr_sec,
                translate_sec=translate_sec,
                write_sec=write_sec,
            )
            write_sec += max(0.0, time.perf_counter() - write_started_at)
            emit(
                {
                    "type": "json_cache_final_written",
                    "file": audio_file.name,
                    "path": str(out),
                    "status": status,
                }
            )
            logging.error("已写入失败 JSON：%s", out)
            logging.error("由于翻译错误，批处理提前终止。")
            failed += 1
            emit_file_stats("failed")
            emit(
                {
                    "type": "file_end",
                    "file_index": idx,
                    "total_files": total,
                    "file": audio_file.name,
                    "status": "failed",
                    "output": str(out),
                    "error": err_msg,
                }
            )
            emit(
                {
                    "type": "batch_end",
                    "ok": ok,
                    "failed": failed,
                    "total_files": total,
                    "cancelled": bool(cancel_token and cancel_token.is_cancelled()),
                    "rc": 2,
                }
            )
            return 2

        lrc_out: Path | None = None
        if cfg.output.write_lrc:
            try:
                write_started_at = time.perf_counter()
                lrc_out = write_lrc(output_dir=output_dir, base_name=base_name, sentences=sentences)
                write_sec += max(0.0, time.perf_counter() - write_started_at)
            except Exception as e:
                err_msg = f"LRC 写入失败：{e}"
                logging.error(err_msg)
                write_started_at = time.perf_counter()
                out = _write_json_snapshot(
                    output_dir=output_dir,
                    base_name=base_name,
                    audio_path=str(audio_file),
                    duration_sec=duration_sec,
                    sentences=sentences,
                    status="failed",
                    error=err_msg,
                    cfg=cfg,
                    source_meta=source_meta,
                    batch_size=batch_size,
                    asr_sec=asr_sec,
                    translate_sec=translate_sec,
                    write_sec=write_sec,
                )
                write_sec += max(0.0, time.perf_counter() - write_started_at)
                emit(
                    {
                        "type": "json_cache_final_written",
                        "file": audio_file.name,
                        "path": str(out),
                        "status": "failed",
                    }
                )
                emit_file_stats("failed")
                emit(
                    {
                        "type": "file_end",
                        "file_index": idx,
                        "total_files": total,
                        "file": audio_file.name,
                        "status": "failed",
                        "output": str(out),
                        "lrc_output": "",
                        "error": err_msg,
                    }
                )
                failed += 1
                continue

        ok += 1
        logging.info("处理完成：%s", out)
        emit_file_stats("ok")
        emit(
            {
                "type": "file_end",
                "file_index": idx,
                "total_files": total,
                "file": audio_file.name,
                "status": "ok",
                "output": str(out),
                "lrc_output": str(lrc_out) if lrc_out is not None else "",
            }
        )

    logging.info("批处理结束 | 成功=%d 失败=%d 总数=%d", ok, failed, total)
    emit(
        {
            "type": "batch_end",
            "ok": ok,
            "failed": failed,
            "total_files": total,
            "cancelled": bool(cancel_token and cancel_token.is_cancelled()),
            "rc": 0 if failed == 0 else 2,
        }
    )
    return 0 if failed == 0 else 2
