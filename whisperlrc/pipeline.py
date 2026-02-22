from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from whisperlrc.config import AppConfig
from whisperlrc.output.lrc_writer import write_lrc
from whisperlrc.output.review_json_writer import write_review_json
from whisperlrc.translate.factory import build_translator
from whisperlrc.types import FileProcessResult, SentenceItem


class CancelToken:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled


def _iter_audio_files(input_dir: Path, exts: list[str]) -> list[Path]:
    ext_set = {e.lower().lstrip(".") for e in exts}
    out: list[Path] = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in ext_set:
            out.append(p)
    return sorted(out)


def _translate_sentences(
    sentences: list[SentenceItem],
    cfg: AppConfig,
    retry: int,
    event_cb: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancelToken | None = None,
) -> tuple[list[SentenceItem], str | None]:
    translator = build_translator(cfg.translation)
    ja_lines = [s.ja_text for s in sentences]
    zh_lines = translator.translate_batch(
        ja_lines,
        src="ja",
        tgt=cfg.translation.target,
        retry=retry,
        event_cb=event_cb,
        cancel_token=cancel_token,
    )
    if len(zh_lines) != len(sentences):
        return sentences, "翻译结果数量与句子数量不一致"
    for s, zh in zip(sentences, zh_lines):
        s.zh_text = zh
        s.translation_status = "ok"
    return sentences, None


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


def _mark_translation_failed(sentences: list[SentenceItem]) -> list[SentenceItem]:
    for s in sentences:
        s.zh_text = None
        s.translation_status = "failed"
    return sentences


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
    for idx, audio_file in enumerate(audio_files, start=1):
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
        run_output, asr_err = _run_asr_with_retry(audio_file, asr_engine, cfg.pipeline.retry_asr)
        if run_output is None:
            result = FileProcessResult(status="failed", sentences=[], error=asr_err, logs=[])
            out = write_review_json(
                output_dir=output_dir,
                base_name=audio_file.stem,
                audio_path=str(audio_file),
                duration_sec=0.0,
                result=result,
                cfg=cfg,
            )
            logging.error("ASR 失败，已写入失败 JSON：%s", out)
            failed += 1
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

        asr_lines: list[str] = []
        for sent in run_output.sentences:
            conf = "" if sent.segment_confidence is None else f" | conf={sent.segment_confidence:.4f}"
            asr_lines.append(f"[{sent.start_sec:.2f}-{sent.end_sec:.2f}] {sent.ja_text}{conf}")
        emit(
            {
                "type": "asr_output",
                "file": audio_file.name,
                "text": "\n".join(asr_lines),
            }
        )

        try:
            sentences, tr_err = _translate_sentences(
                run_output.sentences,
                cfg,
                cfg.pipeline.retry_translate,
                event_cb=event_cb,
                cancel_token=cancel_token,
            )
            if tr_err is not None:
                raise RuntimeError(tr_err)
        except Exception as e:
            err_msg = f"翻译失败，批处理终止：{e}"
            logging.error(err_msg)
            failed_sentences = _mark_translation_failed(run_output.sentences)
            result = FileProcessResult(status="failed", sentences=failed_sentences, error=err_msg, logs=[])
            out = write_review_json(
                output_dir=output_dir,
                base_name=audio_file.stem,
                audio_path=str(audio_file),
                duration_sec=run_output.duration_sec,
                result=result,
                cfg=cfg,
            )
            logging.error("已写入失败 JSON：%s", out)
            logging.error("由于翻译错误，批处理提前终止。")
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
                    "failed": failed + 1,
                    "total_files": total,
                    "cancelled": bool(cancel_token and cancel_token.is_cancelled()),
                    "rc": 2,
                }
            )
            return 2

        result = FileProcessResult(status="ok", sentences=sentences, error=tr_err, logs=[])
        out = write_review_json(
            output_dir=output_dir,
            base_name=audio_file.stem,
            audio_path=str(audio_file),
            duration_sec=run_output.duration_sec,
            result=result,
            cfg=cfg,
        )
        lrc_out: Path | None = None
        if cfg.output.write_lrc:
            try:
                lrc_out = write_lrc(output_dir=output_dir, base_name=audio_file.stem, sentences=sentences)
            except Exception as e:
                err_msg = f"LRC 写入失败：{e}"
                logging.error(err_msg)
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
