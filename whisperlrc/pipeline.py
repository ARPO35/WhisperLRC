from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from whisperlrc.config import AppConfig
from whisperlrc.output.review_json_writer import write_review_json
from whisperlrc.translate.factory import build_translator
from whisperlrc.types import FileProcessResult, SentenceItem


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
) -> tuple[list[SentenceItem], str | None]:
    translator = build_translator(cfg.translation)
    ja_lines = [s.ja_text for s in sentences]
    zh_lines = translator.translate_batch(ja_lines, src="ja", tgt=cfg.translation.target)
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


def _run_translation_with_retry(sentences: list[SentenceItem], cfg: AppConfig, retry: int) -> tuple[list[SentenceItem], str | None]:
    max_attempts = retry + 1
    err: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _translate_sentences(sentences, cfg)
        except NotImplementedError as e:
            err = f"翻译已跳过（功能未实现）：{e}"
            logging.warning(err)
            for s in sentences:
                s.zh_text = None
                s.translation_status = "skipped"
            return sentences, err
        except Exception as e:
            err = f"翻译第 {attempt}/{max_attempts} 次尝试失败：{e}"
            logging.error(err)
    for s in sentences:
        s.zh_text = None
        s.translation_status = "failed"
    return sentences, err


def process_batch(input_dir: Path, output_dir: Path, cfg: AppConfig) -> int:
    audio_files = _iter_audio_files(input_dir, cfg.pipeline.input_formats)
    if not audio_files:
        logging.warning("在目录 %s 中未找到可处理的音频文件", input_dir)
        return 0

    from whisperlrc.asr.faster_whisper_engine import FasterWhisperEngine

    asr_engine = FasterWhisperEngine(cfg.asr)
    ok = 0
    partial = 0
    failed = 0
    for audio_file in audio_files:
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
            continue

        sentences, tr_err = _run_translation_with_retry(run_output.sentences, cfg, cfg.pipeline.retry_translate)
        status = "ok" if tr_err is None else "partial"
        result = FileProcessResult(status=status, sentences=sentences, error=tr_err, logs=[])
        out = write_review_json(
            output_dir=output_dir,
            base_name=audio_file.stem,
            audio_path=str(audio_file),
            duration_sec=run_output.duration_sec,
            result=result,
            cfg=cfg,
        )
        if status == "ok":
            ok += 1
            logging.info("处理完成：%s", out)
        else:
            partial += 1
            logging.warning("部分完成（翻译不可用或失败）：%s", out)

    logging.info("批处理结束 | 成功=%d 部分成功=%d 失败=%d 总数=%d", ok, partial, failed, len(audio_files))
    return 0 if failed == 0 else 2
