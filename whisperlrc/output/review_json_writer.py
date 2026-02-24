from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from whisperlrc.config import AppConfig
from whisperlrc.types import FileProcessResult, SentenceItem


def _build_json_path(output_dir: Path, basename: str, ext: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{basename}{ext}"


def _sentence_to_dict(sentence: SentenceItem) -> dict[str, Any]:
    # Avoid deep-copying token_items (can be very large and is not exported).
    return {
        "sentence_id": sentence.sentence_id,
        "start_sec": sentence.start_sec,
        "end_sec": sentence.end_sec,
        "ja_text": sentence.ja_text,
        "zh_text": sentence.zh_text,
        "translation_status": sentence.translation_status,
        "segment_confidence": sentence.segment_confidence,
        "review_text": sentence.review_text,
        "word_items": [
            {
                "word": w.word,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "confidence": w.confidence,
            }
            for w in sentence.word_items
        ],
    }


def write_review_json(
    *,
    output_dir: Path,
    base_name: str,
    audio_path: str,
    duration_sec: float,
    result: FileProcessResult,
    cfg: AppConfig,
    progress: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    source_meta: dict[str, Any] | None = None,
) -> Path:
    out_path = _build_json_path(output_dir, base_name, cfg.output.json_ext)
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": cfg.schema.version,
        "file_id": base_name,
        "audio_path": audio_path,
        "duration_sec": duration_sec,
        "status": result.status,
        "created_at": now,
        "updated_at": now,
        "settings_snapshot": cfg.to_dict(),
        "sentences": [_sentence_to_dict(s) for s in result.sentences],
        "logs": result.logs + ([result.error] if result.error else []),
    }
    if isinstance(progress, dict):
        payload["progress"] = progress
    if isinstance(runtime, dict):
        payload["runtime"] = runtime
    if isinstance(source_meta, dict):
        payload["source_meta"] = source_meta

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path
