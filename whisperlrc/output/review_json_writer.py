from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from whisperlrc.config import AppConfig
from whisperlrc.types import FileProcessResult, SentenceItem


def _ensure_unique_json_path(output_dir: Path, basename: str, ext: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{basename}{ext}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = output_dir / f"{basename}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _sentence_to_dict(sentence: SentenceItem) -> dict[str, Any]:
    return asdict(sentence)


def write_review_json(
    *,
    output_dir: Path,
    base_name: str,
    audio_path: str,
    duration_sec: float,
    result: FileProcessResult,
    cfg: AppConfig,
) -> Path:
    out_path = _ensure_unique_json_path(output_dir, base_name, cfg.output.json_ext)
    payload = {
        "schema_version": cfg.schema.version,
        "file_id": base_name,
        "audio_path": audio_path,
        "duration_sec": duration_sec,
        "status": result.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings_snapshot": cfg.to_dict(),
        "sentences": [_sentence_to_dict(s) for s in result.sentences],
        "logs": result.logs + ([result.error] if result.error else []),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
