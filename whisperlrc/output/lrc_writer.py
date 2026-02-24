from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _build_lrc_path(output_dir: Path, basename: str, ext: str = ".lrc") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{basename}{ext}"


def format_lrc_time(seconds: float) -> str:
    total_cs = int(round(max(0.0, seconds) * 100))
    minutes, rem_cs = divmod(total_cs, 6000)
    secs, cs = divmod(rem_cs, 100)
    return f"[{minutes:02d}:{secs:02d}.{cs:02d}]"


def _get_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_sentence_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return {
        "start_sec": _get_item_value(item, "start_sec", 0.0),
        "zh_text": _get_item_value(item, "zh_text", None),
        "review_text": _get_item_value(item, "review_text", None),
    }


def _select_export_text(sentence: dict[str, Any]) -> str:
    review_text = str(sentence.get("review_text") or "").strip()
    if review_text:
        return review_text
    zh_text = str(sentence.get("zh_text") or "").strip()
    if zh_text:
        return zh_text
    return ""


def build_lrc_lines(sentences: list[Any]) -> list[str]:
    lines: list[str] = []
    for raw in sentences:
        sentence = _to_sentence_dict(raw)
        start_sec = float(sentence.get("start_sec") or 0.0)
        text = _select_export_text(sentence)
        lines.append(f"{format_lrc_time(start_sec)}{text}")
    return lines


def write_lrc(*, output_dir: Path, base_name: str, sentences: list[Any]) -> Path:
    out_path = _build_lrc_path(output_dir, base_name, ".lrc")
    lrc_text = "\n".join(build_lrc_lines(sentences)).strip() + "\n"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(lrc_text, encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path
