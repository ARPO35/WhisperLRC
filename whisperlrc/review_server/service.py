from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from whisperlrc.output.lrc_writer import format_lrc_time


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


class ReviewService:
    MIN_SENTENCE_SEC = 0.05
    DEFAULT_INSERT_MIN_DURATION_SEC = 0.5

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _iter_task_paths(self) -> list[Path]:
        paths = [p for p in self.output_dir.glob("*.json") if p.is_file()]
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return paths

    def _resolve_task_path(self, task_id: str) -> Path:
        name = Path(task_id).name
        if not name.lower().endswith(".json"):
            raise ValueError("task_id 必须是 .json 文件名")
        path = (self.output_dir / name).resolve()
        if path.parent != self.output_dir:
            raise ValueError("非法 task_id")
        if not path.exists():
            raise FileNotFoundError(f"任务不存在：{task_id}")
        return path

    def _load_task(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"JSON 读取失败：{e}") from e
        if not isinstance(data, dict):
            raise ValueError("JSON 顶层必须是对象")
        return data

    def _save_task(self, path: Path, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        _atomic_write_json(path, payload)

    def _next_sentence_id(self, sentences: list[dict[str, Any]]) -> str:
        max_id = 0
        for s in sentences:
            sid = str(s.get("sentence_id") or "")
            m = re.match(r"^s_(\d+)$", sid)
            if not m:
                continue
            n = _safe_int(m.group(1), 0)
            if n > max_id:
                max_id = n
        return f"s_{max_id + 1:04d}"

    def _clamp_insert_min_duration(self, value: float | None) -> float:
        if value is None:
            return self.DEFAULT_INSERT_MIN_DURATION_SEC
        v = _safe_float(value, self.DEFAULT_INSERT_MIN_DURATION_SEC)
        return max(self.MIN_SENTENCE_SEC, v)

    def _audio_duration_hint(self, payload: dict[str, Any]) -> float | None:
        source_meta = payload.get("source_meta")
        if isinstance(source_meta, dict):
            for k in ("duration_sec", "audio_duration_sec", "duration"):
                if k in source_meta:
                    v = _safe_float(source_meta.get(k), 0.0)
                    if v > 0:
                        return v
        for k in ("duration_sec", "audio_duration_sec", "duration"):
            if k in payload:
                v = _safe_float(payload.get(k), 0.0)
                if v > 0:
                    return v
        return None

    def _mark_in_review_status(self, payload: dict[str, Any]) -> None:
        status = str(payload.get("status") or "").strip()
        if status in {"todo", "asr_done", "ok"}:
            payload["status"] = "in_review"

    def _build_task_summary(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        sentences = payload.get("sentences")
        if not isinstance(sentences, list):
            sentences = []

        reviewed = 0
        for s in sentences:
            if not isinstance(s, dict):
                continue
            review_zh = str(s.get("review_zh_text") or s.get("review_text") or "").strip()
            review_ja = str(s.get("review_ja_text") or "").strip()
            if review_zh or review_ja:
                reviewed += 1

        quality = payload.get("quality")
        risk_count = len(quality) if isinstance(quality, list) else 0

        progress_obj = payload.get("progress")
        progress = progress_obj if isinstance(progress_obj, dict) else {}

        return {
            "task_id": path.name,
            "file_id": str(payload.get("file_id") or path.stem),
            "status": str(payload.get("status") or "todo"),
            "audio_path": str(payload.get("audio_path") or ""),
            "total_sentences": len(sentences),
            "reviewed_sentences": reviewed,
            "risk_count": risk_count,
            "updated_at": str(payload.get("updated_at") or payload.get("created_at") or ""),
            "progress": {
                "translated_sentences": _safe_int(progress.get("translated_sentences"), 0),
                "total_sentences": _safe_int(progress.get("total_sentences"), len(sentences)),
                "completed_groups": _safe_int(progress.get("completed_groups"), 0),
                "total_groups": _safe_int(progress.get("total_groups"), 0),
            },
        }

    def list_tasks(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in self._iter_task_paths():
            try:
                payload = self._load_task(p)
                out.append(self._build_task_summary(p, payload))
            except Exception as e:
                out.append(
                    {
                        "task_id": p.name,
                        "file_id": p.stem,
                        "status": "failed",
                        "audio_path": "",
                        "total_sentences": 0,
                        "reviewed_sentences": 0,
                        "risk_count": 0,
                        "updated_at": "",
                        "progress": {"translated_sentences": 0, "total_sentences": 0, "completed_groups": 0, "total_groups": 0},
                        "error": str(e),
                    }
                )
        return out

    def get_task(self, task_id: str) -> dict[str, Any]:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        sentences_obj = payload.get("sentences")
        if not isinstance(sentences_obj, list):
            sentences_obj = []

        normalized_sentences: list[dict[str, Any]] = []
        for idx, s in enumerate(sentences_obj, start=1):
            if not isinstance(s, dict):
                continue
            sid = str(s.get("sentence_id") or f"s_{idx:04d}")
            normalized_sentences.append(
                {
                    "sentence_id": sid,
                    "start_sec": _safe_float(s.get("start_sec"), 0.0),
                    "end_sec": _safe_float(s.get("end_sec"), 0.0),
                    "ja_text": str(s.get("ja_text") or ""),
                    "zh_text": str(s.get("zh_text") or ""),
                    "review_text_ja": str(s.get("review_ja_text") or ""),
                    "review_text_zh": str(s.get("review_zh_text") or s.get("review_text") or ""),
                    "review_state": str(s.get("review_state") or "pending"),
                    "translation_status": str(s.get("translation_status") or ""),
                    "word_items": [
                        {
                            "word": str(w.get("word") or ""),
                            "start_sec": _safe_float(w.get("start_sec"), 0.0) if w.get("start_sec") is not None else None,
                            "end_sec": _safe_float(w.get("end_sec"), 0.0) if w.get("end_sec") is not None else None,
                            "confidence": _safe_float(w.get("confidence"), 0.0) if w.get("confidence") is not None else None,
                        }
                        for w in (s.get("word_items") if isinstance(s.get("word_items"), list) else [])
                        if isinstance(w, dict)
                    ],
                }
            )

        task = self._build_task_summary(path, payload)
        task["sentences"] = normalized_sentences
        task["path"] = str(path)
        task["source_meta"] = payload.get("source_meta", {})
        task["logs"] = payload.get("logs", [])
        return task

    def update_sentence(
        self,
        task_id: str,
        sentence_id: str,
        *,
        review_text_ja: str | None = None,
        review_text_zh: str | None = None,
        review_state: str | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> dict[str, Any]:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        sentences = payload.get("sentences")
        if not isinstance(sentences, list):
            raise ValueError("任务不包含 sentences 数组")

        target: dict[str, Any] | None = None
        for s in sentences:
            if isinstance(s, dict) and str(s.get("sentence_id") or "") == sentence_id:
                target = s
                break
        if target is None:
            raise FileNotFoundError(f"句子不存在：{sentence_id}")

        if review_text_ja is not None:
            target["review_ja_text"] = review_text_ja
        if review_text_zh is not None:
            target["review_zh_text"] = review_text_zh
            target["review_text"] = review_text_zh
        if review_state is not None:
            target["review_state"] = review_state
        if start_sec is not None:
            target["start_sec"] = float(start_sec)
        if end_sec is not None:
            target["end_sec"] = float(end_sec)

        cur_start = _safe_float(target.get("start_sec"), 0.0)
        cur_end = _safe_float(target.get("end_sec"), 0.0)
        if cur_start < 0:
            raise ValueError("start_sec 不能小于 0")
        if cur_end <= cur_start:
            raise ValueError("end_sec 必须大于 start_sec")

        review = payload.get("review")
        if not isinstance(review, list):
            review = []
            payload["review"] = review

        found = None
        for item in review:
            if isinstance(item, dict) and str(item.get("sentence_id") or "") == sentence_id:
                found = item
                break
        if found is None:
            found = {"sentence_id": sentence_id}
            review.append(found)

        if review_text_ja is not None:
            found["edited_ja"] = review_text_ja
        if review_text_zh is not None:
            found["edited_zh"] = review_text_zh
        if review_state is not None:
            found["review_state"] = review_state
        found["reviewed_at"] = _now_iso()

        self._mark_in_review_status(payload)

        self._save_task(path, payload)
        return self.get_task(task_id)

    def insert_sentence_after(
        self,
        task_id: str,
        sentence_id: str,
        *,
        min_duration_sec: float | None = None,
    ) -> dict[str, Any]:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        sentences_obj = payload.get("sentences")
        if not isinstance(sentences_obj, list):
            raise ValueError("任务不包含 sentences 数组")

        sentences = [s for s in sentences_obj if isinstance(s, dict)]
        if not sentences:
            raise ValueError("任务中没有可插入的句子")

        idx = -1
        for i, s in enumerate(sentences):
            if str(s.get("sentence_id") or "") == sentence_id:
                idx = i
                break
        if idx < 0:
            raise FileNotFoundError(f"句子不存在：{sentence_id}")

        cur = sentences[idx]
        next_item = sentences[idx + 1] if idx + 1 < len(sentences) else None
        min_dur = self._clamp_insert_min_duration(min_duration_sec)
        cur_start = _safe_float(cur.get("start_sec"), 0.0)
        cur_end = _safe_float(cur.get("end_sec"), cur_start + self.MIN_SENTENCE_SEC)
        cur_end = max(cur_end, cur_start + self.MIN_SENTENCE_SEC)

        if next_item is not None:
            next_start = _safe_float(next_item.get("start_sec"), cur_end)
            gap = next_start - cur_end
            if gap >= min_dur:
                new_start = cur_end
                new_end = cur_end + min_dur
            elif gap >= self.MIN_SENTENCE_SEC:
                new_start = cur_end
                new_end = next_start
            else:
                cur_len = cur_end - cur_start
                max_shrink = max(0.0, cur_len - self.MIN_SENTENCE_SEC)
                required = self.MIN_SENTENCE_SEC - gap
                shrink = min(max_shrink, max(0.0, required))
                new_start = cur_end - shrink
                cur["end_sec"] = new_start
                available = next_start - new_start
                if available >= self.MIN_SENTENCE_SEC:
                    new_end = min(next_start, new_start + min_dur)
                    new_end = max(new_end, new_start + self.MIN_SENTENCE_SEC)
                else:
                    new_end = new_start + self.MIN_SENTENCE_SEC
        else:
            new_start = cur_end
            new_end = new_start + min_dur
            duration_hint = self._audio_duration_hint(payload)
            if duration_hint is not None and duration_hint > new_start + self.MIN_SENTENCE_SEC:
                new_end = min(new_end, duration_hint)
            new_end = max(new_end, new_start + self.MIN_SENTENCE_SEC)

        new_sentence = {
            "sentence_id": self._next_sentence_id(sentences),
            "start_sec": float(max(0.0, new_start)),
            "end_sec": float(max(new_end, new_start + self.MIN_SENTENCE_SEC)),
            "ja_text": "",
            "zh_text": "",
            "review_ja_text": "",
            "review_zh_text": "",
            "review_text": "",
            "review_state": "pending",
            "translation_status": "",
        }

        sentences.insert(idx + 1, new_sentence)
        payload["sentences"] = sentences
        self._mark_in_review_status(payload)
        self._save_task(path, payload)
        return self.get_task(task_id)

    def delete_sentence(self, task_id: str, sentence_id: str) -> dict[str, Any]:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        sentences_obj = payload.get("sentences")
        if not isinstance(sentences_obj, list):
            raise ValueError("任务不包含 sentences 数组")

        sentences = [s for s in sentences_obj if isinstance(s, dict)]
        if len(sentences) <= 1:
            raise ValueError("至少保留一条字幕，不能删除最后一条")

        idx = -1
        for i, s in enumerate(sentences):
            if str(s.get("sentence_id") or "") == sentence_id:
                idx = i
                break
        if idx < 0:
            raise FileNotFoundError(f"句子不存在：{sentence_id}")

        sentences.pop(idx)
        if 0 < idx < len(sentences):
            prev_item = sentences[idx - 1]
            next_item = sentences[idx]
            prev_start = _safe_float(prev_item.get("start_sec"), 0.0)
            next_start = _safe_float(next_item.get("start_sec"), prev_start + self.MIN_SENTENCE_SEC)
            prev_item["end_sec"] = max(next_start, prev_start + self.MIN_SENTENCE_SEC)

        review_obj = payload.get("review")
        if isinstance(review_obj, list):
            payload["review"] = [
                item for item in review_obj if not (isinstance(item, dict) and str(item.get("sentence_id") or "") == sentence_id)
            ]

        payload["sentences"] = sentences
        self._mark_in_review_status(payload)
        self._save_task(path, payload)
        return self.get_task(task_id)

    def update_task_status(self, task_id: str, status: str) -> dict[str, Any]:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        payload["status"] = status
        self._save_task(path, payload)
        return self.get_task(task_id)

    def _select_ja_text(self, sentence: dict[str, Any]) -> str:
        review_ja = str(sentence.get("review_ja_text") or "").strip()
        if review_ja:
            return review_ja
        return str(sentence.get("ja_text") or "").strip()

    def _select_zh_text(self, sentence: dict[str, Any]) -> str:
        review_zh = str(sentence.get("review_zh_text") or sentence.get("review_text") or "").strip()
        if review_zh:
            return review_zh
        return str(sentence.get("zh_text") or "").strip()

    def _build_lrc_lines(self, sentences: list[dict[str, Any]], selector: str) -> tuple[list[str], list[str]]:
        lines: list[str] = []
        warnings: list[str] = []
        for idx, s in enumerate(sentences, start=1):
            start_sec = _safe_float(s.get("start_sec"), 0.0)
            text = self._select_ja_text(s) if selector == "ja" else self._select_zh_text(s)
            if not text:
                warnings.append(f"第 {idx} 句文本为空")
            lines.append(f"{format_lrc_time(start_sec)}{text}")
        return lines, warnings

    def export_task(
        self,
        task_id: str,
        *,
        export_ja: bool = True,
        export_zh: bool = True,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        sentences_obj = payload.get("sentences")
        if not isinstance(sentences_obj, list):
            raise ValueError("任务不包含 sentences 数组")
        sentences = [s for s in sentences_obj if isinstance(s, dict)]

        base_name = path.stem
        out_dir = Path(output_dir).resolve() if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        exported: list[str] = []

        if export_ja:
            ja_lines, ja_warnings = self._build_lrc_lines(sentences, "ja")
            warnings.extend([f"JA: {w}" for w in ja_warnings])
            ja_path = out_dir / f"{base_name}.final.ja.lrc"
            _atomic_write_text(ja_path, ("\n".join(ja_lines).strip() + "\n"))
            exported.append(str(ja_path))

        if export_zh:
            zh_lines, zh_warnings = self._build_lrc_lines(sentences, "zh")
            warnings.extend([f"ZH: {w}" for w in zh_warnings])
            zh_path = out_dir / f"{base_name}.final.zh.lrc"
            _atomic_write_text(zh_path, ("\n".join(zh_lines).strip() + "\n"))
            exported.append(str(zh_path))

        return {
            "task_id": task_id,
            "exported_files": exported,
            "warnings": warnings,
        }

    def resolve_audio_path(self, task_id: str) -> Path:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        raw = str(payload.get("audio_path") or "").strip()
        if not raw:
            raise FileNotFoundError("任务中缺少 audio_path")
        audio_path = Path(raw)
        if not audio_path.is_absolute():
            audio_path = (Path.cwd() / audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在：{audio_path}")
        return audio_path
