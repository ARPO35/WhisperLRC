from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from whisperlrc.config import load_config
from whisperlrc.output.lrc_writer import format_lrc_time
from whisperlrc.translate.factory import build_translator
from whisperlrc.translate.tooling import SentenceRef, TranslationToolContext


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
    _relisten_model_cache_lock = threading.Lock()
    _relisten_model_cache: dict[tuple[str, str, str], Any] = {}

    def __init__(self, output_dir: Path, *, config_path: Path | None = None) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config_path is not None:
            self.config_path = config_path.resolve()
        else:
            self.config_path = (Path.cwd() / "settings.toml").resolve()

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
                    "segment_confidence": _safe_float(s.get("segment_confidence"), 0.0)
                    if s.get("segment_confidence") is not None
                    else None,
                    "word_items": [
                        {
                            "word": str(
                                w.get("word")
                                or w.get("text")
                                or w.get("token")
                                or w.get("token_text")
                                or ""
                            ),
                            "start_sec": _safe_float(
                                w.get("start_sec", w.get("start")),
                                0.0,
                            )
                            if (w.get("start_sec") is not None or w.get("start") is not None)
                            else None,
                            "end_sec": _safe_float(
                                w.get("end_sec", w.get("end")),
                                0.0,
                            )
                            if (w.get("end_sec") is not None or w.get("end") is not None)
                            else None,
                            "confidence": _safe_float(
                                w.get("confidence", w.get("conf", w.get("probability", w.get("prob")))),
                                0.0,
                            )
                            if (
                                w.get("confidence") is not None
                                or w.get("conf") is not None
                                or w.get("probability") is not None
                                or w.get("prob") is not None
                            )
                            else None,
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

    def _normalize_word_items(self, items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for w in items:
            if not isinstance(w, dict):
                continue
            text = str(w.get("word") or w.get("text") or w.get("token") or w.get("token_text") or "")
            if not text:
                continue
            start_val = w.get("start_sec", w.get("start"))
            end_val = w.get("end_sec", w.get("end"))
            conf_val = w.get("confidence", w.get("conf", w.get("probability", w.get("prob"))))
            out.append(
                {
                    "word": text,
                    "start_sec": _safe_float(start_val, 0.0) if start_val is not None else None,
                    "end_sec": _safe_float(end_val, 0.0) if end_val is not None else None,
                    "confidence": _safe_float(conf_val, 0.0) if conf_val is not None else None,
                }
            )
        return out

    def _sync_review_from_sentences(self, payload: dict[str, Any], sentences: list[dict[str, Any]]) -> None:
        old_review_obj = payload.get("review")
        old_review = old_review_obj if isinstance(old_review_obj, list) else []
        old_map: dict[str, dict[str, Any]] = {}
        for item in old_review:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("sentence_id") or "")
            if sid:
                old_map[sid] = item

        out: list[dict[str, Any]] = []
        now = _now_iso()
        for s in sentences:
            sid = str(s.get("sentence_id") or "")
            if not sid:
                continue
            edited_ja = str(s.get("review_ja_text") or "").strip()
            edited_zh = str(s.get("review_zh_text") or s.get("review_text") or "").strip()
            review_state = str(s.get("review_state") or "pending").strip() or "pending"
            if not edited_ja and not edited_zh and review_state == "pending":
                continue
            old = old_map.get(sid, {})
            out.append(
                {
                    "sentence_id": sid,
                    "edited_ja": edited_ja,
                    "edited_zh": edited_zh,
                    "review_state": review_state,
                    "reviewed_at": str(old.get("reviewed_at") or now),
                }
            )
        payload["review"] = out

    def _compose_sentence_from_draft(
        self,
        draft: dict[str, Any],
        *,
        base: dict[str, Any] | None,
        index: int,
    ) -> dict[str, Any]:
        sid = str(draft.get("sentence_id") or "").strip()
        if not sid:
            raise ValueError(f"第 {index} 句缺少 sentence_id")

        base_obj = dict(base) if isinstance(base, dict) else {}
        start_sec = _safe_float(draft.get("start_sec"), _safe_float(base_obj.get("start_sec"), 0.0))
        end_sec = _safe_float(draft.get("end_sec"), _safe_float(base_obj.get("end_sec"), start_sec + self.MIN_SENTENCE_SEC))
        if start_sec < 0:
            raise ValueError(f"第 {index} 句 start_sec 不能小于 0")
        if end_sec <= start_sec:
            raise ValueError(f"第 {index} 句 end_sec 必须大于 start_sec")

        ja_text = str(draft.get("ja_text") if draft.get("ja_text") is not None else base_obj.get("ja_text") or "")
        zh_text = str(draft.get("zh_text") if draft.get("zh_text") is not None else base_obj.get("zh_text") or "")
        review_text_ja = draft.get("review_text_ja", draft.get("review_ja_text"))
        review_text_zh = draft.get("review_text_zh", draft.get("review_zh_text", draft.get("review_text")))
        if review_text_ja is None:
            review_text_ja = base_obj.get("review_ja_text", "")
        if review_text_zh is None:
            review_text_zh = base_obj.get("review_zh_text", base_obj.get("review_text", ""))

        review_state = draft.get("review_state")
        if review_state is None:
            review_state = base_obj.get("review_state", "pending")

        translation_status = draft.get("translation_status")
        if translation_status is None:
            translation_status = base_obj.get("translation_status")
        if translation_status is None:
            translation_status = "ok" if (str(review_text_zh).strip() or str(zh_text).strip()) else "pending"

        sentence = dict(base_obj)
        sentence.update(
            {
                "sentence_id": sid,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "ja_text": ja_text,
                "zh_text": zh_text,
                "review_ja_text": str(review_text_ja or ""),
                "review_zh_text": str(review_text_zh or ""),
                "review_text": str(review_text_zh or ""),
                "review_state": str(review_state or "pending"),
                "translation_status": str(translation_status or ""),
            }
        )
        if "segment_confidence" in draft:
            sentence["segment_confidence"] = (
                _safe_float(draft.get("segment_confidence"), 0.0) if draft.get("segment_confidence") is not None else None
            )
        if "word_items" in draft:
            sentence["word_items"] = self._normalize_word_items(draft.get("word_items"))
        elif isinstance(base_obj.get("word_items"), list):
            sentence["word_items"] = self._normalize_word_items(base_obj.get("word_items"))
        return sentence

    def save_task_snapshot(self, task_id: str, *, status: str, sentences: list[dict[str, Any]]) -> dict[str, Any]:
        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)

        raw_sentences = payload.get("sentences")
        base_list = raw_sentences if isinstance(raw_sentences, list) else []
        base_map: dict[str, dict[str, Any]] = {}
        for item in base_list:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("sentence_id") or "")
            if sid:
                base_map[sid] = item

        out_sentences: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx, draft in enumerate(sentences, start=1):
            if not isinstance(draft, dict):
                raise ValueError(f"第 {idx} 句不是对象")
            sid = str(draft.get("sentence_id") or "").strip()
            if not sid:
                raise ValueError(f"第 {idx} 句缺少 sentence_id")
            if sid in seen:
                raise ValueError(f"sentence_id 重复：{sid}")
            seen.add(sid)
            out_sentences.append(self._compose_sentence_from_draft(draft, base=base_map.get(sid), index=idx))

        payload["sentences"] = out_sentences
        payload["status"] = str(status or payload.get("status") or "in_review")
        self._sync_review_from_sentences(payload, out_sentences)
        self._save_task(path, payload)
        return self.get_task(task_id)

    def _build_sentence_map(self, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        sentences_obj = payload.get("sentences")
        if not isinstance(sentences_obj, list):
            raise ValueError("任务不包含 sentences 数组")
        sentences = [s for s in sentences_obj if isinstance(s, dict)]
        index_map: dict[str, int] = {}
        for idx, s in enumerate(sentences):
            sid = str(s.get("sentence_id") or "")
            if sid:
                index_map[sid] = idx
        return sentences, index_map

    def _build_draft_map(self, drafts: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not drafts:
            return out
        for d in drafts:
            if not isinstance(d, dict):
                continue
            sid = str(d.get("sentence_id") or "").strip()
            if not sid:
                continue
            out[sid] = d
        return out

    def _select_ja_from_sentence_and_draft(self, sentence: dict[str, Any], draft: dict[str, Any] | None) -> str:
        if isinstance(draft, dict):
            review_ja = str(draft.get("review_text_ja", draft.get("review_ja_text", "")) or "").strip()
            if review_ja:
                return review_ja
            ja = str(draft.get("ja_text") or "").strip()
            if ja:
                return ja
        return self._select_ja_text(sentence)

    def _build_sentence_refs(self, payload: dict[str, Any], *, draft_map: dict[str, dict[str, Any]] | None = None) -> list[SentenceRef]:
        sentences, _ = self._build_sentence_map(payload)
        refs: list[SentenceRef] = []
        for s in sentences:
            sid = str(s.get("sentence_id") or "")
            if not sid:
                continue
            draft = draft_map.get(sid) if isinstance(draft_map, dict) else None
            if isinstance(draft, dict):
                start_sec = _safe_float(draft.get("start_sec"), _safe_float(s.get("start_sec"), 0.0))
                end_sec = _safe_float(draft.get("end_sec"), _safe_float(s.get("end_sec"), start_sec + self.MIN_SENTENCE_SEC))
            else:
                start_sec = _safe_float(s.get("start_sec"), 0.0)
                end_sec = _safe_float(s.get("end_sec"), start_sec + self.MIN_SENTENCE_SEC)
            if end_sec <= start_sec:
                end_sec = start_sec + self.MIN_SENTENCE_SEC
            refs.append(
                SentenceRef(
                    sentence_id=sid,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    ja_text=self._select_ja_from_sentence_and_draft(s, draft),
                )
            )
        return refs

    def _load_runtime_config(self):
        try:
            return load_config(self.config_path)
        except Exception as e:
            raise ValueError(f"加载配置失败：{self.config_path} | {e}") from e

    def _get_shared_relisten_model(self, *, model: str, device: str, compute_type: str) -> Any:
        key = (str(model or ""), str(device or ""), str(compute_type or ""))
        with self._relisten_model_cache_lock:
            cached = self._relisten_model_cache.get(key)
            if cached is not None:
                return cached

        from faster_whisper import WhisperModel

        created = WhisperModel(
            model_size_or_path=key[0],
            device=key[1],
            compute_type=key[2],
        )
        with self._relisten_model_cache_lock:
            cached = self._relisten_model_cache.get(key)
            if cached is not None:
                return cached
            self._relisten_model_cache[key] = created
            return created

    def relisten_sentence_once(
        self,
        *,
        task_id: str,
        sentence_id: str,
        draft_sentence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from whisperlrc.translate.relisten_tool import RelistenOptions, WhisperRelistenExecutor

        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        draft_map = self._build_draft_map([draft_sentence] if isinstance(draft_sentence, dict) else [])
        refs = self._build_sentence_refs(payload, draft_map=draft_map)

        target_idx = -1
        for i, ref in enumerate(refs):
            if ref.sentence_id == sentence_id:
                target_idx = i
                break
        if target_idx < 0:
            raise FileNotFoundError(f"句子不存在：{sentence_id}")

        cfg = self._load_runtime_config()
        audio_path = self.resolve_audio_path(task_id)
        shared_model = self._get_shared_relisten_model(
            model=cfg.asr.model,
            device=cfg.asr.device,
            compute_type=cfg.asr.compute_type,
        )
        tool_ctx = TranslationToolContext(
            audio_path=str(audio_path),
            sentences=refs,
            asr_config=cfg.asr,
            shared_model=shared_model,
        )
        executor = WhisperRelistenExecutor(tool_ctx, RelistenOptions(candidate_count=1))
        result = executor.relisten_by_global_index(target_idx)
        if not bool(result.get("ok")):
            raise RuntimeError(str(result.get("error") or "重识别失败"))

        raw_candidates = result.get("candidates")
        candidates = [str(x) for x in raw_candidates] if isinstance(raw_candidates, list) else []
        ja_text = candidates[0].strip() if candidates else ""
        return {
            "ok": True,
            "sentence_id": sentence_id,
            "ja_text": ja_text,
            "candidates": candidates,
            "meta": {
                "index": result.get("index"),
                "clip_start_sec": result.get("clip_start_sec"),
                "clip_end_sec": result.get("clip_end_sec"),
                "context_prev": result.get("context_prev") if isinstance(result.get("context_prev"), list) else [],
                "context_next": result.get("context_next") if isinstance(result.get("context_next"), list) else [],
            },
        }

    def _auto_translate_texts(self, texts: list[str]) -> list[str]:
        cfg = self._load_runtime_config()
        cfg.translation.llm_batch_size = max(1, len(texts))
        translator = build_translator(cfg.translation)
        return translator.translate_batch(
            texts,
            src="ja",
            tgt=cfg.translation.target,
            retry=cfg.pipeline.retry_translate,
        )

    def auto_translate_sentence(
        self,
        *,
        task_id: str,
        sentence_id: str,
        draft_sentence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = self.auto_translate_sentences(
            task_id=task_id,
            sentence_ids=[sentence_id],
            draft_sentences=[draft_sentence] if isinstance(draft_sentence, dict) else [],
        )
        translations = data.get("translations") if isinstance(data, dict) else []
        first = translations[0] if isinstance(translations, list) and translations else {}
        return {
            "ok": bool(data.get("ok", False)) if isinstance(data, dict) else False,
            "sentence_id": sentence_id,
            "zh_text": str(first.get("zh_text") or ""),
            "meta": {
                "count": 1,
                "target": data.get("target") if isinstance(data, dict) else "",
            },
        }

    def auto_translate_sentences(
        self,
        *,
        task_id: str,
        sentence_ids: list[str],
        draft_sentences: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not sentence_ids:
            raise ValueError("sentence_ids 不能为空")

        path = self._resolve_task_path(task_id)
        payload = self._load_task(path)
        sentences, index_map = self._build_sentence_map(payload)
        draft_map = self._build_draft_map(draft_sentences)

        texts: list[str] = []
        cleaned_ids: list[str] = []
        for sid_raw in sentence_ids:
            sid = str(sid_raw).strip()
            if not sid:
                continue
            idx = index_map.get(sid)
            if idx is None:
                raise FileNotFoundError(f"句子不存在：{sid}")
            sentence = sentences[idx]
            draft = draft_map.get(sid)
            texts.append(self._select_ja_from_sentence_and_draft(sentence, draft))
            cleaned_ids.append(sid)

        if not cleaned_ids:
            raise ValueError("有效 sentence_ids 为空")

        translated = self._auto_translate_texts(texts)
        if len(translated) != len(cleaned_ids):
            raise RuntimeError("翻译返回数量与请求数量不一致")

        cfg = self._load_runtime_config()
        return {
            "ok": True,
            "sentence_ids": cleaned_ids,
            "translations": [{"sentence_id": sid, "zh_text": str(zh)} for sid, zh in zip(cleaned_ids, translated)],
            "target": cfg.translation.target,
        }

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

