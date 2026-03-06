from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from whisperlrc.review_server.service import ReviewService


class JobCancelled(RuntimeError):
    pass


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str, str], None]
CancelCallback = Callable[[], bool]


def _noop_log(_message: str) -> None:
    return


def _noop_progress(_done: int, _total: int, _stage: str, _sentence_id: str) -> None:
    return


def _not_cancelled() -> bool:
    return False


def _check_cancel(is_cancelled: CancelCallback) -> None:
    if is_cancelled():
        raise JobCancelled("cancel requested")


def _ensure_draft_sentence(sid: str, draft_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    obj = draft_map.get(sid)
    if isinstance(obj, dict):
        out = dict(obj)
        out["sentence_id"] = str(out.get("sentence_id") or sid)
        return out
    return {
        "sentence_id": sid,
        "start_sec": 0.0,
        "end_sec": 0.1,
        "ja_text": "",
        "zh_text": "",
        "review_text_ja": "",
        "review_text_zh": "",
        "review_state": "pending",
        "translation_status": "",
        "word_items": [],
    }


def run_job_worker(
    *,
    output_dir: str,
    config_path: str | None,
    payload: dict[str, Any],
    emit_log: LogCallback | None = None,
    emit_progress: ProgressCallback | None = None,
    is_cancelled: CancelCallback | None = None,
) -> dict[str, Any]:
    log = emit_log or _noop_log
    progress = emit_progress or _noop_progress
    check_cancel = is_cancelled or _not_cancelled

    task_id = str(payload.get("task_id") or "").strip()
    kind = str(payload.get("kind") or "").strip()
    raw_ids = payload.get("sentence_ids")
    raw_drafts = payload.get("draft_sentences")
    if not task_id:
        raise ValueError("task_id 不能为空")
    if kind not in {"relisten", "auto_translate", "retranslate"}:
        raise ValueError(f"不支持的任务类型: {kind}")
    if not isinstance(raw_ids, list):
        raise ValueError("sentence_ids 必须是数组")
    if raw_drafts is not None and not isinstance(raw_drafts, list):
        raise ValueError("draft_sentences 必须是数组")

    sentence_ids: list[str] = []
    seen: set[str] = set()
    for x in raw_ids:
        sid = str(x or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        sentence_ids.append(sid)
    if not sentence_ids:
        raise ValueError("有效 sentence_ids 为空")

    draft_map: dict[str, dict[str, Any]] = {}
    for d in raw_drafts or []:
        if not isinstance(d, dict):
            continue
        sid = str(d.get("sentence_id") or "").strip()
        if not sid:
            continue
        draft_map[sid] = dict(d)

    cfg_path = Path(config_path).resolve() if config_path else None
    service = ReviewService(output_dir=Path(output_dir), config_path=cfg_path)
    log(f"[worker] 开始任务 kind={kind} task_id={task_id} sentence_count={len(sentence_ids)}")

    total = len(sentence_ids) * (2 if kind == "retranslate" else 1)
    done = 0
    sid_state: dict[str, dict[str, bool]] = {}
    for sid in sentence_ids:
        if kind == "relisten":
            sid_state[sid] = {"relisten": False, "translate": True}
        elif kind == "auto_translate":
            sid_state[sid] = {"relisten": True, "translate": False}
        else:
            sid_state[sid] = {"relisten": False, "translate": False}

    ja_updates: list[dict[str, str]] = []
    zh_updates: list[dict[str, str]] = []
    errors: list[str] = []

    _check_cancel(check_cancel)

    if kind in {"relisten", "retranslate"}:
        log("[worker] 进入重识别阶段")
        for sid in sentence_ids:
            _check_cancel(check_cancel)
            try:
                log(f"[worker] 重识别开始 sid={sid}")
                draft_sentence = _ensure_draft_sentence(sid, draft_map)
                resp = service.relisten_sentence_once(
                    task_id=task_id,
                    sentence_id=sid,
                    draft_sentence=draft_sentence,
                )
                ja_text = str(resp.get("ja_text") or "").strip()
                if ja_text:
                    ja_updates.append({"sentence_id": sid, "ja_text": ja_text})
                    draft_sentence["review_text_ja"] = ja_text
                    draft_map[sid] = draft_sentence
                    sid_state[sid]["relisten"] = True
                    log(f"[worker] 重识别成功 sid={sid} ja_len={len(ja_text)}")
                else:
                    sid_state[sid]["relisten"] = False
                    errors.append(f"{sid}: 重识别结果为空")
                    log(f"[worker] 重识别失败 sid={sid} reason=结果为空")
            except JobCancelled:
                raise
            except Exception as e:
                sid_state[sid]["relisten"] = False
                errors.append(f"{sid}: 重识别失败 - {e}")
                log(f"[worker] 重识别异常 sid={sid} error={e}")
            done += 1
            progress(done, total, "relisten", sid)

    if kind in {"auto_translate", "retranslate"}:
        try:
            _check_cancel(check_cancel)
            log("[worker] 进入翻译阶段")
            draft_sentences = [_ensure_draft_sentence(sid, draft_map) for sid in sentence_ids]
            resp = service.auto_translate_sentences(
                task_id=task_id,
                sentence_ids=sentence_ids,
                draft_sentences=draft_sentences,
            )
            trans = resp.get("translations")
            trans_list = trans if isinstance(trans, list) else []
            trans_map = {
                str(item.get("sentence_id") or ""): str(item.get("zh_text") or "")
                for item in trans_list
                if isinstance(item, dict)
            }
            for sid in sentence_ids:
                _check_cancel(check_cancel)
                zh_text = trans_map.get(sid)
                if zh_text is not None:
                    zh_updates.append({"sentence_id": sid, "zh_text": zh_text})
                    sid_state[sid]["translate"] = True
                    log(f"[worker] 翻译成功 sid={sid} zh_len={len(zh_text)}")
                else:
                    sid_state[sid]["translate"] = False
                    errors.append(f"{sid}: 未收到翻译结果")
                    log(f"[worker] 翻译失败 sid={sid} reason=未返回结果")
                done += 1
                progress(done, total, "translate", sid)
        except JobCancelled:
            raise
        except Exception as e:
            log(f"[worker] 批量翻译异常 error={e}")
            for sid in sentence_ids:
                sid_state[sid]["translate"] = False
                done += 1
                progress(done, total, "translate", sid)
            errors.append(f"批量翻译失败: {e}")

    success_ids: list[str] = []
    failed_ids: list[str] = []
    for sid in sentence_ids:
        st = sid_state.get(sid) or {}
        if kind == "relisten":
            ok = bool(st.get("relisten"))
        elif kind == "auto_translate":
            ok = bool(st.get("translate"))
        else:
            ok = bool(st.get("relisten")) and bool(st.get("translate"))
        if ok:
            success_ids.append(sid)
        else:
            failed_ids.append(sid)

    if success_ids and not failed_ids:
        status = "succeeded"
    elif success_ids and failed_ids:
        status = "partial_failed"
    else:
        status = "failed"

    log(f"[worker] 任务结束 status={status} ok={len(success_ids)} fail={len(failed_ids)}")

    return {
        "status": status,
        "updates": {
            "ja": ja_updates,
            "zh": zh_updates,
        },
        "success_ids": success_ids,
        "failed_ids": failed_ids,
        "errors": errors[:50],
    }
