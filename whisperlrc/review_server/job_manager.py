from __future__ import annotations

import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from whisperlrc.review_server.job_worker import JobCancelled, run_job_worker


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewJobManager:
    MAX_ATTEMPTS = 2
    TERMINAL = {"succeeded", "partial_failed", "failed", "cancelled"}

    def __init__(self, *, output_dir: Path, config_path: Path | None = None) -> None:
        self.output_dir = output_dir.resolve()
        self.config_path = config_path.resolve() if config_path else None
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._runner = threading.Thread(target=self._run_loop, name="review-job-runner", daemon=True)
        self._runner.start()
        self._console_log("job manager started")

    def _console_log(self, message: str) -> None:
        text = f"[review-job] {message}"
        try:
            print(text, flush=True)
        except Exception:
            pass

    def _append_job_log(self, job_id: str, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            logs = rec.get("logs")
            if not isinstance(logs, list):
                logs = []
            logs.append(text)
            rec["logs"] = logs[-300:]
            rec["updated_at"] = _now_iso()

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return True
            return bool(rec.get("cancel_requested"))

    def shutdown(self) -> None:
        self._stop.set()
        self._queue.put("")
        try:
            self._runner.join(timeout=2.0)
        except Exception:
            pass

    def create_job(
        self,
        *,
        task_id: str,
        kind: str,
        sentence_ids: list[str],
        draft_sentences: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if kind not in {"relisten", "auto_translate", "retranslate"}:
            raise ValueError(f"不支持的任务类型: {kind}")
        cleaned_ids: list[str] = []
        seen: set[str] = set()
        for x in sentence_ids:
            sid = str(x or "").strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            cleaned_ids.append(sid)
        if not cleaned_ids:
            raise ValueError("sentence_ids 不能为空")

        total = len(cleaned_ids) * (2 if kind == "retranslate" else 1)
        job_id = uuid.uuid4().hex
        rec = {
            "job_id": job_id,
            "task_id": task_id,
            "kind": kind,
            "status": "queued",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "attempt": 0,
            "max_attempts": self.MAX_ATTEMPTS,
            "cancel_requested": False,
            "progress": {
                "done": 0,
                "total": total,
                "stage": "queued",
                "current_sentence_id": "",
            },
            "payload": {
                "task_id": task_id,
                "kind": kind,
                "sentence_ids": cleaned_ids,
                "draft_sentences": draft_sentences or [],
            },
            "result": {
                "status": "queued",
                "updates": {"ja": [], "zh": []},
                "success_ids": [],
                "failed_ids": [],
                "errors": [],
            },
            "attempt_errors": [],
            "logs": [],
            "error": "",
        }
        with self._lock:
            self._jobs[job_id] = rec
        self._queue.put(job_id)
        self._console_log(
            f"job created id={job_id} kind={kind} task_id={task_id} sentence_count={len(cleaned_ids)}"
        )
        self._append_job_log(job_id, f"created kind={kind} task_id={task_id} sentence_count={len(cleaned_ids)}")
        return self.get_job(task_id=task_id, job_id=job_id)

    def get_job(self, *, task_id: str, job_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None or str(rec.get("task_id") or "") != str(task_id):
                raise FileNotFoundError(f"任务不存在: {job_id}")
            return self._public_record(rec)

    def cancel_job(self, *, task_id: str, job_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None or str(rec.get("task_id") or "") != str(task_id):
                raise FileNotFoundError(f"任务不存在: {job_id}")
            if rec.get("status") in self.TERMINAL:
                return self._public_record(rec)
            rec["cancel_requested"] = True
            rec["updated_at"] = _now_iso()
            if rec.get("status") == "queued":
                rec["status"] = "cancelled"
                rec["progress"]["stage"] = "cancelled"
                logs = rec.get("logs")
                if not isinstance(logs, list):
                    logs = []
                logs.append("cancelled while queued")
                rec["logs"] = logs[-300:]
                self._console_log(f"job cancelled id={job_id} status=queued")
                return self._public_record(rec)

            logs = rec.get("logs")
            if not isinstance(logs, list):
                logs = []
            logs.append("cancel_requested while running")
            rec["logs"] = logs[-300:]
            self._console_log(f"job cancel requested id={job_id} status=running")
            return self._public_record(rec)

    def _public_record(self, rec: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": rec.get("job_id"),
            "task_id": rec.get("task_id"),
            "kind": rec.get("kind"),
            "status": rec.get("status"),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
            "attempt": rec.get("attempt"),
            "max_attempts": rec.get("max_attempts"),
            "progress": {
                "done": int((rec.get("progress") or {}).get("done") or 0),
                "total": int((rec.get("progress") or {}).get("total") or 0),
                "stage": str((rec.get("progress") or {}).get("stage") or ""),
                "current_sentence_id": str((rec.get("progress") or {}).get("current_sentence_id") or ""),
            },
            "result": rec.get("result") or {},
            "error": str(rec.get("error") or ""),
            "attempt_errors": [str(x or "") for x in (rec.get("attempt_errors") or []) if str(x or "").strip()],
            "logs": [str(x or "") for x in (rec.get("logs") or []) if str(x or "").strip()],
        }

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not job_id:
                continue
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    continue
                if rec.get("status") in self.TERMINAL:
                    continue
            self._console_log(f"job dequeued id={job_id}")
            self._append_job_log(job_id, "dequeued")
            self._run_job(job_id)

    def _run_job(self, job_id: str) -> None:
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                if rec.get("cancel_requested"):
                    rec["status"] = "cancelled"
                    rec["progress"]["stage"] = "cancelled"
                    rec["updated_at"] = _now_iso()
                    return
                rec["status"] = "running"
                rec["attempt"] = attempt
                rec["updated_at"] = _now_iso()
                rec["progress"]["stage"] = "running"
                rec["error"] = ""
                payload = dict(rec.get("payload") or {})
                kind = str(rec.get("kind") or "")
                task_id = str(rec.get("task_id") or "")
                total = int((rec.get("progress") or {}).get("total") or 0)
            self._console_log(
                f"job attempt start id={job_id} attempt={attempt}/{self.MAX_ATTEMPTS} kind={kind} task_id={task_id} total={total}"
            )
            self._append_job_log(job_id, f"attempt_start {attempt}/{self.MAX_ATTEMPTS} kind={kind} total={total}")

            outcome = self._run_attempt(job_id=job_id, payload=payload)
            if outcome["kind"] == "cancelled":
                with self._lock:
                    rec = self._jobs.get(job_id)
                    if rec is None:
                        return
                    rec["status"] = "cancelled"
                    rec["progress"]["stage"] = "cancelled"
                    rec["updated_at"] = _now_iso()
                self._console_log(f"job cancelled id={job_id} during attempt={attempt}")
                self._append_job_log(job_id, f"cancelled during attempt={attempt}")
                return

            if outcome["kind"] == "result":
                result = outcome["result"]
                status = str(result.get("status") or "failed")
                ignored_due_to_cancel = False
                with self._lock:
                    rec = self._jobs.get(job_id)
                    if rec is None:
                        return
                    if rec.get("cancel_requested"):
                        rec["status"] = "cancelled"
                        rec["progress"]["stage"] = "cancelled"
                        rec["updated_at"] = _now_iso()
                        ignored_due_to_cancel = True
                    else:
                        rec["result"] = result
                        rec["status"] = status
                        rec["updated_at"] = _now_iso()
                        rec["progress"]["stage"] = "done"
                if ignored_due_to_cancel:
                    self._append_job_log(job_id, "result_ignored_due_to_cancel")
                    self._console_log(f"job cancelled id={job_id} after result")
                    return
                ok_count = len((result.get("success_ids") or [])) if isinstance(result, dict) else 0
                fail_count = len((result.get("failed_ids") or [])) if isinstance(result, dict) else 0
                self._console_log(
                    f"job attempt done id={job_id} attempt={attempt} status={status} ok={ok_count} fail={fail_count}"
                )
                self._append_job_log(
                    job_id,
                    f"attempt_done {attempt}/{self.MAX_ATTEMPTS} status={status} ok={ok_count} fail={fail_count}",
                )
                if status == "failed" and attempt < self.MAX_ATTEMPTS:
                    with self._lock:
                        rec = self._jobs.get(job_id)
                        if rec is None:
                            return
                        rec["progress"]["stage"] = "retrying"
                        rec["updated_at"] = _now_iso()
                    self._console_log(f"job retry scheduled id={job_id} next_attempt={attempt + 1}")
                    self._append_job_log(job_id, f"retry_scheduled next_attempt={attempt + 1}")
                    continue
                return

            err = str(outcome.get("error") or "后台任务失败")
            retryable = bool(outcome.get("retryable", True))
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec["error"] = err
                hist = rec.get("attempt_errors")
                if not isinstance(hist, list):
                    hist = []
                hist.append(err)
                rec["attempt_errors"] = hist[-20:]
                rec["updated_at"] = _now_iso()
                rec["progress"]["stage"] = "error"
            self._console_log(
                f"job attempt error id={job_id} attempt={attempt} retryable={retryable} error={err}"
            )
            self._append_job_log(
                job_id,
                f"attempt_error {attempt}/{self.MAX_ATTEMPTS} retryable={retryable} error={err}",
            )
            if retryable and attempt < self.MAX_ATTEMPTS:
                with self._lock:
                    rec = self._jobs.get(job_id)
                    if rec is None:
                        return
                    rec["progress"]["stage"] = "retrying"
                    rec["updated_at"] = _now_iso()
                self._console_log(f"job retry scheduled id={job_id} next_attempt={attempt + 1}")
                self._append_job_log(job_id, f"retry_scheduled next_attempt={attempt + 1}")
                continue
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec["status"] = "failed"
                rec["result"] = {
                    "status": "failed",
                    "updates": {"ja": [], "zh": []},
                    "success_ids": [],
                    "failed_ids": list((rec.get("payload") or {}).get("sentence_ids") or []),
                    "errors": [err],
                }
                rec["updated_at"] = _now_iso()
            self._console_log(f"job failed id={job_id} after attempts={attempt}")
            self._append_job_log(job_id, f"failed after attempts={attempt}")
            return

    def _run_attempt(self, *, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._is_cancel_requested(job_id):
            return {"kind": "cancelled"}

        def emit_progress(done: int, total: int, stage: str, sid: str) -> None:
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                prog = rec.get("progress") or {}
                prog["done"] = int(done)
                prog["total"] = int(total)
                prog["stage"] = str(stage or "running")
                prog["current_sentence_id"] = str(sid or "")
                rec["progress"] = prog
                rec["updated_at"] = _now_iso()
            self._console_log(
                f"job progress id={job_id} stage={stage} done={done}/{total} sid={sid}"
            )
            self._append_job_log(
                job_id,
                f"progress stage={stage} done={done}/{total} sid={sid}",
            )

        def emit_log(message: str) -> None:
            text = str(message or "").strip()
            if not text:
                return
            self._console_log(f"job worker id={job_id} {text}")
            self._append_job_log(job_id, f"worker {text}")

        def is_cancelled() -> bool:
            return self._is_cancel_requested(job_id)

        try:
            result = run_job_worker(
                output_dir=str(self.output_dir),
                config_path=str(self.config_path) if self.config_path else None,
                payload=payload,
                emit_log=emit_log,
                emit_progress=emit_progress,
                is_cancelled=is_cancelled,
            )
            if is_cancelled():
                return {"kind": "cancelled"}
            return {"kind": "result", "result": result}
        except JobCancelled:
            return {"kind": "cancelled"}
        except Exception as e:
            base = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc(limit=12)
            err = f"{base}\n{tb}" if tb else base
            self._console_log(f"job worker exception id={job_id} error={base}")
            self._append_job_log(job_id, f"worker_exception {err}")
            return {"kind": "error", "retryable": True, "error": err}
