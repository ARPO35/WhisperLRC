from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from whisperlrc.review_server.schemas import ExportRequest, SentencePatchRequest, TaskStatusPatchRequest
from whisperlrc.review_server.service import ReviewService


def _raise_http(err: Exception) -> None:
    if isinstance(err, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(err)) from err
    raise HTTPException(status_code=400, detail=str(err)) from err


def create_app(*, output_dir: Path) -> FastAPI:
    app = FastAPI(title="WhisperLRC Review UI", version="1.0.0")
    service = ReviewService(output_dir=output_dir)
    static_dir = Path(__file__).resolve().parent / "static"
    index_file = static_dir / "index.html"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if not index_file.exists():
            return HTMLResponse("<h1>Review UI 静态资源缺失</h1>", status_code=500)
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "output_dir": str(service.output_dir),
            "version": "1.0.0",
        }

    @app.get("/api/tasks")
    def list_tasks() -> dict[str, Any]:
        return {"tasks": service.list_tasks()}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        try:
            return service.get_task(task_id)
        except Exception as e:
            _raise_http(e)
            raise

    @app.patch("/api/tasks/{task_id}/sentences/{sentence_id}")
    def patch_sentence(task_id: str, sentence_id: str, body: SentencePatchRequest) -> dict[str, Any]:
        try:
            return service.update_sentence(
                task_id=task_id,
                sentence_id=sentence_id,
                review_text_ja=body.review_text_ja,
                review_text_zh=body.review_text_zh,
                review_state=body.review_state,
                start_sec=body.start_sec,
                end_sec=body.end_sec,
            )
        except Exception as e:
            _raise_http(e)
            raise

    @app.patch("/api/tasks/{task_id}/status")
    def patch_status(task_id: str, body: TaskStatusPatchRequest) -> dict[str, Any]:
        try:
            return service.update_task_status(task_id, status=body.status)
        except Exception as e:
            _raise_http(e)
            raise

    @app.post("/api/tasks/{task_id}/export")
    def export_task(task_id: str, body: ExportRequest) -> dict[str, Any]:
        try:
            return service.export_task(
                task_id=task_id,
                export_ja=body.export_ja,
                export_zh=body.export_zh,
                output_dir=body.output_dir,
            )
        except Exception as e:
            _raise_http(e)
            raise

    @app.get("/api/tasks/{task_id}/audio")
    def get_audio(task_id: str) -> FileResponse:
        try:
            audio_path = service.resolve_audio_path(task_id)
            return FileResponse(path=str(audio_path), filename=audio_path.name)
        except Exception as e:
            _raise_http(e)
            raise

    return app
