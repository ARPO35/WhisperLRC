from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ReviewState = Literal["pending", "accepted", "needs_check"]
TaskStatus = Literal["todo", "in_review", "done", "partial", "failed", "ok", "asr_done", "translating", "cancelled"]


class SentencePatchRequest(BaseModel):
    review_text_ja: str | None = Field(default=None, description="人工校对后的日文")
    review_text_zh: str | None = Field(default=None, description="人工校对后的中文")
    review_state: ReviewState | None = Field(default=None, description="校对状态")
    start_sec: float | None = Field(default=None, description="句子起始秒")
    end_sec: float | None = Field(default=None, description="句子结束秒")


class InsertSentenceRequest(BaseModel):
    min_duration_sec: float | None = Field(default=None, description="插入句子的最小时长（秒）")


class TaskStatusPatchRequest(BaseModel):
    status: TaskStatus = Field(description="任务状态")


class ExportRequest(BaseModel):
    export_ja: bool = Field(default=True, description="是否导出日文 LRC")
    export_zh: bool = Field(default=True, description="是否导出中文 LRC")
    output_dir: str | None = Field(default=None, description="导出目录，留空则使用 JSON 所在目录")
