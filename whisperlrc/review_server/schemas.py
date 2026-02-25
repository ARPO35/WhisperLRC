from __future__ import annotations

from typing import Any, Literal

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


class SentenceDraft(BaseModel):
    sentence_id: str = Field(description="句子 ID")
    start_sec: float = Field(description="句子起始秒")
    end_sec: float = Field(description="句子结束秒")
    ja_text: str = Field(default="", description="原始日文")
    zh_text: str = Field(default="", description="原始中文")
    review_text_ja: str = Field(default="", description="校对日文")
    review_text_zh: str = Field(default="", description="校对中文")
    review_state: str = Field(default="pending", description="校对状态")
    translation_status: str = Field(default="", description="翻译状态")
    segment_confidence: float | None = Field(default=None, description="句级置信度")
    word_items: list[dict[str, Any]] = Field(default_factory=list, description="词级时间戳与置信度")


class TaskSaveRequest(BaseModel):
    status: str = Field(description="任务状态")
    sentences: list[SentenceDraft] = Field(default_factory=list, description="完整句子列表")


class SentenceActionRequest(BaseModel):
    draft_sentence: SentenceDraft | None = Field(default=None, description="可选草稿句子，优先用于动作计算")


class BatchAutoTranslateRequest(BaseModel):
    sentence_ids: list[str] = Field(default_factory=list, description="待翻译句子 ID 列表（按翻译顺序）")
    draft_sentences: list[SentenceDraft] = Field(default_factory=list, description="可选草稿句子列表")
