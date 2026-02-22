from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WordItem:
    word: str
    start_sec: float | None
    end_sec: float | None
    confidence: float | None


@dataclass
class TokenItem:
    token_id: int
    token_text: str
    confidence: None = None


@dataclass
class SentenceItem:
    sentence_id: str
    start_sec: float
    end_sec: float
    ja_text: str
    zh_text: str | None
    translation_status: str
    segment_confidence: float | None
    word_items: list[WordItem] = field(default_factory=list)
    token_items: list[TokenItem] = field(default_factory=list)


@dataclass
class FileProcessResult:
    status: str
    sentences: list[SentenceItem]
    error: str | None = None
    logs: list[str] = field(default_factory=list)
