from __future__ import annotations

from dataclasses import dataclass

from whisperlrc.config import ASRConfig


@dataclass
class SentenceRef:
    sentence_id: str
    start_sec: float
    end_sec: float
    ja_text: str


@dataclass
class TranslationToolContext:
    audio_path: str
    sentences: list[SentenceRef]
    asr_config: ASRConfig
