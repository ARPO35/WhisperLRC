from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faster_whisper import WhisperModel

from whisperlrc.config import ASRConfig
from whisperlrc.types import SentenceItem, TokenItem, WordItem


@dataclass
class ASRRunOutput:
    duration_sec: float
    sentences: list[SentenceItem]


class FasterWhisperEngine:
    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self.model = WhisperModel(
            model_size_or_path=cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe(self, audio_path: str) -> ASRRunOutput:
        segments, info = self.model.transcribe(
            audio_path,
            language=self.cfg.language,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        out: list[SentenceItem] = []
        for idx, seg in enumerate(list(segments), start=1):
            words: list[WordItem] = []
            probs: list[float] = []
            for w in seg.words or []:
                conf = getattr(w, "probability", None)
                if conf is not None:
                    probs.append(float(conf))
                words.append(
                    WordItem(
                        word=w.word,
                        start_sec=float(w.start) if w.start is not None else None,
                        end_sec=float(w.end) if w.end is not None else None,
                        confidence=float(conf) if conf is not None else None,
                    )
                )

            token_items = self._extract_tokens(seg)
            seg_conf = sum(probs) / len(probs) if probs else None
            out.append(
                SentenceItem(
                    sentence_id=f"s_{idx:04d}",
                    start_sec=float(seg.start),
                    end_sec=float(seg.end),
                    ja_text=(seg.text or "").strip(),
                    zh_text=None,
                    translation_status="pending",
                    segment_confidence=seg_conf,
                    word_items=words,
                    token_items=token_items,
                )
            )
        return ASRRunOutput(duration_sec=float(info.duration), sentences=out)

    def _extract_tokens(self, seg: Any) -> list[TokenItem]:
        token_ids = getattr(seg, "tokens", None) or []
        token_items: list[TokenItem] = []
        for token_id in token_ids:
            token_items.append(
                TokenItem(
                    token_id=int(token_id),
                    token_text=self._decode_token(int(token_id)),
                )
            )
        return token_items

    def _decode_token(self, token_id: int) -> str:
        try:
            tok = self.model.hf_tokenizer.decode([token_id])  # type: ignore[attr-defined]
            return str(tok)
        except Exception:
            return f"<id:{token_id}>"
