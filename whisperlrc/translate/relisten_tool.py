from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from whisperlrc.translate.tooling import TranslationToolContext


@dataclass
class RelistenOptions:
    padding_sec: float = 0.5
    candidate_count: int = 3
    sample_rate: int = 16000


class WhisperRelistenExecutor:
    def __init__(self, ctx: TranslationToolContext, options: RelistenOptions | None = None) -> None:
        self.ctx = ctx
        self.options = options or RelistenOptions()
        self._model: WhisperModel | None = None
        self._audio_cache: Any | None = None

    def relisten_by_global_index(self, global_index: int) -> dict[str, Any]:
        if global_index < 0 or global_index >= len(self.ctx.sentences):
            return {
                "ok": False,
                "error": f"index 越界：{global_index}",
                "index": global_index,
                "candidates": [],
            }

        sent = self.ctx.sentences[global_index]
        clip_start = max(0.0, float(sent.start_sec) - self.options.padding_sec)
        clip_end = max(clip_start, float(sent.end_sec) + self.options.padding_sec)
        try:
            clip = self._slice_audio(clip_start, clip_end)
            candidates = self._decode_candidates(clip)
        except Exception as e:
            return {
                "ok": False,
                "error": f"重听失败：{e}",
                "index": global_index,
                "sentence_id": sent.sentence_id,
                "clip_start_sec": clip_start,
                "clip_end_sec": clip_end,
                "candidates": [],
            }

        return {
            "ok": True,
            "index": global_index,
            "sentence_id": sent.sentence_id,
            "source_text": sent.ja_text,
            "clip_start_sec": clip_start,
            "clip_end_sec": clip_end,
            "candidates": candidates,
        }

    def _slice_audio(self, start_sec: float, end_sec: float) -> Any:
        audio = self._load_audio()
        start_idx = max(0, int(start_sec * self.options.sample_rate))
        end_idx = min(len(audio), int(end_sec * self.options.sample_rate))
        if end_idx <= start_idx:
            return audio[start_idx : start_idx + 1]
        return audio[start_idx:end_idx]

    def _load_audio(self) -> Any:
        if self._audio_cache is None:
            self._audio_cache = decode_audio(self.ctx.audio_path, sampling_rate=self.options.sample_rate)
        return self._audio_cache

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            asr_cfg = self.ctx.asr_config
            self._model = WhisperModel(
                model_size_or_path=asr_cfg.model,
                device=asr_cfg.device,
                compute_type=asr_cfg.compute_type,
            )
        return self._model

    def _decode_candidates(self, clip_audio: Any) -> list[str]:
        configs: list[dict[str, Any]] = [
            {"beam_size": 5, "best_of": 5, "temperature": 0.0},
            {"beam_size": 1, "best_of": 1, "temperature": 0.2},
            {"beam_size": 8, "best_of": 8, "temperature": 0.0},
        ]
        candidates: list[str] = []
        for cfg in configs:
            text = self._transcribe_once(clip_audio, cfg).strip()
            if text and text not in candidates:
                candidates.append(text)

        if not candidates:
            candidates = [""]
        while len(candidates) < self.options.candidate_count:
            candidates.append(candidates[-1])
        return candidates[: self.options.candidate_count]

    def _transcribe_once(self, clip_audio: Any, decode_cfg: dict[str, Any]) -> str:
        asr_cfg = self.ctx.asr_config
        model = self._get_model()
        params: dict[str, Any] = {
            "language": asr_cfg.language,
            "vad_filter": False,
            "word_timestamps": False,
            "condition_on_previous_text": False,
        }
        params.update(decode_cfg)
        try:
            segments, _info = model.transcribe(clip_audio, **params)
        except TypeError:
            segments, _info = model.transcribe(
                clip_audio,
                language=asr_cfg.language,
                vad_filter=False,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
        texts = [(seg.text or "").strip() for seg in list(segments)]
        return "".join(t for t in texts if t)
