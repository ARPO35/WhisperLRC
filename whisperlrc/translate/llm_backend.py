from __future__ import annotations

from whisperlrc.config import TranslationConfig
from whisperlrc.translate.base import Translator


class LLMTranslator(Translator):
    """
    预留给后续 LLM 集成的占位翻译器。

    可在 `translate_batch` 中接入 OpenAI 兼容接口、Anthropic
    或自定义网关实现。
    """

    def __init__(self, cfg: TranslationConfig) -> None:
        self.cfg = cfg

    def translate_batch(self, texts: list[str], src: str = "ja", tgt: str = "zh-Hans") -> list[str]:
        raise NotImplementedError(
            "LLM 翻译后端尚未实现。"
            "请实现 whisperlrc.translate.llm_backend.LLMTranslator.translate_batch。"
        )
