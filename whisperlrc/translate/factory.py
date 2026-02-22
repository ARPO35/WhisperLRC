from __future__ import annotations

import logging

from whisperlrc.config import TranslationConfig
from whisperlrc.translate.base import Translator


def build_translator(cfg: TranslationConfig) -> Translator:
    backend = cfg.backend.lower().strip()
    if backend == "llm":
        from whisperlrc.translate.llm_backend import LLMTranslator

        return LLMTranslator(cfg)
    if backend == "auto":
        logging.info("translation.backend=auto 已解析为 llm 占位后端")
        from whisperlrc.translate.llm_backend import LLMTranslator

        return LLMTranslator(cfg)
    raise ValueError(f"未知翻译后端：{cfg.backend}")
