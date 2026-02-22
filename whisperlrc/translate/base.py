from __future__ import annotations

from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def translate_batch(self, texts: list[str], src: str = "ja", tgt: str = "zh-Hans") -> list[str]:
        raise NotImplementedError
