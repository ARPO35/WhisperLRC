from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Translator(ABC):
    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        src: str = "ja",
        tgt: str = "zh-Hans",
        retry: int = 0,
        event_cb: Callable[[dict[str, Any]], None] | None = None,
        cancel_token: Any | None = None,
    ) -> list[str]:
        raise NotImplementedError
