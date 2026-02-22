from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any


@dataclass
class ASRConfig:
    backend: str = "faster_whisper"
    model: str = "large-v3"
    language: str = "ja"
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class PipelineConfig:
    input_formats: list[str] = field(default_factory=lambda: ["mp3", "wav"])
    retry_asr: int = 1
    retry_translate: int = 1


@dataclass
class TranslationConfig:
    backend: str = "llm"
    target: str = "zh-Hans"
    llm_provider: str = "openai_compatible"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    timeout_sec: int = 30


@dataclass
class OutputConfig:
    json_ext: str = ".json"
    write_lrc: bool = False


@dataclass
class SchemaConfig:
    version: str = "1.0"


@dataclass
class AppConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_section(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)


def load_config(config_path: Path) -> AppConfig:
    cfg = AppConfig()
    if not config_path.exists():
        return cfg

    toml_reader = None
    try:
        import tomllib as _tomllib  # type: ignore
        toml_reader = _tomllib
    except ModuleNotFoundError:
        try:
            import tomli as _tomli  # type: ignore
            toml_reader = _tomli
        except ModuleNotFoundError:
            logging.warning(
                "TOML 解析器不可用，将使用内置默认配置。"
                "请使用 Python 3.11+ 或执行 `pip install tomli` 以加载 %s",
                config_path,
            )
            return cfg

    with config_path.open("rb") as f:
        data = toml_reader.load(f)

    if isinstance(data.get("asr"), dict):
        _merge_section(cfg.asr, data["asr"])
    if isinstance(data.get("pipeline"), dict):
        _merge_section(cfg.pipeline, data["pipeline"])
    if isinstance(data.get("translation"), dict):
        _merge_section(cfg.translation, data["translation"])
    if isinstance(data.get("output"), dict):
        _merge_section(cfg.output, data["output"])
    if isinstance(data.get("schema"), dict):
        _merge_section(cfg.schema, data["schema"])
    return cfg


def apply_cli_overrides(
    cfg: AppConfig,
    device: str | None = None,
    model: str | None = None,
    translate_backend: str | None = None,
) -> AppConfig:
    if device:
        cfg.asr.device = device
    if model:
        cfg.asr.model = model
    if translate_backend:
        cfg.translation.backend = translate_backend
    return cfg
