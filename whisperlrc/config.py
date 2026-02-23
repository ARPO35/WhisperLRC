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
    llm_enable_thinking: bool = True
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_prompt_file: str = "prompt.txt"
    llm_preferences_file: str = "preferences.txt"
    llm_batch_size: int = 10
    llm_context_window: int = 5
    timeout_sec: int = 30


@dataclass
class OutputConfig:
    default_input_dir: str = "input_audio"
    default_output_dir: str = "output"
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


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_quote(value)
    raise TypeError(f"不支持的 TOML 标量类型: {type(value)}")


def _toml_list(values: list[Any]) -> str:
    if not values:
        return "[]"
    if all(isinstance(v, str) for v in values):
        return "[" + ", ".join(_toml_quote(v) for v in values) + "]"
    if all(isinstance(v, dict) for v in values):
        rendered_items: list[str] = []
        for item in values:
            pairs = [f"{k} = {_toml_scalar(v)}" for k, v in item.items() if isinstance(v, (str, int, bool))]
            rendered_items.append("{ " + ", ".join(pairs) + " }")
        return "[\n  " + ",\n  ".join(rendered_items) + "\n]"
    return "[" + ", ".join(_toml_scalar(v) for v in values) + "]"


def save_config(config_path: Path, cfg: AppConfig) -> None:
    data = cfg.to_dict()
    lines: list[str] = []
    section_order = ["asr", "pipeline", "translation", "output", "schema"]

    for section in section_order:
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in section_data.items():
            if isinstance(value, list):
                lines.append(f"{key} = {_toml_list(value)}")
            else:
                lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
