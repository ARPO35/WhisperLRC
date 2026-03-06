from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Mapping


LLM_API_KEY_ENV_VAR = "WHISPERLRC_LLM_API_KEY"


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


@dataclass
class ConfigLoadMeta:
    loaded_files: list[str] = field(default_factory=list)
    local_override_path: str = ""
    api_key_source: str = "missing"


def _merge_section(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)


def _resolve_toml_reader() -> Any | None:
    try:
        import tomllib as _tomllib  # type: ignore

        return _tomllib
    except ModuleNotFoundError:
        try:
            import tomli as _tomli  # type: ignore

            return _tomli
        except ModuleNotFoundError:
            return None


def _load_toml_data(path: Path, toml_reader: Any) -> dict[str, Any]:
    with path.open("rb") as f:
        data = toml_reader.load(f)
    if isinstance(data, dict):
        return data
    return {}


def _merge_config_data(cfg: AppConfig, data: dict[str, Any]) -> None:
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


def _extract_api_key(data: dict[str, Any]) -> str:
    translation = data.get("translation")
    if not isinstance(translation, dict):
        return ""
    return str(translation.get("llm_api_key") or "").strip()


def _default_local_override_path(config_path: Path) -> Path:
    if config_path.suffix:
        filename = f"{config_path.stem}.local{config_path.suffix}"
    else:
        filename = f"{config_path.name}.local"
    return config_path.with_name(filename)


def load_config_with_meta(
    config_path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[AppConfig, ConfigLoadMeta]:
    cfg = AppConfig()
    local_override = _default_local_override_path(config_path)
    meta = ConfigLoadMeta(local_override_path=str(local_override))

    toml_reader = _resolve_toml_reader()
    if toml_reader is None:
        logging.warning(
            "TOML 解析器不可用，将使用内置默认配置。"
            "请使用 Python 3.11+ 或执行 `pip install tomli` 以加载 %s",
            config_path,
        )
        return cfg, meta

    for source_path in (config_path, local_override):
        if not source_path.exists():
            continue
        data = _load_toml_data(source_path, toml_reader)
        _merge_config_data(cfg, data)
        meta.loaded_files.append(str(source_path))
        file_key = _extract_api_key(data)
        if file_key:
            meta.api_key_source = f"file:{source_path.name}"

    env_map: Mapping[str, str]
    if env is None:
        env_map = os.environ
    else:
        env_map = env
    env_api_key = str(env_map.get(LLM_API_KEY_ENV_VAR, "")).strip()
    if env_api_key:
        cfg.translation.llm_api_key = env_api_key
        meta.api_key_source = f"env:{LLM_API_KEY_ENV_VAR}"

    return cfg, meta


def load_config(config_path: Path) -> AppConfig:
    cfg, _meta = load_config_with_meta(config_path)
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
