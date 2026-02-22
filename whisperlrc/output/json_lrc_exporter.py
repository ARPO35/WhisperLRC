from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whisperlrc.output.lrc_writer import write_lrc


def _export_one_json(json_path: Path, output_dir: Path | None = None) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根对象格式错误：{json_path}")
    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(f"JSON 不包含 sentences 数组：{json_path}")
    base_name = str(payload.get("file_id") or json_path.stem)
    out_dir = output_dir or json_path.parent
    return write_lrc(output_dir=out_dir, base_name=base_name, sentences=sentences)


def export_lrc_from_json_path(path: Path, output_dir: Path | None = None) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"仅支持 .json 文件：{path}")
        return [_export_one_json(path, output_dir)]

    if not path.exists() or not path.is_dir():
        raise ValueError(f"路径不存在或不是目录：{path}")

    out_paths: list[Path] = []
    for json_file in sorted(path.glob("*.json")):
        out_paths.append(_export_one_json(json_file, output_dir))
    if not out_paths:
        raise ValueError(f"目录中未找到 JSON 文件：{path}")
    return out_paths

