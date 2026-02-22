# WhisperLRC

第一阶段本地程序，用于批量 ASR 处理与审核 JSON 导出。
翻译功能目前仍是 LLM 接口占位实现。

## 快速开始

```bash
pip install faster-whisper tomli
python main.py
```

## 配置

默认配置文件是 `settings.toml`。当前版本仅支持交互式运行，程序会在菜单中读取配置并执行。
请在 `[translation]` 中设置你后续的 LLM 集成参数，并实现
`whisperlrc/translate/llm_backend.py`。
