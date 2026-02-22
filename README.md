# WhisperLRC

第一阶段本地程序，用于批量 ASR 处理与审核 JSON 导出。
翻译功能已支持 OpenAI 兼容接口（YAML 返回协议）。

## 快速开始

```bash
pip install faster-whisper tomli pyyaml
python main.py
```

## 交互方式

- 程序启动后直接进入主菜单，再进入功能页（批处理/配置/检查/帮助）。
- 数字键即时进入对应页面或执行选项，不需要回车确认。
- `q` 在任意页面都返回主菜单。
- `Esc` 全局返回（信息页/输入页返回上一级，在主菜单按 `Esc` 退出程序）。
- 关键动作（如开始批处理）会要求 `y/n` 二次确认，默认 `n`。
- 所有结果/摘要信息使用独立页面展示，按 `Esc` 关闭返回。
- 路径输入时，直接回车为空会取消本次操作并返回当前页面。
- 配置页支持逐项修改所有配置字段；修改后需在“保存并写入当前配置文件”生效到磁盘。

## 配置

默认配置文件是 `settings.toml`。当前版本仅支持交互式运行，程序会在菜单中读取配置并执行。
请在 `[translation]` 中至少配置：

- `llm_model`
- `llm_base_url`
- `llm_api_key`
- `llm_system_prompt`（可配置系统提示词）
- `llm_batch_size`（默认每组 10 句）
- `llm_context_window`（默认前后文各 5 句）
- `llm_preferences`（可配置翻译偏好，支持术语映射或自然语言说明）

LLM 必须返回 YAML，格式要求如下（`terms` 可选）：

```yaml
translations:
  - index: 0
    text: 翻译结果
terms:
  - src: 原文术语
    tgt: 译文术语
    note: 说明
```

程序会把 `terms` 作为会话内术语词典增量合并，仅在当前批处理中生效，不会写回磁盘。
