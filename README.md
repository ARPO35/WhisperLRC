# WhisperLRC

第一阶段本地程序，用于批量 ASR 处理与审核 JSON 导出。
翻译功能已支持 OpenAI 兼容接口（JSON 返回协议）。

## 快速开始

```bash
pip install faster-whisper tomli
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
- 关键执行前会出现醒目的确认提示（`y` 执行 / `n` 取消）。
- 检查页提供 API 测试（发送 `hello` 并检查回复）。
- 在“主菜单->配置->修改配置项”中，LLM 参数使用独立聚合页面管理。

## 配置

默认配置文件是 `settings.toml`。当前版本仅支持交互式运行，程序会在菜单中读取配置并执行。
请在 `[output]` 中设置默认路径：

- `default_input_dir`（批处理默认输入目录）
- `default_output_dir`（批处理默认输出目录）

请在 `[translation]` 中至少配置：

- `llm_model`
- `llm_base_url`
- `llm_api_key`
- `llm_prompt_file`（提示词文件路径，支持相对或绝对路径）
- `llm_preferences_file`（翻译偏好文件路径，支持相对或绝对路径）
- `llm_batch_size`（默认每组 10 句）
- `llm_context_window`（默认前后文各 5 句）

提示词与翻译偏好从上述路径读取。若配置相对路径，则按项目根目录解析：

- `prompt.txt`：系统提示词模板。必须包含 `{input}`，可使用 `{perf}` 插入偏好字典。
- `preferences.txt`：每行一条偏好。
- 术语语法：`src => tgt` 或 `src => tgt | note`
- 说明语法：`note: 文本`
- 注释语法：`# ...`

LLM 必须返回 JSON，格式要求如下（`terms` 可选）：

```json
{
  "translations": [
    { "index": 0, "text": "翻译结果" }
  ],
  "terms": [
    { "src": "原文术语", "tgt": "译文术语", "note": "说明" }
  ]
}
```

若本次没有新增术语，可只返回 `translations`，省略 `terms`。

程序会把 `terms` 作为会话内术语词典增量合并，仅在当前批处理中生效，不会写回磁盘。

## 处理中页面

- 从“主菜单->批处理”确认执行后，程序会立即跳转到“主菜单->批处理->处理中”。
- 页面会显示文件级进度（当前文件/总文件）和翻译分组进度（当前分组/总分组）。
- 页面会显示最近一次发送给 LLM 的原始请求 JSON，以及最近一次返回的原始响应 JSON（格式化展示）。
- 处理中按 `Esc` 会发起取消请求，并返回上一级页面；取消在当前步骤安全结束后生效。

### COT 处理

LLM 响应支持两种思考内容形式，用于提升翻译质量：
- `<think>...</think>` 块
- 可选 `cot` 字段

程序处理规则：
- 思考内容会被提取用于运行时观测（处理中页面/调试）。
- 思考内容不参与最终翻译结果写入。
- 解析时会优先严格解析；若响应带有前后缀文本，会自动清洗并提取 JSON 对象。
- 主流程仍以 `translations` 为必需字段，`terms` 与 `cot` 均为可选。

## LRC 导出与校对

- `output.write_lrc = true` 时，批处理会在写出 JSON 的同时写出中文 LRC（`<basename>.lrc`）。
- 若同名 LRC 已存在，会自动写为 `<basename>_1.lrc`、`<basename>_2.lrc`，不会覆盖旧文件。
- JSON 的 `sentences` 中包含 `start_sec`/`end_sec`，并支持可选字段 `review_text`：
  - 导出 LRC 时优先使用 `review_text`
  - 若 `review_text` 为空则回退 `zh_text`
- 可在“主菜单->检查”中使用“从 JSON 路径导出 LRC”对人工校对后的 JSON 重新导出。
