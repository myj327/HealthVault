# 切换到 DeepSeek

先在 `BearCode` 目录操作。密钥由你在本机填写，不发到聊天里。本地配置检查不会调用真实模型。

打开项目根目录的 `.env`，修改同名配置；没有的才新增，避免重复定义。用 DeepSeek 平台生成的密钥替换下方占位文字：

```dotenv
HEALTHVAULT_OPENAI_API_KEY=在这里填写你的DeepSeek密钥
HEALTHVAULT_OPENAI_BASE_URL=https://api.deepseek.com
HEALTHVAULT_OPENAI_MODEL=deepseek-flash
HEALTHVAULT_OPENAI_REASONING_EFFORT=high
```

示例使用 `deepseek-flash/high`，接口兼容 OpenAI Chat Completions；上面的变量名表示接口格式。DeepSeek 的 `low`、`high` 都开启思考，`none` 表示关闭。配置检查只证明项目读取到设置，不证明模型已连通或质量提高。[官方接口](https://api-docs.deepseek.com/)、[参数说明](https://api-docs.deepseek.com/api/create-chat-completion/)

先执行下面的本地配置检查。它只读配置，不联网，仅显示端点、模型、推理强度；不会输出密钥：

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B -c "from healthvault.web.model_runtime import configured_web_model; c=configured_web_model(); print('endpoint:', c.api_base); print('model:', c.model); print('reasoning:', c.reasoning_effort)"
```

预期模型是 `deepseek-flash`、推理强度是 `high`。如果仍显示旧配置，检查启动后端的终端是否设置了同名环境变量；进程环境优先于 `.env`。不要贴出整个 `.env` 排查。

在原后端终端按 `Ctrl+C` 停止服务，再用原启动命令启动，保留之前的端口、数据库和检索配置。例如原服务使用 8000 端口：

```powershell
& .\.venv\Scripts\python.exe -B -m uvicorn healthvault.web.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

重新提问才会调用新模型并产生供应商用量；旧的历史回答仍是旧结果。网页中的回答、复核、模型抽取和普通 Agent 都使用这套配置，当前没有按角色分别配置模型。切换回答模型不需要重新安装 BGE-M3，也不需要重建本地向量索引。

运行时支持流式 reasoning_content 的接收及历史回传，它只作为下一轮供应商请求的字段，不混入显示正文、工具参数或最终 JSON。该字段保证思考模式工具调用协议兼容，不能证明语义复核、工具选择或回答完整性更准确。[思考模式要求](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)

正式评测使用独立的固定模型配置和输入，不随 Web 的 `.env` 自动切换；结果另存本地。
