# HealthVault read-only MCP bridge

The stdio server binds one synthetic profile_id at startup. Its six tools
reuse the existing ToolRegistry handlers: report preview, timeline, indicator
comparison, local knowledge search, approved MedlinePlus search, and visit brief. It never registers
confirm_health_records or generic file, shell, or browser tools.

Start from the project root with an **existing** confirmed-record SQLite
database:

~~~powershell
& .\.venv\Scripts\python.exe -B -m healthvault.mcp.server --profile-id synthetic-demo --database .task_outputs/healthvault-ui/healthvault.sqlite3
~~~

The server reads newline-delimited JSON-RPC from stdin and writes only protocol
responses to stdout. It opens the database in SQLite read-only mode. For
timeline, comparison, and visit brief calls, it injects the startup profile.
A different client-supplied profile_id is rejected.

example-config.json is an opt-in example and is **not loaded automatically**.
The approved client reads only an explicitly provided path. Its command must
resolve to the current Python interpreter; its arguments must name the exact
HealthVault server module, runtime profile, and runtime database.

~~~python
from healthvault.mcp.client import ApprovedMCPClient
from healthvault.mcp.runtime_bridge import attach_approved_mcp

client = ApprovedMCPClient.from_config_file(
    "healthvault/mcp/example-config.json",
    profile_id="synthetic-demo",
    database_path=".task_outputs/healthvault-ui/healthvault.sqlite3",
)
await client.connect()
tools = client.get_tool_definitions()
await client.disconnect_all()

# For an already built HealthVaultAgentRuntime:
async with await attach_approved_mcp(
    runtime, config_path="healthvault/mcp/example-config.json"
):
    result = await runtime.run(prompt)
~~~

ApprovedMCPClient connects only the healthvault server and exposes only the
six read-only names, even if the configuration contains other servers or the
server advertises extra tools. It does not read global or project MCP settings.
The runtime helper leaves the generic BearCode MCP manager disabled and adds
the approved prefixed tools only for the explicit attachment lifetime.
The Web application omits the official web tool from its question-answer route
and the Agent tool list unless that message explicitly sets `allow_web_search:true`. Direct MCP clients
must enforce their own user consent before calling this network-capable tool.
The Web Agent advertises only the MCP-prefixed tools during a message; its
in-process domain tools remain available to deterministic Web API routes.

This is a local synthetic-data bridge. The profile ID is a process binding,
not user authentication. Keep the process and its SQLite database on a trusted
local machine; production use requires authentication and authorization.

The official web tool checks local RAG first and sends only a catalogued generic
topic to the fixed NLM HTTPS endpoint. It does not send report text or arbitrary
user questions to the web service. Results are limited to MedlinePlus URLs and
carry a retrieval date and content hash. MedlinePlus does not provide a reliable
clinical-guideline update date in this search response, so current-guideline
questions return insufficient evidence.
The optional `local_answer_insufficient` flag permits fallback after a local
answer explicitly reports that its evidence cannot answer the question.
