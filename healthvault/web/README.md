# HealthVault local Web API

This adapter is for the **local synthetic-data demo**. It reuses
`HealthVaultUIController` for OCR, preview, confirmation, timeline, comparison,
knowledge search, and visit briefs. The two question-answering endpoints call
the OpenAI-compatible model service configured in the existing `.env`.

From the project root on Windows:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
& .\.venv\Scripts\python.exe -m uvicorn healthvault.web.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open `http://127.0.0.1:8000/docs` to inspect and try the API. The database
defaults to `.task_outputs/healthvault-ui/healthvault.sqlite3`; set
`HEALTHVAULT_UI_DB_PATH` to use another local database.

The intended request sequence is:

1. Register or log in at `/api/auth/register` or `/api/auth/login`. The server sets an HttpOnly session cookie and returns the account's profile ID and CSRF token. Send the token as `X-CSRF-Token` for mutations. `GET /api/auth/me` restores a session; `POST /api/auth/logout` ends it.
2. `POST /api/drafts/text` with JSON `{profile_id, source_id, report_text, auto_extract:true, model_consent:true}`,
   or `POST /api/drafts/image` or `POST /api/drafts/pdf` with multipart `profile_id` and `file`.
3. For images and PDFs, compare extracted text with the original file at the returned
   `image_url`. Then `PUT /api/drafts/{token}` with
   `{profile_id, report_text, reviewed:true, model_consent:true}` when model extraction is desired. Text edits use the same endpoint.
   Each update invalidates the old token and returns a new preview/hash.
4. Show `preview.indicators`, `preview.imaging_findings`, `preview.clinical_items`, `preview.issues`, and
   `preview.warnings` if present. Do not enable confirmation unless
   `preview.can_confirm` and OCR review is complete.
5. On an explicit confirmation action, `POST /api/drafts/{token}/confirm` with
   `{profile_id, report_text, preview_sha256, acknowledged:true}`. The server
   checks the profile, source text, token, and preview hash. For ordinary
   reports, it also rebuilds the evidence-checked extraction before persistence.
6. Refresh `/api/profiles/{profile_id}/timeline` and
   `/api/profiles/{profile_id}/comparisons`. Knowledge and brief read endpoints
   are `/api/knowledge?query=...` and
   `/api/profiles/{profile_id}/visit-brief`.
7. Ask a question with `POST /api/answers/knowledge` or
   `POST /api/profiles/{profile_id}/answers/visit`, JSON `{ "question": "...", "model_consent": true }`.
   The first retrieves curated knowledge; set `allow_web_search:true` on that request to permit official-domain fallback when local retrieval or the local-evidence answer is insufficient. It defaults to false. The second uses only confirmed records
   from the selected profile and their citations. Both return an answer and the
   source cards used to ground it. A separate verifier checks semantic support and question relevance; code checks exact quoted spans, citations and literal numbers. Only supported answers are stored at `/api/profiles/{profile_id}/answers/history`. Missing evidence returns
   `insufficient_evidence` without calling the model.

The API reads `HEALTHVAULT_OPENAI_API_KEY`, `HEALTHVAULT_OPENAI_BASE_URL`, and
`HEALTHVAULT_OPENAI_MODEL` from the process environment or the existing local
`.env` (falling back to the corresponding `OPENAI_*` names). It never sends the
key to the browser. Asking a question transmits the question and selected
knowledge excerpts or confirmed-record evidence to the configured model
service. This is a hosted service with the current configuration, not a model
running on this computer. Local tests mock the transport and make no paid calls.

Optional `HEALTHVAULT_OPENAI_REASONING_EFFORT` is forwarded to the compatible
provider; leave it unset to retain the provider default. See the
[DeepSeek switch instructions](DEEPSEEK_SETUP.md) for the exact local configuration
and restart steps. Frozen evaluation configurations do not follow Web settings.

The model's internal answer protocol is a JSON status with individually cited
claims. The UI still displays Chinese prose. The verifier must cover every
displayed fact and bind a source quote; omitted facts fail verification. Only
the verified rendering is displayed and stored. Citation validity and literal
checks do not establish medical accuracy.

Natural reports also extract explicitly labelled diagnoses, advice and follow-up
plans using the existing local clinical-item rules. These remain review candidates
until the report is confirmed. `GET /api/profiles/{profile_id}/calendar` synchronizes
confirmed follow-up plans into pending events. Users activate, edit, complete or
cancel events through the profile-scoped calendar endpoints. A plan with an
ambiguous date is shown in `unresolved_plans`; the user supplies and acknowledges
a date at `POST /api/profiles/{profile_id}/calendar/plans/{clinical_item_id}/schedule`.
The selected calendar date does not rewrite the report's original date expression.
Events are local in-app reminders; no external notification service is involved.

Ordinary reports use the configured model to propose fields, then server-side
checks verify exact source lines, numbers, units, and report date. Known numeric
indicators require an evidence-backed reference range. Imaging nodules and
uterine fibroids keep original dimensions and units; cm is converted to mm
while the source wording remains available. Unverified candidates become review
warnings and do not enter structured records. PDF input is limited to 8 MB and
12 pages; text over 50,000 characters must be split. These checks cannot
guarantee complete extraction or clinical correctness for every report.

Interactive Agent tasks use:

- `GET /api/profiles/{profile_id}/agent/capabilities`
- `GET /api/profiles/{profile_id}/agent/tasks` and `GET /api/profiles/{profile_id}/agent/tasks/{task_id}`
- `POST /api/profiles/{profile_id}/agent/tasks/{task_id}/messages` with
  `{prompt, expected_version, model_consent:true, allow_web_search:false}`
- `POST /api/profiles/{profile_id}/agent/tasks/{task_id}/compact` with
  `{expected_version, model_consent:true}`

The web Agent uses profile-bound task recovery records and session history,
automatic or manual session compaction, and approved read-only tools. Official
web search is off by default for each message and accesses whitelisted official
documents and the NLM service only when `allow_web_search:true`; the other tools use local data. The
generic MCP loader, global memory, and global Skills are not enabled. The task
ID resumes an interactive conversation after restarts. This is not a durable
business checkpoint: current stage, next action, artifacts, Skill version,
idempotency keys, and exact resume step are not reliably saved. There is no
unattended background scheduler.

The web conversation now uses the native ToolRegistry without launching an MCP
child process. MCP remains an optional read-only adapter for external clients.
The developer's `HealthVault-Task-Execution` Skill is loaded from `.bear/skills/`.
Each authenticated user's personal workflow is stored separately under
`.bear/skill-evolution/healthvault/personal/` and loaded only for that user.
Developer safety rules remain active. Task-end traces store tool names, argument
types, statuses and version hashes without report contents or patient values.
Eligible evidence creates a bounded private patch. Static safety checks and six
fixed regression/held-out workflow-plan probes must pass against the current
and candidate versions before publication. These probes **do not execute the
medical task or measure clinical accuracy**. Failure or unavailable evaluation
keeps the candidate inactive.

Private candidates default to automatic activation after verification. The user
can enable optional manual review, inspect the diff/evaluation/redacted traces,
and roll back. `GET /api/skills`, `PUT /api/skills/settings`, the private
`/api/skills/candidates/{id}/decision` and `/share` endpoints, and
`POST /api/skills/rollback` are bound to the authenticated account. Sharing is
opt-in; administrator approval marks an experience for developer review and
does not automatically rewrite the project's global Skill.

Registered accounts have no administrative role, including accounts named
`admin`. The server owner grants a role to an existing account locally:

```powershell
& .\.venv\Scripts\python.exe -m healthvault.web.auth --database .task_outputs/healthvault-ui/healthvault-auth.sqlite3 --username YOUR_ACCOUNT
```

Add `--revoke` to revoke the role. Refresh the signed-in page after provisioning.
The “我的工作流” page shows the signed-in user's own learning records. The
administrator's “工作流管理” page also shows redacted personal records, shared
proposals and project-level publication management. Ordinary users cannot read
another user's traces or candidates.

`GET /api/skills` also returns private usage observations for the actual injected
version. Task IDs are deduplicated, provider failures are excluded, and statistics
measure workflow contracts rather than clinical correctness. Sustained contract
failures or a safety hard failure stop that version and attempt rollback; a
disabled hash is not re-injected after restart. `POST /api/skills/disable` lets the
owner stop the private workflow manually, preserving its history and the
developer's project Skill. Candidate generation remains separate from usage
observation.

Report intake lives in a profile-owned `*-uploads.sqlite3` database. Originals
are saved before OCR or extraction, including failed and unstructured reports.
`GET /api/profiles/{profile_id}/uploads` lists unconfirmed intake;
`GET /api/drafts/{token}?profile_id=...` reopens it after a restart. Updating a
draft still invalidates its old token. The earliest uploaded/OCR text is retained
separately from edited text and is also available in confirmed report details.
Confirmed intake cannot be overwritten through the draft-edit route.
Saving original text or running local extraction
rules needs no model consent; model extraction continues to require consent.
Common catalogue rows use deterministic name/value/unit/range checks first.
Unstructured lines appear under `other_findings` with source line positions.
Field evidence includes character spans and, where available, page/row/column,
block ID and polygons, including value/unit/range/date relationships. Locators
survive confirmation and appear in timeline/brief evidence. The frontend can
highlight the original image or `/api/drafts/{token}/pages/{page}` PDF region.
Image processing corrects EXIF orientation, crops blank margins, deskews and
denoises; quality flags remain visible. Mixed PDFs preserve native text while
OCR also examines uncovered embedded-image regions. Failed regions retain the
original file and page warnings. These features do not remove report review.

Administrators also have a knowledge version panel. Import a normalized JSON
source update, inspect its diff, dates and retrieval probes, then approve,
reject, watch or roll back. The default release root is
`.bear/knowledge-releases/`; `HEALTHVAULT_KNOWLEDGE_RELEASE_ROOT` overrides it.
The panel can download a single-source update template, check source changes and
build a matching local BGE-M3/FAISS index with content-hash embedding reuse.
Versions without vectors use lexical retrieval; versions with validated vectors
use hybrid retrieval. They never attach a previous corpus's vector index. New web Agent requests
and newly started external MCP processes use the published corpus. See
`healthvault/rag/README.md` for the update file format and offline command.
Official-source changes are checked every 24 hours while the Web process runs
and enter pending snapshots for administrator comparison, never automatic
medical-content publication. A dated document alone does not prove it is the
latest guideline; unverified latest-guideline requests remain refused.

**Local multi-user prototype only:** the session cookie, per-account profile checks,
CSRF token, and SameSite policy block the earlier arbitrary-profile access. Old
`demo-profile` rows are not attached to any new user. Keep the server bound to
`127.0.0.1`; no permissive CORS is configured. Internet-facing deployment still
requires HTTPS, login rate limiting, protected storage and backups, and a review
of uploaded and model-bound health-data handling. Report extraction and answer
generation now use the model-only Harness Agent with no tools. Knowledge answers
use local RAG first, then the approved MedlinePlus read-only tool when the local
evidence is insufficient. The official tool sends only generic catalogued topics;
current clinical-guideline questions remain unsupported without dated sources.

网页知识、就诊和持续 Agent 的事实回答均核对自己的原文引用与问题要点。要点清单在生成前只读取原问题和已验证指代，保留明确对象、用途、时间等要求；开放问题由独立无工具任务补充，不能读取已有回答或患者记录。复核必须逐项返回固定清单，不能省掉要点后宣称完整。复杂含义与隐含要点仍需模型判断，结构检查不等于医学正确率。

知识问答复核记录“未找到证据”时，应用最多补搜两个要点；新增来源才允许一次重新生成和复核。已找到但漏答不会扩大搜索，同编号正文冲突明确失败。普通 Agent 保持自身工具预算，复核失败不会保存为成功对话或合格学习 Trace。单次网页操作可能包含问题规划、生成与复核，补检索恢复时再增加一次生成/复核；用户模型同意覆盖这些受限任务。Agent 的 tokens 仍是主循环用量，各无工具任务单独记录日志，不把它当全部网页用量。
