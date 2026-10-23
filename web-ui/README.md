# HealthVault Web

React + TypeScript front end for the local HealthVault demo. It replaces Streamlit as the primary interface while reusing the existing application services through the FastAPI adapter.

From the project root, start the API (see [`healthvault/web/README.md`](../healthvault/web/README.md)):

```powershell
& .\.venv\Scripts\python.exe -B -m uvicorn healthvault.web.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

In a second terminal:

```powershell
cd web-ui
npm ci
npm run dev
```

Open <http://127.0.0.1:5173/>. Vite proxies `/api` to the local backend. Use `npm run build` to check the production bundle.

In the restricted Windows sandbox, use `npm run build -- --configLoader runner`
if esbuild's config bundler cannot access an ancestor directory. The Vite runner
loads the same project configuration without requiring broader filesystem access.

Register a local account or sign in before using the app. Each account receives a server-generated profile ID. Records, report files, drafts, dialogue, and answer history are checked against the signed-in account. Earlier `demo-profile` data is not assigned to a new account automatically.

The interface accepts ordinary report text, PNG/JPG images, and PDFs. Images use local OCR; PDFs use selectable text or local page OCR. Free-form text is sent to the existing configured model for candidate extraction only after the user checks the consent box. A server check accepts only fields grounded in the report text; the user must review the original, values, units, dates, and evidence before confirmation. The 60-item catalog is a set of supported numeric mappings, not a guarantee that every field in every report is captured. Unsupported or unverified content remains in the source text. Reports longer than 50,000 characters must be split.

Confirmed nodule and fibroid dimensions retain the original size and unit plus a millimeter conversion. The trends page lists imaging findings by report date; it does not draw a line between lesions that may be different. Numeric indicators retain selectable line charts and full data tables. Source IDs and report text stay collapsed until opened for review. The **报告原件** page opens the original uploaded PNG/JPEG/PDF and confirmed text for both laboratory and imaging reports. Text-only and earlier records have no retrievable file.

The Agent task page uses the existing model key, persistent profile-bound conversation records and session history, manual and automatic session compaction, and the native read-only ToolRegistry. MCP remains an optional external adapter. Reopen the same task ID to continue after a page or server restart. This is conversation recovery; business-stage Checkpoint and unattended Workers remain deferred. The developer's allowed global Skill and the signed-in user's own verified workflow are injected explicitly. Generic global memory, unrestricted Skill discovery, and unrestricted MCP discovery remain disabled. Knowledge and visit preparation retain separate question forms.

Drafts and original files are persisted before OCR/extraction and can be reopened after restart, including failed or unstructured uploads. Confirmed records, accounts, Agent task history, and supported answer history are stored in local SQLite. Evidence controls show original spans, field relationships and image/PDF regions when available. The configured hosted model receives free-form report text for extraction and questions with necessary evidence for answers only after explicit consent. Knowledge/visit answers undergo citation, numeric and semantic support checks before saving. These checks do not establish clinical accuracy. Knowledge and visit pages can reopen saved answers and citations. This is a local multi-user prototype: keep it bound to `127.0.0.1`. Internet-facing deployment still requires TLS, rate limiting, protected backups and storage, and a security review.

“我的工作流” shows private candidates and optional manual review; verified candidates activate only for that user by default and support rollback. Administrators also see redacted learning records, opt-in shared proposals and project Skill publication management. Global project Skills remain developer-owned. The knowledge version panel downloads single-source update templates, observes official-source changes and builds matching local vectors with content-hash reuse; publication still requires administrator review. See the backend and RAG READMEs for configuration and update steps.

The **健康日历** page shows confirmed report follow-up plans, source quotes,
pending dates and scheduled events. Users choose or confirm the date before
activation and can edit, complete or cancel it. These are local in-app reminders.
The workflow page displays private-version usage observations and supports manual
disable. Before clearing or replacing unsaved report edits, the page offers
继续编辑 or 放弃编辑并继续; cancelling retains the text.
