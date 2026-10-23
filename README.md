# HealthVault

面向个人纵向健康档案与就诊准备的 Agent 应用。

把文字、图片和 PDF 报告整理为有原文证据的健康档案，连接历年指标比较、官方知识检索、就诊准备和健康日历；通过 harness 管理模型、工具、会话与权限，并将可复用的任务经验沉淀为个人 Skill。

[项目预览](#项目预览) · [核心功能](#核心功能) · [系统流程](#系统流程) · [本地部署](#本地部署) · [Demo 演示](#demo-演示) · [常见问题](#常见问题)

## 项目预览

登录后，用户可以在「报告核对」中上传原件或粘贴文字，对照证据核对候选字段，再明确确认保存。「健康概览」「指标趋势」「报告原件」展示已确认档案；「知识问答」「就诊准备」「持续对话」提供带来源的模型交互；「健康日历」管理待确认复诊计划和个人日程；个人工作流页面展示 Skill 候选、版本和使用观察。

**基础 Demo 使用仓库内合成数据，报告确认、趋势、原件和日历可以不调用模型。** 模型抽取、知识问答、就诊准备问答及持续对话需要配置模型，并在页面同意发送必要资料。演示步骤和预期结果见下文。

本项目用于健康档案整理和健康教育，不提供诊断、处方或治疗建议。示例报告中的数值、诊断和计划均为合成内容。

## 核心功能

### 报告导入与纵向档案

支持文字、PNG/JPG 和 PDF。图片使用本地 RapidOCR；PDF 提取原生文字，并对需要识别的图像区域进行 OCR。原件和草稿先保存，识别失败或暂时不能结构化的内容可以重新打开核对。

常见指标由规则提取，复杂字段可交给模型提出候选；代码检查字段与原文、数值、单位、参考范围和日期的对应关系。用户确认后才写入正式档案。数值指标按时间展示，结节和肌瘤保留原始尺寸及单位，并提供毫米换算。报告中明示的医生建议和复诊计划可进入后续核对与日历流程。

### 官方知识检索与有据回答

知识库保存版本化的官方资料和稳定引用。基础检索使用 SQLite FTS5/BM25；配置匹配的本地 BGE-M3 与 FAISS 索引后，可使用关键词和语义检索的 RRF 融合。语料、索引和模型需要匹配，缺少可用向量时保留关键词检索。

问答根据检索证据生成内容，再核对引用、原文片段、数值及语义支持关系，展示有依据的内容和未回答部分。用户允许联网后，可在本地证据不足时访问官方白名单来源。管理员可以审核单份来源更新、构建匹配索引、发布或回滚知识版本。

### harness 与持续对话

harness 统一模型协议、Agent 循环、工具执行、会话保存与上下文压缩。Web Agent 通过原生工具注册表调用只读领域能力；MCP 保留为外部客户端的可选接入方式。

账户身份由服务端绑定，模型不能自行指定其他用户的档案。会话按用户和任务 ID 保存，刷新页面或重启后可重新打开同一个任务继续对话。报告确认写入走独立的用户确认接口；当前任务恢复覆盖对话历史，业务级 Checkpoint 和无人值守 Worker 仍待实现。

### 个人 Skill 可控自进化

从脱敏任务轨迹、工具结果和明确反馈中提取工作流经验，生成有范围限制的 Skill 修改。候选检查通过后默认仅对本人发布，用户也可以开启人工审核；支持查看差异、使用观察、手动下线和回滚。

Skill 记录工具选择、步骤顺序、停止条件和输出组织。医疗事实、知识库内容及工具权限由各自的业务模块管理。固定工作流探针检查执行计划，不等同于真实任务成功率或医学准确率；实际效果另用任务评测验证。

## 系统流程

~~~mermaid
flowchart TB
    UI["React Web"] --> API["FastAPI<br/>认证与用户隔离"]
    API --> APP["健康业务<br/>导入与确认 · 趋势 · 日历<br/>知识问答 · 就诊准备"]
    APP --> RECORDS[("SQLite 档案<br/>报告原件与证据")]
    APP --> RAG["官方知识库 / RAG<br/>关键词 · 可选混合检索"]
    API --> HARNESS["harness<br/>模型调用 · Agent 循环<br/>工具 · 会话 · 权限"]
    APP -.->|"抽取 / 生成 / 复核"| HARNESS
    HARNESS -->|"只读领域工具"| APP
    HARNESS -.->|"任务轨迹 / 反馈"| SKILL["个人 Skill<br/>候选 → 验证 → 生效<br/>观察 · 下线 · 回滚"]
    SKILL -.->|"本人工作流程"| HARNESS
~~~

几个关键检查可以直接由代码完成：用户身份与登录会话比较；确认时重新计算预览哈希并检查是否与用户看到的一致；数值和单位与绑定的原文位置比较；引用 ID 与本次实际提供的来源集合比较。这些检查依赖明确的数据关系。复杂语义是否受到证据支持，还需要受约束的模型复核与人工结果检查。

## 技术栈

| 层次 | 技术 | 用途 |
| --- | --- | --- |
| Web | React 19、TypeScript、Vite 7 | 报告核对、趋势、问答、日历与持续对话 |
| API 与运行时 | Python 3.12、FastAPI、AsyncIO、harness | 接口、模型调用、工具执行、会话与权限 |
| 档案与检索 | SQLite、FTS5、BM25/trigram | 账户、档案、草稿、历史和关键词检索 |
| 文档处理 | RapidOCR 3.9.2、ONNX Runtime、PyMuPDF | 本地 OCR、PDF 文字与页面证据 |
| 可选语义检索 | BGE-M3、FAISS、RRF | 本地向量召回与混合排序 |
| 模型与扩展 | 可配置的 Chat Completions 兼容服务、Skills、MCP | 模型推理、个人工作流与外部工具接入 |

## 本地部署

以下以 **Windows PowerShell** 为例，所有根目录命令都在同时包含 `healthvault/`、`web-ui/` 和 `requirements-web.txt` 的目录执行。前后端分别使用两个终端。

### 环境要求

| 组件 | 要求 |
| --- | --- |
| Python | 3.12，SQLite 支持 FTS5 和 trigram tokenizer |
| Node.js | 22.12 或更高版本；当前前端已在 Node.js 24 构建验证 |
| npm | 随 Node.js 安装，使用仓库的 package-lock.json |
| 模型服务 | 基础离线 Demo 可跳过；模型功能需要可用 API 配置 |

先检查环境：

~~~powershell
py -3.12 --version
node --version
npm --version
~~~

如果没有 Windows 的 `py` 启动器，可以将后文的 `py -3.12` 换成已确认版本为 3.12 的 `python`。基础 Web 和 OCR 使用 CPU 即可。

### 1. 安装后端依赖

~~~powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-web.txt
~~~

`requirements-web.txt` 已包含基础运行依赖、OCR 和 PDF 支持。

### 2. 配置模型（基础 Demo 可跳过）

在仓库根目录新建或编辑 `.env`，更新同名字段，保留已有其他配置。以下为项目已有的 DeepSeek 兼容配置示例，密钥和模型名以自己的供应商账户为准：

~~~dotenv
HEALTHVAULT_OPENAI_API_KEY=your-provider-api-key
HEALTHVAULT_OPENAI_BASE_URL=https://api.deepseek.com
HEALTHVAULT_OPENAI_MODEL=deepseek-flash
HEALTHVAULT_OPENAI_REASONING_EFFORT=high
~~~

变量名表示接口格式。使用智谱时，填写对应密钥、`https://open.bigmodel.cn/api/paas/v4` 及账户可用的 GLM 型号；推理参数按供应商支持范围设置。详细说明见 [模型配置与切换](healthvault/web/DEEPSEEK_SETUP.md)。

密钥只保存在本机，不提交到 Git，也不放进前端。进程环境变量优先于 `.env`；修改模型配置后需重启后端。模型调用会产生供应商用量，基础离线演示不需要验证真实模型连通。

### 3. 启动后端

在第一个终端、仓库根目录运行：

~~~powershell
$env:PYTHONUTF8 = "1"
$env:HEALTHVAULT_UI_DB_PATH = ".task_outputs/healthvault-demo/healthvault.sqlite3"
& ./.venv/Scripts/python.exe -B -m uvicorn healthvault.web.api:app --host 127.0.0.1 --port 8000 --no-access-log
~~~

数据库及对应账户、草稿、会话文件会自动初始化。上述路径将演示档案与默认开发数据库分开；再次使用同一路径启动，会保留此前的演示账户和数据。

另开终端检查 API 是否启动：

~~~powershell
(Invoke-RestMethod -Uri "http://127.0.0.1:8000/openapi.json").info.title
~~~

预期返回 `HealthVault local demo API`。接口文档位于 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)；业务接口需要先登录。

### 4. 启动前端

在第二个终端、仓库根目录运行：

~~~powershell
Set-Location web-ui
npm ci
npm run dev -- --strictPort
~~~

访问 [http://127.0.0.1:5173](http://127.0.0.1:5173)。Vite 将 `/api` 代理到 `http://127.0.0.1:8000`。首次注册用户名并设置至少 12 个字符的密码；再次演示可以直接登录。

如果后端换端口，需同步修改 `web-ui/vite.config.ts` 中的代理目标并重启前端。建议固定使用上述地址，避免在不同主机名之间切换造成登录状态混淆。

### 5. 构建前端并预览

在 `web-ui/` 中运行：

~~~powershell
npm run build -- --configLoader runner
npm run preview -- --port 5173 --strictPort
~~~

构建产物位于 `web-ui/dist/`。启动预览前，先在开发服务器终端按 `Ctrl+C` 释放 5173 端口；后端继续运行。预览沿用已有 API 代理，固定 5173 也与当前允许的前端来源一致。

这是本机展示部署。当前仓库的 Dockerfile 是 harness CLI 入口，Web 按以上步骤启动；公网托管还需配置 HTTPS、静态文件服务及同源 `/api` 反向代理。Vite preview 用于检查构建产物。

### 6. 可选：启用本地混合检索

新环境可以先用关键词检索完成演示。需要语义检索时，再安装可选依赖、下载本地模型并构建索引。以下命令针对仓库内置语料，在仓库根目录执行：

~~~powershell
& ./.venv/Scripts/python.exe -m pip install -r requirements-hybrid-rag.txt

# 此命令需要联网下载模型；已有完整本地目录时跳过。
& ./.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='C:/models/bge-m3')"

& ./.venv/Scripts/python.exe -B -m healthvault.rag.configure_model --model-path C:/models/bge-m3
& ./.venv/Scripts/python.exe -B -m healthvault.rag.build_hybrid_index --model-path C:/models/bge-m3 --output .task_outputs/healthvault-hybrid-index
& ./.venv/Scripts/python.exe -B -m healthvault.rag.retrieval_config --index-directory .task_outputs/healthvault-hybrid-index
~~~

完成后重启后端。模型、语料和索引校验通过后，内置语料启用混合检索。已有管理员发布的知识版本时，在管理页为该版本构建匹配索引；不要把内置语料索引挂到不同版本。知识更新和回滚见 [RAG 说明](healthvault/rag/README.md)。

### 停止、重启与数据保存

在前后端各自终端按 `Ctrl+C` 停止，重新执行相应启动命令即可恢复。停止服务不会清空数据。

| 位置 | 内容 |
| --- | --- |
| 本文配置的数据库目录 | 正式档案及日历，配套的账户、上传草稿与原件、Agent 会话、回答历史数据库 |
| .bear/skills/ | 项目发布的共通 Skill |
| .bear/skill-evolution/healthvault/personal/ | 按用户隔离的个人 Skill、候选、版本及使用记录 |
| .bear/knowledge-releases/ | 知识发布状态、版本和本地模型配置 |
| 自己指定的本地模型及索引目录 | BGE-M3 文件与向量索引 |

备份前先停止后端，保留整个数据库目录、Skill 与知识发布目录，以及实际使用的模型和索引。更换数据库路径相当于切换账户与档案空间；旧数据库中的账户和档案不会自动迁移。

## Demo 演示

### 基础 Demo：无需模型调用

**1. 注册或登录。** 打开前端，以新账户开始。每个账户绑定自己的档案，旧版演示账户的数据不会自动分配给新账户。

**2. 导入两年的报告。** 在「报告核对」中展开「粘贴报告文本」，依次粘贴下面两个文件的内容。来源 ID 分别填 `readme-checkup-2024` 和 `readme-checkup-2025`，不勾选模型发送同意，点击「生成核对预览」，核对后确认保存。每份报告保存后，再开始下一份。

在仓库根目录可用下面命令复制第一份样例；第二份替换文件名：

~~~powershell
Get-Content ./tests/fixtures/health_reports/synthetic_checkup_2024.txt -Raw -Encoding utf8 | Set-Clipboard
~~~

样例文件：

- [2024 年体检](tests/fixtures/health_reports/synthetic_checkup_2024.txt)
- [2025 年体检](tests/fixtures/health_reports/synthetic_checkup_2025.txt)

这些是本地规则可解析的固定格式合成报告。检查结果中，空腹血糖为 **2024 年 5.4 → 2025 年 6.5 mmol/L**，HbA1c 为 **5.5% → 5.9%**。这是样例原文中的数值，不代表真实患者，也不据此推断疾病。

**3. 展示趋势与证据。** 进入「指标趋势」，选择空腹血糖或 HbA1c，查看日期、数值、表格和原文来源。返回报告相关页面查看确认状态；只有已确认记录进入正式档案。

**4. 展示 OCR 与原件。** 上传 [合成报告图片](tests/fixtures/health_reports/images/synthetic_checkup_2026.png)。等待本地识别，在「核对原图与 OCR 文本」逐行核对日期、数值和单位，勾选已核对原件，再生成预览并确认。随后在「报告原件」查看已保存的原件和文字。PDF 也使用同一核对入口；固定格式样例可走本地规则，复杂报告可在明确同意后使用模型补充候选。

**5. 展示健康日历。** 再导入 [中文合成临床报告](tests/fixtures/health_reports/synthetic_clinical_zh.txt)，来源 ID 使用 `readme-clinical-2026`。确认后到「健康日历」查看报告明示的复诊计划、原文和待确认状态，核对日期后激活；也可以新建、编辑或完成个人日程。样例日期固定，可能早于演示当天，可另外创建未来日程展示提醒状态。日历安排不改写原报告的日期事实。

这条路线展示的是：**原件 → 候选 → 用户确认 → 跨年档案 → 证据 → 日程**。日历提醒在应用内展示。

### 模型 Demo：问答、持续对话与个人工作流

先完成模型配置并重启后端，再按页面要求同意发送本次问题及必要证据。以下步骤会产生模型调用。

在「知识问答」输入：

> HbA1c 检查通常需要空腹吗？请列出来源。

查看回答、来源卡片、引用及未回答提示。本地证据不足时，可以选择允许官方来源联网补充。证据检查不通过时，系统可能只回答部分内容或返回证据不足；演示时保留并解释实际结果。

在「就诊准备」输入：

> 请按时间整理我的已确认检查记录，并逐项给出原文来源。

核对回答使用的报告日期、数值和来源，并从历史记录重新打开已保存的回答。模型不会因为用户输入了一个档案编号就取得其他人的记录。

在「持续对话」中新建任务，输入：

> 比较已确认档案中空腹血糖历年变化，按日期列出数值、单位和来源。

完成后记录当前任务 ID，刷新页面或重新登录，再打开同一任务继续提问。可以展示会话历史和手动整理上下文；当前请求仍需等待执行完成，对话恢复不等同于后台任务断点续跑。

最后在任务的工作流纠正入口提交流程反馈，例如：

> 以后比较历年指标时，先列日期和数值，再统一列出引用。

到个人工作流页面查看是否形成候选、验证状态、版本差异及审核设置。候选可能因经验不足、内容重复或验证失败而未生效；展示实际状态即可。已生效版本可以手动下线或回滚，不承诺每条反馈都会生成新版本。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 前端无法连接后端 | 先检查 8000 端口的 /openapi.json；确认后端运行，Vite 的 /api 代理目标与实际端口一致 |
| 端口占用 | 停止自己此前启动的对应服务；修改后端端口时同步修改 Vite 代理。strictPort 会明确报错，避免悄悄换端口 |
| 登录失败、401 或确认请求 403 | 检查是否切换了数据库或主机名；回到固定前端地址重新登录。业务写接口使用会话 Cookie 和 CSRF token |
| 模型接口 401、模型不可用或服务未配置 | 核对供应商、端点、模型名和本机密钥，修改后重启；检查同名进程环境变量是否覆盖 .env。不要粘贴整份 .env 排查 |
| 图片/PDF 识别不全或无法确认 | 保留原件，在核对页补充文字并重新生成预览。OCR 置信度和格式通过不代表字段完整、数值准确或医学正确 |
| 没有启用语义检索 | 检查可选依赖、本地模型、当前语料版本和索引映射；缺少匹配向量时使用关键词检索 |
| Vite 配置加载或依赖解析失败 | 先在 web-ui 执行 npm ci；配置打包读取受限时可用 npm run dev -- --configLoader runner。runner 仍失败时检查运行环境对本地依赖和父目录的访问权限 |
| 日程日期已过去或未收到外部提醒 | 合成样例日期固定；应用只展示本地日历提醒。含糊复诊日期需用户补充确认 |

## 验证与评测

本地开发验证使用 mock 模型传输或纯规则路径，不消费模型 API。以下测试覆盖报告预览与确认、账户隔离、临床计划和日历；它们证明相应功能契约，不代表医学准确率：

~~~powershell
& ./.venv/Scripts/python.exe -B -m unittest -v tests.healthvault.test_web_api tests.healthvault.test_web_auth tests.healthvault.test_web_calendar
~~~

OCR、RAG、Agent 与 Skill 的评测代码及可公开的固定样例保存在 `healthvault/evaluation/`。运行结果、模型响应、调用账目和开发过程记录保存在本地，不随源码发布。固定格式合成 OCR 结果不能外推到真实医院版式；检索召回不能代替回答质量；个人 Skill 生命周期功能和任务成功率收益分别验证。

评测使用说明见 [评测入口](healthvault/evaluation/README_CURRENT.md)和 [OCR 评测](healthvault/evaluation/ocr/README.md)。部分历史实验入口需要另行准备本地输入；具体要求见各入口的 `--help`。正式模型评测使用独立配置和调用上限，不随 Web 的 `.env` 自动切换。

## 目录结构

~~~text
.
├── web-ui/                      # React + TypeScript 前端
├── healthvault/
│   ├── web/                     # FastAPI、鉴权、草稿、问答与 Agent 接口
│   ├── application/             # 导入、确认、时间线、比较、日历与证据检查
│   ├── domain/                  # 报告、指标、知识与日历领域对象
│   ├── infrastructure/          # SQLite、FTS 与混合检索实现
│   ├── ocr/                     # 本地图片识别及预处理
│   ├── rag/                     # 官方语料、切块、向量、版本与更新
│   ├── evolution/               # 个人 Skill 候选、验证、发布与使用观察
│   └── evaluation/              # 评测代码与固定样例
├── agents/                      # harness 内核
├── .bear/skills/                # 项目共通 Skill
├── tests/fixtures/health_reports/ # 可复现的合成报告和图片
├── requirements-web.txt         # Web、OCR、PDF 与基础运行依赖
└── requirements-hybrid-rag.txt  # 可选本地语义检索依赖
~~~

更细的接入说明见 [Web 前端](web-ui/README.md)、[Web API](healthvault/web/README.md)、[MCP](healthvault/mcp/README.md)及 [知识库维护](healthvault/rag/README.md)。
