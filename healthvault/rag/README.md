# 服务器知识库与局部更新

知识库由服务器维护。**更新的单位是来源文档快照**，用户不用下载整库，也不用修改检索算法。管理员可以只改一份来源中的几个章节，再提交该来源的完整快照；没有提交的来源由服务器保留。当前小型语料会重建版本化 chunk 与 FTS 索引，向量按嵌入正文和本地模型哈希缓存，未变内容可以复用。chunk/citation ID 带 corpus 版本，避免混用新旧引用。

切块由代码完成，模型不参与。新候选使用 `bounded-v2` 策略：优先按标题、段落、句子、列表和表格边界分组，再用本地 embedding 模型的 tokenizer 核对预算。超长单句或表格行继续按字符窗口拆分；正文片段当前采用零重叠，必要表头在预算允许时重复。零重叠简化了连续原文覆盖和长度限制，没有证明它是最佳检索策略；跨边界上下文仍需通过评测验证。原文不截断，字符位置和降级原因写入 `boundary-spans.json`。没有配置 tokenizer 时使用保守 UTF-8 字节预算，并明确标记为 fallback，不能称为实际 token 数。旧语料与冻结评测继续使用原策略。

没有匹配向量索引的版本使用 SQLite FTS 关键词检索和主题别名；构建了匹配索引的版本使用 BGE-M3 语义检索、FTS 与 RRF 融合。问答路径可将问题改写成检索词，但回答仍以原问题和检索证据为依据。检索命中不等于证据充分：本地证据不足、用户允许联网时，才访问官方白名单来源；模型回答还要通过引用、数值和语义支持检查。

## 更新一份知识文档

管理员在“工作流管理 → 知识版本管理”选择来源，下载该来源的更新模板。改动模板的 `sections`，填写实际文档版本、抓取日期及有来源证明的发布日期/更新日期，并调整检索探针。**提交一份来源的完整快照，不提交模型推测的医学补丁。** `source_sha256` 保持 `auto` 时，服务器按规范化正文计算哈希。

上传 JSON 后先生成待审核版本，查看章节 diff、来源日期、变更来源和检索探针。需要语义检索时先点击“构建语义索引”；面板显示实际复用/新嵌入的 chunk 数。核对来源内容与评测后，管理员批准才切换生效版本，也可以拒绝、观察或回滚。正文哈希、模型哈希、chunk IDs、维度和 FAISS 文件哈希必须与候选一致，否则不能把旧索引挂到新语料。

## 来源与更新文件

先从获批官方来源取得新文件或网页，核对文档身份、发布日期、更新日期和复用许可，再整理为 `sources/*.json` 的现有规范化格式。网页规范化可参考 `acquire_official_sources.py`，它按真实标题路径和段落、列表、表格组织内容。新来源或标题发生变化时需要同步核对标题白名单。程序不自动合并医学内容，不从抓取日期推断发布日期。

管理员准备更新 JSON，`source_updates` 只放需要替换或新增的规范化来源。每个改动来源提供一个 `query` 与 `expected_source_id` 检索探针；可选 `aliases` 用于同时更新主题别名。示例结构如下，`source_updates` 中放完整的来源快照对象：

```json
{
  "corpus_version": "r3-2026-10-01-v1",
  "source_updates": [],
  "source_metadata": {
    "niddk-a1c-test": {
      "document_kind": "general_health",
      "published_on": null,
      "updated_on": "2026-09-30"
    }
  },
  "probes": [
    {"query": "A1C blood test", "expected_source_id": "niddk-a1c-test"}
  ]
}
```

示例日期只演示格式，实际更新必须填写来源证据提供的日期；未知日期用 `null`。`formal_guideline` 需要发布日期和正式文档版本，`retrieved-*` 抓取标签不能充当正式版本，通用健康解释继续使用 `general_health`。当前“最新指南”问答拒答规则保留，不能因为录入了一份指南就声称已验证它是当前最新版。

快照的 `source_sha256` 是规范化 `sections` 的 UTF-8 JSON 哈希，序列化参数为 `ensure_ascii=False, sort_keys=True, separators=(",", ":")`。正文变化必须同步更新哈希，也可以填写 `auto` 交给服务器计算。日期、版本等元数据保存在 `provenance.json`，整个发布版本另有哈希。旧快照、旧 corpus 和历史评测不覆盖。

可以在管理员的“工作流与知识管理”页面上传更新 JSON，或从项目根目录离线生成候选：

```powershell
& .\.venv\Scripts\python.exe -B -m healthvault.rag.releases --file PATH_TO_UPDATE.json
```

候选先进入待审核区，来源域名/机构、schema、哈希、切块边界和检索探针通过后仍需管理员人工审核 diff、来源内容和日期，点击批准才生效。发布后重新构建网页的知识 Repository；后续 Web Agent 请求、新启动 MCP 与重启后的服务读取同一发布版本。回滚恢复上一版语料与索引选择。

## 本地模型与定期巡检

服务器先配置已经下载好的本地模型；命令验证 tokenizer 和模型文件，不下载远程模型，也不调用付费服务：

```powershell
& .\.venv\Scripts\python.exe -B -m healthvault.rag.configure_model --model-path C:/models/bge-m3
```

模型配置和 tokenizer 存在忽略 Git 的 `.bear/knowledge-releases/`；也支持现有 `HEALTHVAULT_HYBRID_MODEL_PATH`。首次构建需要生成全部缓存，后续只生成正文或模型发生变化的向量，但仍重新建立同版 FAISS 映射。没有配置模型时可以审核发布 lexical 版本；系统不会冒用历史向量索引。

Web 服务运行时每 24 小时巡检当前来源，也可以在管理页手动检查。巡检使用条件请求、官方域名和每次重定向的公网地址校验，变化只保存为待比对原始快照和原文。首次抓取也要求比对，不能把未知网页直接覆盖生产知识；管理员仍通过更新模板规范化并审核。进程停止期间不会巡检，也没有业务 Worker/Checkpoint。

官方白名单覆盖 MedlinePlus、NIDDK、NHLBI、ODPHP、CDC、WHO 和 NICE；当前已收录的文档来自前四者。联网问答优先访问与本地主题相关的官方来源，再使用 MedlinePlus 服务补充，默认不联网。**当前无法证明某份正式文件是最新版时，“最新指南”请求继续拒答**，不能仅凭抓取时间或录入时间声称最新。

检索探针只验证指定来源能被召回；回答检查和合成回归不等于正式医学准确率或 held-out A/B。正式评测应固定语料、输入和实现版本，结果单独保存在本地。

## 已有语料启用向量与问题重写

安装 BGE-M3 不会自动生成旧文档向量。已有不可变语料需要对应的 FAISS 文件、chunk ID 映射、语料哈希和模型哈希；验证通过后才能绑定。向量索引由本地构建，公开仓库只提供知识正文、别名和切块快照。可在项目根目录执行：

```powershell
& .\.venv\Scripts\python.exe -B -m healthvault.rag.retrieval_config --index-directory PATH_TO_MATCHING_VECTOR_INDEX
```

命令先验证文件哈希、模型身份和实际 FAISS 维度/映射，再保存服务器配置。Web 与新启动的外部 MCP 读取同一选择；`HEALTHVAULT_KNOWLEDGE_MODE=lexical` 可显式覆盖默认语料。运行中的旧服务需要重新加载配置。BGE 模型和向量索引由服务器加载，用户浏览器不下载模型。FTS5 关键词索引当前在 Repository 初始化时建于内存，向量索引预先保存；查询时仍需生成问题向量并计算排名。

共享入口为 `healthvault/application/query_rewrite.py`。它先检查支持范围，从本问题或上一用户问题中解析单一已知指标；缺上下文或多对象时要求澄清，不猜患者事实。知识 Web 前端在模型同意后请求可选模型整理查询，代码核对已知主题、数值、时间、侧别和否定标记，失败则使用规则结果。接口的 `allow_query_rewrite` 默认关闭，便于程序调用明确选择；就诊与 Agent 路径共用规则解析，原生检索工具也用同一组件。当前不是对所有复杂指代都有效的通用对话解析器。

`context_question` 只提供指标名称，不能指定用户归属或补充报告事实。`question` 保留原问题，`resolved_question` 给后续明确指代使用；回答和复核同时收到原问题及必要解析。模型提出的 `search_query` 仅用于检索，不会写入档案。规则检查不等于证明所有改写都语义等价，最终证据复核仍保留。新增改写不并入既有冻结 baseline 实验。
