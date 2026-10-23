# 评测与离线验证

本目录保存 OCR、RAG、Agent 和 Skill 的评测实现，以及公开知识和合成记录组成的固定输入。运行生成的回答、评测分数、调用账目、预检文件和开发记录留在本地，不随源码发布。回归用例中的固定重放数据用于验证已知缺陷，不代表新的独立测试结果。

从项目根目录运行基础功能回归，不需要模型密钥：

```powershell
& .\.venv\Scripts\python.exe -B -m unittest -v tests.healthvault.test_web_api tests.healthvault.test_web_auth tests.healthvault.test_web_calendar
```

OCR 合成样例评测见 [OCR 使用说明](ocr/README.md)。可通过以下命令查看 RAG 入口的参数：

```powershell
& .\.venv\Scripts\python.exe -B -m healthvault.evaluation.rag --help
& .\.venv\Scripts\python.exe -B -m healthvault.evaluation.skill.benchmark --help
```

部分历史实验入口依赖未发布的本地预检文件、配对配置或模型响应，需要自行准备并显式传入对应路径；它们不能作为新安装后的默认演示步骤。依赖这些本地材料的开发测试也需要相应输入。

真实模型评测会产生供应商用量。执行前应固定模型、样本、语料、版本和调用上限，并使用新的结果目录。生成模型不能读取评分答案；人工确认状态只能由实际审核者填写。工具契约通过率、检索召回、回答质量和医学准确率应分别解释，不能相互替代。
