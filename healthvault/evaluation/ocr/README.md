# 合成报告 OCR 评测

评测器用固定 seed 生成合成报告和字段答案，再通过 RapidOCR 与报告预览流程检查识别及抽取结果。输入覆盖日期、报告编号和部分常见检验指标；这些固定模板不能代表真实医院版式、手写报告或医学准确率。

安装额外依赖后，从项目根目录运行：

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-ocr-eval.txt
& .\.venv\Scripts\python.exe -B -m healthvault.evaluation.ocr.runner --output .task_outputs/ocr-evaluation --seed 46
& .\.venv\Scripts\python.exe -B -m unittest -v tests.healthvault.test_task_046_ocr_eval
```

输出目录保存生成图片、字段答案、OCR 文本、抽取结果、差异和运行配置。使用新的目录保存每次运行，避免覆盖旧结果。运行依赖本地 OCR 模型，不调用付费语言模型；首次安装 OCR 组件时可能需要取得对应模型文件。
