---
name: cumcm-paper-writing
description: 生成符合全国大学生数学建模竞赛（高教社杯 / CUMCM）全国一等奖排版标准的 LaTeX 论文。适用于从赛题、项目代码、实验数据与结果出发撰写完整数模论文，或对已有论文按国赛规范排版成稿。触发词包括“写国赛论文”“生成数模论文”“把实验结果写成论文”“数学建模竞赛论文”“国赛 LaTeX 论文”“CUMCM paper”等。
---

# CUMCM 国赛论文写作 Workflow

产出一篇数据真实、结构完整、符合国赛格式规范与一等奖通行排版（A4、页边距 2.5 cm、摘要页起连续编页码、正文小四宋体、题目三号黑体、一级标题四号黑体居中、二级/三级小四黑体、图表题五号黑体、无页眉、正文不设目录、附录含全部源程序）的 LaTeX 论文。

## 工作流程

### 1. 盘点项目事实
- 通读赛题 PDF、数据说明、`src/` 代码、训练日志（如 `runs/*/results.csv`）、评估输出（JSON/CSV）与既有方案文档。
- 建立「真实数字清单」：数据规模与类别分布、逐类与整体指标（mAP@0.5、mAP@0.5:0.95、P/R/F1）、模型参数量/FLOPs/推理速度、提交文件统计。
- 严格区分「已做实验」与「未做实验」：未验证的改进只写设计/展望，绝不编造结果；鲁棒性等未执行项标注为建议协议。

### 2. 确定三个问题的建模与结果口径
- 问题 1（判别/分类）：无负样本时优先开放集思路，如用检测最大置信度拟合 Weibull 分布（极值理论），给出分布参数、最优阈值与判别统计。
- 问题 2（检测/定位）：YOLO 系模型 + 针对性数据策略（类别不平衡用 Copy-Paste、无负样本用无标注区域裁剪、光照/尺度增强），给出验证集最优指标与逐类指标。
- 问题 3（评估）：六维度框架——检测精度、分类性能、消融实验、鲁棒性、推理效率、错误分析；每个维度都要有真实依据或明确标注为待做。

### 3. 搭建论文工程
- 将 `assets/latex-template/` 复制为论文目录（如 `论文/`），沿用其版式配置（`main.tex` 已按国赛规范设置）。
- 按 `sections/` 骨架逐章填写；图用 matplotlib 生成时设置 `plt.rcParams['font.sans-serif']=['SimHei']` 与 `axes.unicode_minus=False`。
- 附录用 `\lstinputlisting` 引用全部源程序副本；源码注释含 Unicode 数学符号（∈ ≈ σ →）时先替换为 ASCII，避免 listings 缺字形。
- 电子版保持 `\paperversionfalse`（第一页为摘要页）；纸质版改 `\paperversiontrue` 并核对当年官方承诺书文本。

### 4. 编译与验证（门禁）
- 运行 `build.ps1`（`xelatex main.tex` 两次）。
- 运行 `scripts/verify_pdf.py 论文/main.pdf`，检查：A4 页面、页码、无 `??`、摘要单页含关键词、字体嵌入、无未定义引用。
- 修复所有 LaTeX Error、Overfull hbox、Missing character；二次编译后 undefined reference 必须清零。
- 摘要必须单页；正文尽量 ≤ 20–25 页；附录页数不限；图表/公式编号在文中全部被引用。

### 5. 交付前核对
- 输出文件格式与赛题要求一致（如 CSV 的 `image_id` 是否含扩展名、列名与顺序）。
- 正文与附录不得出现身份/学校信息；参考文献按 GB/T 7714 著录并与正文 `\cite` 一一对应。

## 参考文件（按需读取）
- `references/format-spec.md`：国赛格式规范原文要点、本模板排版约定、编译环境与常见坑。
- `references/writing-checklist.md`：摘要、结构、数字一致性、诚实性、图表与门禁的逐项检查清单。

## 资源
- `assets/latex-template/`：可直接编译的论文骨架（main.tex + pages + sections + refs.tex + build.ps1）。
- `scripts/verify_pdf.py`：编译后的 PDF 门禁检查脚本（PyMuPDF，用法见脚本 docstring）。
