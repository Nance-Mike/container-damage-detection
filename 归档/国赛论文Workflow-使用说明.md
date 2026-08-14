# 国赛论文 Workflow（cumcm-paper-writing）使用说明

## 这是什么

一套可复用的全国大学生数学建模竞赛（高教社杯 / CUMCM）论文写作工作流，产出符合国赛一等奖排版规范的 LaTeX 论文。核心是五步流程：

1. 盘点项目事实（赛题、代码、实验结果、数据规模与指标），区分"已做实验"与"未做实验"；
2. 确定三个问题的建模与结果口径；
3. 搭建论文工程（复制 `assets/latex-template/` 模板）；
4. 编译门禁（XeLaTeX 两次通过，`scripts/verify_pdf.py` 核验 PDF）；
5. 交付核对（电子版 / 纸质版页眉、页码、附录源码齐全）。

## 安装位置

技能已同时安装到两处，Antigravity 均会自动发现：

| 位置 | 作用范围 |
| --- | --- |
| `~/.gemini/antigravity/skills/cumcm-paper-writing/` | 全局，所有工作区可用 |
| `.agents/skills/cumcm-paper-writing/`（本工作区） | 仅"选题D"工作区 |

技能结构：`SKILL.md`（入口与五步工作流）、`references/`（排版规范与写作清单）、`scripts/verify_pdf.py`（PDF 门禁）、`assets/latex-template/`（完整 LaTeX 模板）。

## 在 Antigravity 中使用

1. 重启 Antigravity 或新开会话（技能列表在会话开始时加载）；
2. 直接提需求即可，例如：
   - "写国赛论文"
   - "生成数模论文"
   - "把实验结果写成论文"
   - "数学建模竞赛论文"
   - "国赛 LaTeX 论文" / "CUMCM paper"
3. 也可点名技能确保使用：在提示中写明 `cumcm-paper-writing`。

## 环境要求

- MiKTeX / TeX Live，含 `ctex`、`fontspec`、`tikz`，用 XeLaTeX 编译；
- 首次在用户环境编译前需完成 MiKTeX 初始化（若提示 "fresh TeX installation"，先运行一次 MiKTeX 设置）；
- 若 PowerShell 执行策略禁止运行 `.ps1`，可在论文目录内手动执行：
  `xelatex -interaction=nonstopmode -halt-on-error main.tex`（连跑两次）。

## 更新技能

技能源目录为 `C:\Users\administrator\.codex\skills\cumcm-paper-writing\`。修改源后，把整个文件夹重新复制覆盖到上表两处即可（`.agents` 目录带防篡改 ACL，需要管理员权限写入）。
