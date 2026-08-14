# 2026 高教社杯国赛选题 D —— 论文 LaTeX 源码

本目录为《基于YOLOv8与极值理论的集装箱破损智能检测模型》的 LaTeX 论文工程，排版遵循全国大学生数学建模竞赛论文格式规范（摘要专用页起连续编页码、正文不设目录、附录含全部源程序）。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `main.tex` | 主文件（编译入口） |
| `sections/` | 摘要与正文各章 |
| `refs.tex` | 参考文献（GB/T 7714 著录） |
| `pages/` | 承诺书与编号专用页（仅纸质版使用） |
| `figures/` | 正文插图（由 `prep_figures.py`、`prep_ablation.py` 生成） |
| `code/` | 附录所用源程序副本 |
| `scripts/` | 辅助脚本 |
| `build.ps1` | 一键编译脚本 |

## 编译方法

要求：MiKTeX 或 TeX Live，XeLaTeX 引擎（`ctex` 宏包会自动调用）。

```powershell
cd 论文
.\build.ps1
```

或手动执行：

```text
xelatex main.tex
xelatex main.tex
```

## 纸质版 / 电子版切换

竞赛电子版论文不得包含承诺书与编号专用页，默认 `main.tex` 中的 `\paperversionfalse` 即为电子版（第一页为摘要页）。若需输出纸质版（前两页为承诺书与编号专用页），将 `main.tex` 中的开关改为：

```latex
\paperversiontrue
```

注意：承诺书与编号页文本请以当年组委会发布的正式版本核对后填写队伍信息。

## 数据与结果复现

正文所有实验数据来源于项目根目录：

- 训练日志：`runs/baseline/`、`runs/improved/`、`runs/detect/improved_with_neg/`
- 赛后补充实验（2026-08-13~14，汇总见 `results/实验记录.md`）：消融补格 `runs/exp_ablation_v8n_cp05`、`runs/exp_ablation_v8s_cp0`；负样本比例 `runs/exp_neg300`、`runs/exp_neg900`；Rusty 类加权 `runs/exp_rusty_w`；P2 头 `runs/exp_p2`；WD-Focal 端到端 `runs/exp_wd_focal`（v1 归档于 `runs/exp_wd_focal_v1_alpha_sat`）
- 逐类验证指标：`results/eval_final/val_summary.json`（可由 `scripts/run_val.py` 重新生成，需 GPU + ultralytics）
- EVT 判别结果：`results/evt*`（含赛后补跑的 `results/evt_neg300`、`results/evt_neg900`）
- 鲁棒性扰动测试：`results/robustness/robustness_results.json`
- 最终提交：`test_result.csv`

图表素材由以下脚本生成：

```powershell
python prep_figures.py
python prep_ablation.py
```

`prep_figures.py` 复制附录源码时会自动把代码注释中的 Unicode 数学符号（∈、≈、σ 等）替换为 ASCII，保证 XeLaTeX 的 listings 正常渲染；如需恢复原始内容，以 `src/` 为准重新复制即可。

注意：`test_result.csv` 的 `image_id` 目前带 `.jpg` 后缀，与 `.agents/rules/problem_d_rules.md` 中“不含扩展名”的规范不一致，正式提交前需与组委会确认后统一。
