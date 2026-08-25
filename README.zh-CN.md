# 基于 YOLOv8 与极值理论的集装箱破损检测

中文 | [English](./README.md)

本仓库对应 2026 年全国大学生数学建模竞赛（高教社杯）D 题，构建了一个
结合 **YOLOv8 目标检测** 与 **极值理论（EVT）开集判别** 的集装箱破损智能检测方案。

缺陷类别如下：

- `Dent`（凹陷，class 0）
- `Hole`（破洞，class 1）
- `Rusty`（锈蚀，class 2）

## 项目亮点

- 一套检测模型同时服务于目标定位与图像级破损判别。
- 采用 EVT（基于最大置信度的 Weibull 拟合）区分破损/未破损图像。
- 包含赛后补充的消融实验与鲁棒性实验，结果可复现。
- 提供完整论文交付物：LaTeX 源码、PDF、Word 版与打包脚本。

## 仓库结构

```text
src/                  训练、评估、EVT 与鲁棒性脚本
论文/                 论文 LaTeX 源码与构建脚本
results/              评估结果、EVT 分析与实验记录
data/, 数据集3713/    处理后/原始数据资源
runs/                 训练产物与权重文件
pack_submission.py    提交材料打包脚本
```

## 环境要求

- Python 3.13
- 关键依赖：`ultralytics==8.4.110`、`torch==2.13.0`
- 若 Ultralytics 配置写入失败，请设置可写目录 `YOLO_CONFIG_DIR`。

## 快速开始

```bash
# 1）训练基线模型
python src/train_yolo.py --mode baseline --model yolov8n.pt --name baseline

# 2）训练改进/最终风格模型
python src/train_yolo.py --mode improved --model yolov8s.pt --copy-paste 0.5 --epochs 150

# 3）在验证集上评估
python src/eval_model.py --weights runs/detect/improved_with_neg/weights/best.pt

# 4）执行 EVT 伪负样本标定
python src/run_evt_probe.py

# 5）鲁棒性评估（噪声/亮度/模糊）
python src/robustness_eval.py
```

## 论文构建

```powershell
cd 论文
.\build.ps1
python scripts/verify_pdf.py main.pdf
```

## 核心结果（验证集 494 张）

| 模型 | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- |
| Baseline（YOLOv8n） | 0.395 | 0.190 |
| Improved（YOLOv8s + Copy-Paste） | 0.413 | 0.212 |
| Final（Improved + 600 负样本） | 0.405 | 0.205 |

## 交付物

- `test_result.csv` — 最终测试集预测结果
- `论文/main.pdf` / `论文/main.docx` — 最终论文（PDF/Word）
- `AI工具使用详情.pdf` — AI 工具使用说明
- `pack_submission.py` — 将提交材料打包为 `20262026104.zip`

## 说明

- 以 `src/` 为主代码目录。
- `论文/code/` 为论文附录展示用代码副本。
- 详细实验记录见 `results/实验记录.md`。
