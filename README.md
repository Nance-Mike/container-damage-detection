# Container Damage Detection with YOLOv8 and Extreme Value Theory

[中文文档](./README.zh-CN.md) | English

2026 CUMCM (China Undergraduate Mathematical Modeling Contest, Higher Education Cup) — Topic D.
This repository provides an intelligent container-damage detection pipeline that combines a
**YOLOv8 detector** with an **Extreme Value Theory (EVT) open-set classifier**.

Defect classes:

- `Dent` (class 0)
- `Hole` (class 1)
- `Rusty` (class 2)

## Highlights

- Uses one detection model for both localization and image-level defect judgment.
- Uses EVT (Weibull fit on max confidence) to separate damaged vs. undamaged images.
- Includes reproducible post-competition ablation and robustness experiments.
- Provides complete paper artifacts: LaTeX source, PDF, Word version, and packaging script.

## Repository Layout

```text
src/                  Training, evaluation, EVT, and robustness scripts
论文/                 LaTeX paper source and build scripts
results/              Evaluation outputs, EVT analysis, and experiment logs
data/, 数据集3713/    Processed/original dataset assets
runs/                 Training artifacts and checkpoints
pack_submission.py    Submission packaging script
```

## Environment

- Python 3.13
- Key dependencies: `ultralytics==8.4.110`, `torch==2.13.0`
- If Ultralytics config fails, set `YOLO_CONFIG_DIR` to a writable directory.

## Quick Start

```bash
# 1) Train baseline model
python src/train_yolo.py --mode baseline --model yolov8n.pt --name baseline

# 2) Train improved/final-style model
python src/train_yolo.py --mode improved --model yolov8s.pt --copy-paste 0.5 --epochs 150

# 3) Evaluate on validation set
python src/eval_model.py --weights runs/detect/improved_with_neg/weights/best.pt

# 4) EVT pseudo-negative calibration
python src/run_evt_probe.py

# 5) Robustness evaluation (noise/brightness/blur)
python src/robustness_eval.py
```

## Build the Paper

```powershell
cd 论文
.\build.ps1
python scripts/verify_pdf.py main.pdf
```

## Key Results (Validation, 494 Images)

| Model | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- |
| Baseline (YOLOv8n) | 0.395 | 0.190 |
| Improved (YOLOv8s + Copy-Paste) | 0.413 | 0.212 |
| Final (Improved + 600 negatives) | 0.405 | 0.205 |

## Deliverables

- `test_result.csv` — final test-set predictions
- `论文/main.pdf` / `论文/main.docx` — final paper (PDF/Word)
- `AI工具使用详情.pdf` — AI usage details
- `pack_submission.py` — packages required submission files into `20262026104.zip`

## Notes

- Authoritative source code is under `src/`.
- `论文/code/` contains appendix copies of source files for the paper.
- Detailed experiment records are in `results/实验记录.md`.
