# Container Damage Detection with YOLOv8 and Extreme Value Theory

[中文文档](./README.zh-CN.md) | English

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Supported-005CED?logo=onnx&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-Optimized-76B900?logo=nvidia&logoColor=white)
![Task](https://img.shields.io/badge/Task-Container%20Damage%20Detection-0A66C2)
![Model](https://img.shields.io/badge/Model-MSF--CNN%20%2B%20YOLO-6f42c1)
![Loss](https://img.shields.io/badge/Loss-WD--Focal-8A2BE2)
![Defect Classes](https://img.shields.io/badge/Defect%20Classes-5-2ea44f)
![Docs](https://img.shields.io/badge/Docs-English%20%7C%20中文-orange)
![GitHub stars](https://img.shields.io/github/stars/Nance-Mike/container-damage-detection?style=flat)
![GitHub forks](https://img.shields.io/github/forks/Nance-Mike/container-damage-detection?style=flat)
![GitHub issues](https://img.shields.io/github/issues/Nance-Mike/container-damage-detection)
![Last commit](https://img.shields.io/github/last-commit/Nance-Mike/container-damage-detection)
![Repo size](https://img.shields.io/github/repo-size/Nance-Mike/container-damage-detection)
[![GitHub Discussions](https://img.shields.io/badge/Discussions-Join-1f883d?logo=github)](https://github.com/Nance-Mike/container-damage-detection/discussions)
[![GitHub Issues](https://img.shields.io/badge/Issues-Feedback-blue?logo=github)](https://github.com/Nance-Mike/container-damage-detection/issues)
![CI](https://img.shields.io/github/actions/workflow/status/Nance-Mike/container-damage-detection/ci.yml?branch=main&label=CI)
![Release](https://img.shields.io/github/v/release/Nance-Mike/container-damage-detection?display_name=tag)

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
