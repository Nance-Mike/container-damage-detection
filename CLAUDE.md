# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mathematical modeling competition project (2026 CUMCM Topic D) for intelligent detection of container damage using YOLOv8 and Extreme Value Theory (EVT). The project includes:

- Object detection model training with YOLOv8
- Custom loss functions (WD-Focal Loss, Rusty class weighting)
- EVT-based abnormality detection
- LaTeX paper generation for competition submission
- Comprehensive evaluation and robustness testing

## Directory Structure

```
├── src/                    # Source code for model training and evaluation
├── 论文/                   # LaTeX paper source and build scripts
├── 论文/code/              # Copy of source code for paper appendix
├── data/                   # Processed dataset configurations
├── 数据集3713/             # Original dataset (tracked in this repo)
├── runs/                   # Training outputs and model weights
├── results/                # Evaluation results and EVT analysis
├── .venv/                  # Virtual environment
└── .agents/                # Claude Code agents and skills
```

## Key Commands

> 本机 Python 解释器为 `E:\python2025\python.exe`；以下命令以 `python` 代指。

### Model Training

```bash
# Train baseline model (YOLOv8n)
python src/train_yolo.py --mode baseline --model yolov8n.pt --name baseline

# Train improved model (YOLOv8s/m with enhanced augmentation)
python src/train_yolo.py --mode improved --model yolov8s.pt --name improved

# Train with custom loss functions
python src/train_yolo.py --mode improved --model yolov8s.pt --loss wd_focal
python src/train_yolo.py --mode improved --model yolov8s.pt --loss rusty_w

# Train with negative samples
python src/train_yolo.py --mode improved --model yolov8s.pt --name improved_with_neg --copy-paste 0.5
```

### Inference and Evaluation

```bash
# Run inference on test set
python src/train_yolo.py --mode predict --weights runs/improved/weights/best.pt

# Generate evaluation metrics
python 论文/scripts/run_val.py

# Run EVT analysis on pseudo-negative samples
python src/run_evt_probe.py

# Robustness evaluation
python src/robustness_eval.py
```

### LaTeX Paper Build

```powershell
# Navigate to paper directory and build
cd 论文
.\build.ps1
```

This script performs two compilation passes to generate the final PDF from `main.tex`.
Figure restyling scripts are `论文/restyle_figures.py`, `论文/restyle_montages.py`
and `论文/regenerate_weibull.py`; English AI-image prompts live in `论文/配图提示词.md`.

## Model Architecture and Training Strategy

### Baseline Model
- YOLOv8n backbone
- Standard training settings: 100 epochs, batch size 16
- Data augmentation: mosaic=1.0, copy-paste=0.0

### Improved Model
- YOLOv8s backbone
- Extended training: 150 epochs
- Enhanced augmentation: mosaic=1.0, copy-paste=0.5
- Learning rate scheduling with cosine decay

### Custom Loss Functions

1. **WD-Focal Loss**: Combines Wasserstein-distance (W1) weighting and Focal Loss for class
   imbalance. Adaptive threshold tau=0 (median of W1 distances), alpha clamped to [0.25, 0.75]
   (v2); updates every 10 epochs. End-to-end result was negative vs. baseline — details in
   `results/实验记录.md` (the W1 weighting did shift more weight to Hole, as designed).

2. **Rusty Class Weighting**: BCE loss with pos_weight=[1.0, 1.0, 1.246] based on class
   frequency ratio. Measured result was negative (mAP50-95 −0.033, Rusty AP/recall unchanged):
   Rusty's bottleneck is class confusion / texture, not sample count.

### EVT Integration
The EVT classifier uses maximum confidence scores from:
- Validation set (positive samples)
- Unused pseudo-negative samples
- AUC score calculation with optimal threshold determination

## Data Configuration

The project uses YOLO format with 3 classes:
- 0: Dent
- 1: Hole  
- 2: Rusty

Data YAML files are located in `data/processed/` with subdirectories for different experimental conditions (negative sampling, perturbation tests).

## Important Implementation Details

1. **Project Root Reference**: Key scripts (`train_yolo.py`, `eda.py`, `evt_classifier.py`)
   resolve the project root with `PROJECT_ROOT = Path(__file__).resolve().parents[1]`;
   no hardcoded absolute paths remain under `src/`
2. **Training Checkpoints**: Models save to `runs/{name}/weights/` with `best.pt` and `last.pt`
3. **Resume Training**: Use `--resume` flag to continue from last checkpoint
4. **Environment Variable**: `YOLO_CONFIG_DIR` is set to avoid config file conflicts
5. **Unicode to ASCII**: Paper prep script converts Unicode symbols (∈, ≈, σ) to ASCII for LaTeX compatibility

## File Duplication Note

The `论文/code/` directory contains copies of source code for the paper appendix. The authoritative source code is in `src/`.

## Results and Experiments

Training outputs and evaluation results are stored in:
- `runs/` - Training logs and model weights
- `results/` - Metrics, EVT analysis, and robustness test results
- `test_result.csv` - Final predictions for test set

The project includes extensive ablation studies on negative samples, perturbation robustness, and loss function comparisons.

Current state (2026-08-16): all planned experiments are finished and the final model remains
`improved_with_neg` (mAP50-95 ≈ 0.205, EVT k = 3.064); the 84-page paper passes `verify_pdf`
with 18 references, and the repository is synced with GitHub `origin/main`.
