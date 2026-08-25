[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

# Container Damage Detection

> **Industrial-grade long-tail defect detection with robust edge deployment for real-world container inspection.**

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

Container Damage Detection is an end-to-end system for intelligent container defect inspection, combining **MSF-CNN feature fusion**, **YOLO-evolved anchor-free detection**, and **WD-Focal Loss** to improve robustness on long-tail, imbalanced, and hard-example industrial data.

Supported defect categories include: **Dent**, **Rust**, **Hole**, **Scratch**, **Lock Damage**.

---

## 🆕 Highlights / Key Features

| Capability | Baseline Detector | This Project |
|---|---|---|
| Long-tail learning | Standard BCE/Focal | **WD-Focal Loss** with Wasserstein-distance-aware reweighting |
| Multi-scale representation | Conventional FPN/PAN | **MSF-CNN** enhanced multi-scale fusion for small and subtle defects |
| Hard sample handling | Generic augmentation | Hard-case-oriented training and imbalance-aware optimization |
| Deployment readiness | Research-only scripts | **Dockerized APIs + ONNX/TensorRT export + C++/Qt integration** |
| Production usability | Limited monitoring | Visual quality-inspection workflow with severity-aware outputs |

> **Key idea:** Improve rare-class and hard-sample sensitivity without sacrificing real-time deployability at the edge.

---

## 🏗️ System Architecture (ASCII Flow)

```text
┌─────────────────────────────┐
│ Industrial Cameras / Stream │
│  (Line-scan / IPC / RTSP)   │
└──────────────┬──────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ Image Ingestion & Preprocessing        │
│ - Resize / Normalize / Denoise         │
│ - Illumination correction              │
│ - Data augmentation for robustness     │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ MSF-CNN + YOLO Evolution Backbone      │
│ - Multi-scale feature fusion           │
│ - Anchor-free dense prediction         │
│ - WD-Focal Loss optimization           │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ Defect Detection & Severity Scoring    │
│ - Multi-class defect classification    │
│ - Bounding box localization            │
│ - Severity level estimation            │
└──────────────┬─────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌───────────────┐   ┌──────────────────────┐
│ Qt Dashboard  │   │ Docker/C++ API Output│
│ QC Visualization│  │ MES/WMS Integration  │
└───────────────┘   └──────────────────────┘
```

---

## 🚀 Quick Start

### 1) Environment Setup (Conda)

```bash
conda create -n cdd python=3.10 -y
conda activate cdd
pip install -r requirements.txt
```

### 2) Docker One-Command Run

```bash
docker pull nancemike/container-damage-detection:latest
docker run --gpus all -it --rm -p 8000:8000 nancemike/container-damage-detection:latest
```

### 3) Minimal Inference Demo (Python)

```python
from ultralytics import YOLO
model = YOLO("runs/improved_with_neg/weights/best.pt")
result = model("assets/container.jpg")
result[0].save(filename="outputs/pred_container.jpg")
```

### 4) Standard Project Layout

```text
container-damage-detection/
├── configs/           # training, augmentation, export configs
├── data/              # dataset definitions and processed metadata
├── models/            # model blocks (MSF-CNN/YOLO variants)
├── losses/            # WD-Focal and other criterion implementations
├── tools/             # train/eval/infer/export utilities
├── deploy/            # ONNX/TensorRT, C++ API, Docker service
├── src/               # core training and evaluation code
├── results/           # benchmark reports and ablation outputs
└── README.md
```

---

## 📊 The WD-Focal Loss & Benchmarks

### WITHOUT WD-FOCAL LOSS vs. WITH WD-FOCAL LOSS

- **WITHOUT WD-FOCAL LOSS**  
  Rare defect classes are under-optimized; hard samples contribute weak gradients; recall drops under severe class imbalance.

- **WITH WD-FOCAL LOSS**  
  Wasserstein-distance-aware modulation amplifies informative hard samples and long-tail classes, improving robustness and consistency on industrial edge cases.

### Benchmark Comparison

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS (TensorRT FP16) | GPU Memory (GB) |
|---|---:|---:|---:|---:|
| YOLO Baseline (n/s) | 0.68 | 0.19 | 86 | 2.4 |
| + MSF-CNN Fusion | 0.71 | 0.20 | 80 | 2.7 |
| **Ours (MSF-CNN + WD-Focal)** | **0.74** | **0.21** | **78** | **2.8** |

---

## 📖 Pipeline Roadmap

1. **Phase 1: Data Flow & Preprocessing**  
   Build acquisition, cleaning, augmentation, and annotation-quality control pipeline.

2. **Phase 2: Hard Sample Mining & Loss Optimization**  
   Mine difficult/rare cases and optimize WD-Focal Loss for long-tail distribution.

3. **Phase 3: Model Training & Fine-tuning**  
   Train YOLO-evolved detector with MSF-CNN, then fine-tune for domain-specific robustness.

4. **Phase 4: C++/Qt Edge Deployment**  
   Export ONNX/TensorRT, integrate C++ runtime APIs, and deliver Qt visual inspection interface.
