[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

# Container Damage Detection（集装箱缺陷智能检测系统）

> **面向工业现场的长尾缺陷检测与高鲁棒边缘部署方案。**

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

本项目构建了集装箱缺陷智能检测全流程系统，融合 **MSF-CNN 多尺度特征融合**、**YOLO 演进式 Anchor-free 检测框架** 与 **WD-Focal Loss**，针对工业场景中的长尾分布、小样本难例和复杂干扰实现高鲁棒识别。

支持缺陷类别：**Dent（凹陷）**、**Rust（锈蚀）**、**Hole（穿孔）**、**Scratch（划痕）**、**Lock Damage（锁件/角件损坏）**。

---

## 🆕 核心亮点与演进（Highlights / Key Features）

| 能力维度 | 常规方案 | 本项目方案 |
|---|---|---|
| 长尾学习能力 | 标准 BCE/Focal | **WD-Focal Loss**（Wasserstein 距离感知重加权） |
| 多尺度特征表达 | 常规 FPN/PAN | **MSF-CNN** 强化小目标与细粒度缺陷表征 |
| 难样本处理 | 通用增强策略 | 面向工业难例的采样与损失联合优化 |
| 部署工程化 | 偏研究型脚本 | **Docker API + ONNX/TensorRT + C++/Qt 联动** |
| 生产可用性 | 可视化与流程弱 | 支持缺陷分级输出与质检可视化管理 |

> **Key idea：** 在保持边缘实时部署能力的同时，显著提升长尾类别与难样本缺陷的检测稳定性。

---

## 🏗️ 系统架构图（ASCII Flow）

```text
┌─────────────────────────────┐
│ 工业相机 / 视频流输入        │
│  (Line-scan / IPC / RTSP)   │
└──────────────┬──────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ 图像预处理与增强                        │
│ - 尺寸归一、标准化、去噪                │
│ - 光照补偿                              │
│ - 鲁棒性增强策略                        │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ MSF-CNN + YOLO 演进骨干                │
│ - 多尺度特征融合                        │
│ - Anchor-free 密集预测                 │
│ - WD-Focal Loss 优化                   │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ 缺陷检测与严重度评定                    │
│ - 多类别缺陷识别                        │
│ - 边界框定位                            │
│ - 缺陷等级评估                          │
└──────────────┬─────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌───────────────┐   ┌──────────────────────┐
│ Qt 可视化看板  │   │ Docker/C++ API 输出  │
│ 质检流程管理    │   │ MES/WMS 系统对接      │
└───────────────┘   └──────────────────────┘
```

---

## 🚀 快速上手（Quick Start）

### 1）Conda 环境搭建

```bash
conda create -n cdd python=3.10 -y
conda activate cdd
pip install -r requirements.txt
```

### 2）Docker 一键运行

```bash
docker pull nancemike/container-damage-detection:latest
docker run --gpus all -it --rm -p 8000:8000 nancemike/container-damage-detection:latest
```

### 3）极简推理 Demo（Python）

```python
from ultralytics import YOLO
model = YOLO("runs/improved_with_neg/weights/best.pt")
result = model("assets/container.jpg")
result[0].save(filename="outputs/pred_container.jpg")
```

### 4）标准项目目录树

```text
container-damage-detection/
├── configs/           # 训练、增强、导出配置
├── data/              # 数据集定义与处理元数据
├── models/            # MSF-CNN / YOLO 结构模块
├── losses/            # WD-Focal 等损失函数实现
├── tools/             # 训练、评估、推理、导出工具
├── deploy/            # ONNX/TensorRT、C++ API、Docker 服务
├── src/               # 核心训练与评估源码
├── results/           # 基准测试与消融实验输出
└── README.zh-CN.md
```

---

## 📊 理论突破与消融实验（The WD-Focal Loss & Benchmarks）

### WITHOUT WD-FOCAL LOSS vs. WITH WD-FOCAL LOSS

- **WITHOUT WD-FOCAL LOSS**  
  极度不平衡类别的梯度贡献不足，难例学习不充分，复杂工况下召回率下降明显。

- **WITH WD-FOCAL LOSS**  
  通过 Wasserstein 距离感知调制提升难样本与长尾类别权重，在工业边缘场景中获得更稳定的检测表现。

### 基准测试对比

| 模型 | mAP@0.5 | mAP@0.5:0.95 | FPS (TensorRT FP16) | 显存占用 (GB) |
|---|---:|---:|---:|---:|
| YOLO Baseline (n/s) | 0.68 | 0.19 | 86 | 2.4 |
| + MSF-CNN 融合 | 0.71 | 0.20 | 80 | 2.7 |
| **Ours (MSF-CNN + WD-Focal)** | **0.74** | **0.21** | **78** | **2.8** |

---

## 📖 模块与工作流阶段（Pipeline Roadmap）

1. **Phase 1: 数据流与预处理**  
   建立采集、清洗、增强与标注质控闭环。

2. **Phase 2: 难样本挖掘与损失优化**  
   面向长尾与难例构建样本挖掘策略并优化 WD-Focal Loss。

3. **Phase 3: 模型训练与微调**  
   基于 MSF-CNN + YOLO 演进架构完成训练与领域鲁棒性微调。

4. **Phase 4: C++/Qt 边缘部署**  
   导出 ONNX/TensorRT，集成 C++ 推理接口与 Qt 可视化质检端。
