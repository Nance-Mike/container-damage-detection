# -*- coding: utf-8 -*-
"""Unified restyle of paper data figures under Okabe-Ito palette A.

Regenerates the six EDA/ablation figures directly from the raw dataset and
training logs so that colors, typography, grids and annotations are consistent.
The Weibull figure is handled separately because it requires per-image max
confidence scores produced by model inference.
"""
from pathlib import Path
import random

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "数据集3713"
FIG = ROOT / "论文" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GREY = "#4D4D4D"
CHARCOAL = "#333333"
GRID = "#D9D9D9"

CLASS_NAMES = {0: "Dent（凹陷）", 1: "Hole（破洞）", 2: "Rusty（锈蚀）"}
CLASS_SHORT = {0: "Dent", 1: "Hole", 2: "Rusty"}
CLASS_COLORS = {0: BLUE, 1: ORANGE, 2: GREEN}


def set_style():
    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "text.color": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "legend.frameon": False,
    })


def _spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def parse_labels():
    rows = []
    boxes_per_image = []
    lbl_dir = RAW / "labels" / "train"
    for f in sorted(lbl_dir.glob("*.txt")):
        lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        boxes_per_image.append(len(lines))
        for ln in lines:
            p = ln.split()
            cid = int(p[0])
            cx, cy, w, h = map(float, p[1:5])
            rows.append({
                "class": cid,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "area": w * h * 640 * 640,
            })
    return pd.DataFrame(rows), boxes_per_image


def fig_class_dist(df):
    counts = df["class"].value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in counts.index]
    colors = [CLASS_COLORS[i] for i in counts.index]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, counts.values, color=colors, width=0.58)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v}",
                ha="center", va="bottom", fontsize=11, color=CHARCOAL)
    ax.set_ylabel("标注框数量")
    ax.set_title("训练集各类别标注框数量分布", fontsize=13)
    ax.set_ylim(0, max(counts.values) * 1.14)
    ax.grid(axis="y", alpha=0.7)
    _spines(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_class_dist.png")
    plt.close(fig)


def fig_boxes_per_image(boxes):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    bins = range(1, max(boxes) + 2)
    ax.hist(boxes, bins=bins, color=BLUE, alpha=0.85, edgecolor="white",
            linewidth=0.4, align="left")
    ax.set_xlabel("每图缺陷数量")
    ax.set_ylabel("图片数量")
    ax.set_title("训练集每图缺陷数量分布", fontsize=13)
    ax.grid(axis="y", alpha=0.7)
    _spines(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_boxes_per_image.png")
    plt.close(fig)


def fig_scale_dist(df):
    def scale_of(a):
        if a < 32 * 32:
            return "小目标"
        if a <= 96 * 96:
            return "中目标"
        return "大目标"
    df = df.copy()
    df["scale"] = df["area"].apply(scale_of)

    cats = ["小目标", "中目标", "大目标"]
    colors = [BLUE, ORANGE, GREEN]
    classes = sorted(df["class"].unique())

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(classes))
    left = np.zeros(len(classes))
    pct = np.zeros((len(classes), len(cats)))
    for j, cat in enumerate(cats):
        counts = [int(((df["class"] == c) & (df["scale"] == cat)).sum()) for c in classes]
        totals = [int((df["class"] == c).sum()) for c in classes]
        pct[:, j] = np.array(counts) / np.array(totals) * 100
        ax.barh(y, pct[:, j], left=left, color=colors[j], height=0.52,
                label=cat, edgecolor="white", linewidth=0.6)
        left += pct[:, j]

    for i, c in enumerate(classes):
        ax.text(101.5, i, CLASS_SHORT[c], va="center", ha="left",
                fontsize=11, color=CHARCOAL)
        xmid = 0.0
        for j, cat in enumerate(cats):
            v = pct[i, j]
            if v >= 4:
                ax.text(xmid + v / 2, i, f"{v:.1f}%", va="center", ha="center",
                        fontsize=8.5, color="white")
            xmid += v
    ax.set_yticks([])
    ax.set_xlabel("占该类别标注框比例（%）")
    ax.set_title("训练集各类别目标尺度分布（COCO 标准）", fontsize=13)
    ax.set_xlim(0, 118)
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    ax.grid(axis="x", alpha=0.7)
    _spines(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_scale_dist.png")
    plt.close(fig)


def _cv2_read(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def fig_quality():
    img_dir = RAW / "images" / "train"
    files = list(img_dir.glob("*.jpg"))
    sample = random.Random(42).sample(files, min(100, len(files)))
    bright, contrast, sat, hue = [], [], [], []
    for f in sample:
        img = _cv2_read(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bright.append(float(np.mean(gray)))
        contrast.append(float(np.std(gray)))
        sat.append(float(np.mean(hsv[:, :, 1])))
        hue.append(float(np.mean(hsv[:, :, 0])))

    panels = [
        ("亮度", bright, "灰度均值"),
        ("对比度", contrast, "灰度标准差"),
        ("饱和度", sat, "HSV 饱和度均值"),
        ("色调", hue, "HSV 色调均值"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.2))
    for ax, (title, data, xlabel) in zip(axes.flat, panels):
        ax.hist(data, bins=28, color=BLUE, alpha=0.8, edgecolor="white",
                linewidth=0.4, density=True)
        ax.set_title(f"图像{title}分布（100 张采样）", fontsize=11.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("频率密度", fontsize=10)
        ax.grid(alpha=0.6)
        _spines(ax)
    fig.suptitle("训练集图像视觉特征统计", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "fig_quality.png")
    plt.close(fig)


def fig_heatmap(df):
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    hist, xedges, yedges = np.histogram2d(
        df["cx"].to_numpy(), df["cy"].to_numpy(),
        bins=24, range=[[0, 1], [0, 1]],
    )
    hist = np.rot90(hist)
    im = ax.imshow(hist, extent=[0, 1, 0, 1], origin="lower",
                   cmap="Blues", aspect="equal", interpolation="bicubic")
    ax.set_xlabel("归一化横向位置 $x$")
    ax.set_ylabel("归一化纵向位置 $y$")
    ax.set_title("训练集标注框位置热力图", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("标注框数量密度", fontsize=10)
    _spines(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_heatmap.png")
    plt.close(fig)


def fig_ablation():
    runs = {
        "基线 YOLOv8n": ROOT / "runs" / "baseline" / "results.csv",
        "改进 YOLOv8s+CP": ROOT / "runs" / "improved" / "results.csv",
        "最终模型（+负样本）": ROOT / "runs" / "detect" / "improved_with_neg" / "results.csv",
    }
    series = [
        ("基线 YOLOv8n", GREY),
        ("改进 YOLOv8s+CP", ORANGE),
        ("最终模型（+负样本）", BLUE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))

    best = []
    for label, color in series:
        df = pd.read_csv(runs[label])
        ep = df["epoch"]
        axes[0].plot(ep, df["metrics/mAP50(B)"], label=f"{label} mAP@0.5",
                     color=color, lw=1.7)
        axes[0].plot(ep, df["metrics/mAP50-95(B)"], ls="--", lw=1.2,
                     color=color, alpha=0.9)
        i = df["metrics/mAP50(B)"].idxmax()
        best.append((label, df.loc[i, "metrics/mAP50(B)"],
                     df.loc[i, "metrics/mAP50-95(B)"]))

    axes[0].set_xlabel("训练轮数 Epoch")
    axes[0].set_ylabel("mAP")
    axes[0].set_title("训练过程 mAP 曲线", fontsize=12)
    axes[0].legend(fontsize=8.5, ncol=1, loc="lower right")
    axes[0].grid(alpha=0.6)
    _spines(axes[0])

    labels = [b[0] for b in best]
    m50 = [b[1] for b in best]
    m5095 = [b[2] for b in best]
    x = np.arange(len(labels))
    w = 0.36
    axes[1].bar(x - w / 2, m50, width=w, color=ORANGE, label="mAP@0.5")
    axes[1].bar(x + w / 2, m5095, width=w, color=BLUE, label="mAP@0.5:0.95")
    for xi, v in zip(x - w / 2, m50):
        axes[1].text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    for xi, v in zip(x + w / 2, m5095):
        axes[1].text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["基线", "改进", "最终"], fontsize=10)
    axes[1].set_ylabel("mAP")
    axes[1].set_title("验证集最优 mAP 对比", fontsize=12)
    axes[1].set_ylim(0, 0.5)
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(axis="y", alpha=0.6)
    _spines(axes[1])

    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)


def fig_results_curves():
    csv_path = ROOT / "runs" / "detect" / "improved_with_neg" / "results.csv"
    df = pd.read_csv(csv_path)
    ep = df["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0))

    loss_specs = [
        ("train/box_loss", "Box", BLUE),
        ("train/cls_loss", "Cls", ORANGE),
        ("train/dfl_loss", "DFL", GREEN),
    ]
    for col, label, color in loss_specs:
        axes[0, 0].plot(ep, df[col], color=color, lw=1.4, label=label)
    axes[0, 0].set_title("训练损失", fontsize=11.5)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.6)
    _spines(axes[0, 0])

    for col, label, color in loss_specs:
        axes[0, 1].plot(ep, df["val/" + col.split("/")[1]], color=color, lw=1.4, label=label)
    axes[0, 1].set_title("验证损失", fontsize=11.5)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.6)
    _spines(axes[0, 1])

    axes[1, 0].plot(ep, df["metrics/precision(B)"], color=BLUE, lw=1.5, label="Precision")
    axes[1, 0].plot(ep, df["metrics/recall(B)"], color=ORANGE, lw=1.5, label="Recall")
    axes[1, 0].set_title("精确率与召回率", fontsize=11.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("分数")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.6)
    _spines(axes[1, 0])

    axes[1, 1].plot(ep, df["metrics/mAP50(B)"], color=ORANGE, lw=1.5, label="mAP@0.5")
    axes[1, 1].plot(ep, df["metrics/mAP50-95(B)"], color=BLUE, lw=1.5, label="mAP@0.5:0.95")
    axes[1, 1].set_title("mAP", fontsize=11.5)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("mAP")
    axes[1, 1].set_ylim(0, 0.6)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.6)
    _spines(axes[1, 1])

    fig.suptitle("最终模型训练过程曲线", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG / "fig_results_curves.png")
    plt.close(fig)


def main():
    set_style()
    df, boxes = parse_labels()
    fig_class_dist(df)
    fig_boxes_per_image(boxes)
    fig_scale_dist(df)
    fig_quality()
    fig_heatmap(df)
    fig_ablation()
    fig_results_curves()
    print("restyled 7 figures into", FIG)


if __name__ == "__main__":
    main()
