# -*- coding: utf-8 -*-
"""Regenerate the Weibull-fit figure under Okabe-Ito palette A.

Steps:
  1. run the final model on the 494 validation images (conf=0.01, iou=0.5);
  2. collect the image-level maximum confidence score;
  3. fit a Weibull distribution;
  4. redraw the two-panel figure with the unified palette.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.environ.get("TEMP", "/tmp")) / "yolo_cfg_weibull"))
os.environ.setdefault("WINDIR", r"C:\Windows")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from ultralytics import YOLO

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
CHARCOAL = "#333333"
GREY = "#4D4D4D"
GRID = "#D9D9D9"


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


def collect_val_scores(weights, val_dir):
    model = YOLO(str(weights))
    results = model.predict(
        source=str(val_dir),
        conf=0.01,
        iou=0.5,
        verbose=False,
        save=False,
        device=0,
    )
    scores = []
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            scores.append(float(boxes.conf.max().cpu()))
        else:
            scores.append(0.0)
    return np.asarray(scores, dtype=float)


def main():
    set_style()
    weights = ROOT / "runs" / "detect" / "improved_with_neg" / "weights" / "best.pt"
    val_dir = ROOT / "data" / "processed" / "images" / "val"
    out_npy = ROOT / "论文" / "val_scores.npy"
    out_png = ROOT / "论文" / "figures" / "fig_weibull.png"

    scores = collect_val_scores(weights, val_dir)
    np.save(out_npy, scores)
    print(f"collected {len(scores)} val scores -> {out_npy}")

    valid = scores[scores > 0]
    k, loc, lam = weibull_min.fit(valid, floc=0)
    print(f"weibull k={k:.4f} lambda={lam:.4f} valid={len(valid)}")

    x = np.linspace(0.001, 1.0, 400)
    pdf = weibull_min.pdf(x, k, loc=loc, scale=lam)
    cdf = weibull_min.cdf(x, k, loc=loc, scale=lam)
    tau = 0.1
    s_star = lam * (-np.log(1 - tau)) ** (1 / k)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.7))

    axes[0].hist(valid, bins=50, density=True, color=BLUE, alpha=0.75,
                 edgecolor="white", linewidth=0.4, label="实测分布")
    axes[0].plot(x, pdf, color=ORANGE, lw=2.2,
                 label=f"Weibull 拟合（$k={k:.2f},\\lambda={lam:.2f}$）")
    axes[0].set_xlabel("最大检测置信度 $s$")
    axes[0].set_ylabel("概率密度")
    axes[0].set_title("Weibull 分布拟合效果", fontsize=12)
    axes[0].legend(fontsize=9.5)
    axes[0].grid(alpha=0.6)
    _spines(axes[0])

    axes[1].plot(x, cdf, color=GREEN, lw=2.2,
                 label="$P_{\\mathrm{defect}}(s)$ = Weibull CDF")
    axes[1].axhline(y=tau, color=RED, ls="--", lw=1.2,
                    label=f"$\\tau^*={tau}$")
    axes[1].axvline(x=s_star, color=GREY, ls=":", lw=1.2,
                    label=f"$s^*\\approx{s_star:.2f}$")
    axes[1].set_xlabel("最大检测置信度 $s$")
    axes[1].set_ylabel("缺陷概率 $P_{\\mathrm{defect}}$")
    axes[1].set_title("置信度 $\\rightarrow$ 缺陷概率映射", fontsize=12)
    axes[1].legend(fontsize=9.5, loc="upper left")
    axes[1].grid(alpha=0.6)
    _spines(axes[1])

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
