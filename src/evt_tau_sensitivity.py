# -*- coding: utf-8 -*-
"""EVT 判别阈值灵敏度分析：计算 tau 邻域内的等价置信度、TPR 与 FPR。
只读取证：用验证集（494 幅正样本）与 600 张未使用伪负样本，
不改动任何现有训练/评估文件，结果写入 results/evt_tau_sensitivity.json。
"""
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/detect/improved_with_neg/weights/best.pt"
VAL_DIR = ROOT / "data/processed/images/val"
NEG_DIR = ROOT / "data/processed/negative_samples/images"
TRAIN_DIR = ROOT / "data/processed/images/train"
EVT_RESULTS = ROOT / "results/evt_improved_with_neg/evt_results.json"
OUT = ROOT / "results/evt_tau_sensitivity.json"
TAUS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


def max_conf_map(model, image_dir):
    results = model.predict(
        source=str(image_dir), conf=0.01, iou=0.5, verbose=False, save=False, device=0
    )
    out = {}
    for r in results:
        boxes = r.boxes
        out[Path(r.path).stem] = (
            float(boxes.conf.max().cpu()) if (boxes is not None and len(boxes) > 0) else 0.0
        )
    return out


def main():
    model = YOLO(str(WEIGHTS))
    pos = max_conf_map(model, VAL_DIR)
    train_neg_names = {p.name for p in TRAIN_DIR.glob("neg_*")}
    unused_stems = {p.stem for p in NEG_DIR.glob("*.jpg") if p.name not in train_neg_names}
    all_neg = max_conf_map(model, NEG_DIR)
    neg = {k: v for k, v in all_neg.items() if k in unused_stems}

    evt = json.loads(EVT_RESULTS.read_text(encoding="utf-8"))
    k = float(evt["weibull_params"]["k"])
    lam = float(evt["weibull_params"]["lambda"])
    s_pos = np.array(list(pos.values()))
    s_neg = np.array(list(neg.values()))

    rows = []
    for tau in TAUS:
        s_star = lam * (-np.log(1.0 - tau)) ** (1.0 / k)
        tpr = float((s_pos >= s_star).mean())
        fpr = float((s_neg >= s_star).mean())
        rows.append(
            {
                "tau": tau,
                "s_star": round(float(s_star), 4),
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
            }
        )

    report = {
        "val_images": len(pos),
        "negatives": len(neg),
        "k": round(k, 4),
        "lambda": round(lam, 4),
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
