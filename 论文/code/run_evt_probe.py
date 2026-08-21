# -*- coding: utf-8 -*-
"""只读取证：用 600 张未用伪负样本评估 EVT 判别（不写任何现有文件）"""
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/detect/improved_with_neg/weights/best.pt"
VAL_DIR = ROOT / "data/processed/images/val"
NEG_DIR = ROOT / "data/processed/negative_samples/images"
TRAIN_DIR = ROOT / "data/processed/images/train"
EVT_RESULTS = ROOT / "results/evt_improved_with_neg/evt_results.json"
OUT = ROOT / "results/evt_pseudo_neg_results.txt"


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


def scheme_metrics(scores, y, threshold, rule):
    pred = (scores >= threshold).astype(int)
    tpr = float(pred[y == 1].mean())
    fpr = float(pred[y == 0].mean())
    return {"rule": rule, "threshold": threshold, "tpr": tpr, "fpr": fpr}


def main():
    model = YOLO(str(WEIGHTS))
    pos = max_conf_map(model, VAL_DIR)
    train_neg_names = {p.name for p in TRAIN_DIR.glob("neg_*")}
    unused_stems = {p.stem for p in NEG_DIR.glob("*.jpg") if p.name not in train_neg_names}
    all_neg = max_conf_map(model, NEG_DIR)
    neg = {k: v for k, v in all_neg.items() if k in unused_stems}

    y = np.array([1] * len(pos) + [0] * len(neg))
    s = np.array(list(pos.values()) + list(neg.values()))
    auc = float(roc_auc_score(y, s))
    fpr, tpr, ths = roc_curve(y, s)
    i34 = int(np.argmin(np.abs(np.array(ths) - 0.34)))
    mask = fpr <= 0.05
    i5 = int(np.argmax(tpr[mask])) if mask.any() else None

    evt = json.loads(EVT_RESULTS.read_text(encoding="utf-8"))
    k = float(evt["weibull_params"]["k"])
    lam = float(evt["weibull_params"]["lambda"])
    tau = float(evt["threshold"])
    equiv_conf = lam * (-np.log(1.0 - tau)) ** (1.0 / k)

    schemes = {
        "any_box": scheme_metrics(s, y, 0.01, "存在任一检测框（conf>=0.01）即判有缺陷"),
        "fixed_conf_0.25": scheme_metrics(s, y, 0.25, "最大置信度固定阈值 0.25（提交检测口径）"),
        "fixed_conf_0.34": scheme_metrics(s, y, 0.34, "最大置信度固定阈值 0.34（与 EVT 阈值等价）"),
        "evt_tau0.1": {
            "rule": "EVT 缺陷概率阈值 tau*=0.1（等价置信度 s*~%.3f）" % equiv_conf,
            "threshold": tau,
            "equiv_conf": float(equiv_conf),
            "tpr": float((s >= equiv_conf)[y == 1].mean()),
            "fpr": float((s >= equiv_conf)[y == 0].mean()),
        },
    }

    report = {
        "val_images": len(pos),
        "unused_negatives": len(neg),
        "pos_mean": float(np.mean(list(pos.values()))),
        "pos_median": float(np.median(list(pos.values()))),
        "neg_mean": float(np.mean(list(neg.values()))),
        "neg_median": float(np.median(list(neg.values()))),
        "auc": auc,
        "tpr_at_conf_0.34": float(tpr[i34]),
        "fpr_at_conf_0.34": float(fpr[i34]),
        "tpr_at_fpr5": float(tpr[i5]) if i5 is not None else None,
        "neg_over_0.34": int(sum(1 for v in neg.values() if v >= 0.34)),
        "pos_over_0.34": int(sum(1 for v in pos.values() if v >= 0.34)),
        "neg_zero_conf": int(sum(1 for v in neg.values() if v == 0.0)),
        "evt_equiv_conf": float(equiv_conf),
        "schemes": schemes,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
