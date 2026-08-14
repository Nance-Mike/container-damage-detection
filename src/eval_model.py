# -*- coding: utf-8 -*-
"""任意权重/数据集的验证评估：输出 val_summary.json；可选打印 results.csv 最优轮指标"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(os.environ.get("TEMP", "/tmp")) / "yolo_cfg_eval"),
)
from ultralytics import YOLO
import pandas as pd


def dump_summary(res, out_dir: Path) -> dict:
    out = {
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "instances": int(res.box.nc),
        "precision": [float(v) for v in res.box.p],
        "recall": [float(v) for v in res.box.r],
        "ap50": [float(v) for v in res.box.ap50],
        "ap50_95": [float(v) for v in res.box.ap],
        "f1": [float(v) for v in res.box.f1],
        "speed": {k: float(v) for k, v in res.speed.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def best_from_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    i = df["metrics/mAP50-95(B)"].idxmax()
    row = df.loc[i]
    print(
        f"best mAP50-95={row['metrics/mAP50-95(B)']:.5f} @ep{int(row['epoch'])} "
        f"mAP50={row['metrics/mAP50(B)']:.5f} P={row['metrics/precision(B)']:.5f} "
        f"R={row['metrics/recall(B)']:.5f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="模型权重路径")
    ap.add_argument("--data", default="data/processed/data.yaml", help="数据集 yaml")
    ap.add_argument("--project", default="runs/exp_eval", help="输出项目目录")
    ap.add_argument("--name", default="val", help="输出子目录名")
    ap.add_argument("--best", default="", help="可选：results.csv 路径，先打印最优轮指标")
    ap.add_argument("--augment", action="store_true", help="启用 TTA 验证（augment=True）")
    args = ap.parse_args()

    if args.best:
        best_from_csv(args.best)
    model = YOLO(args.weights)
    res = model.val(
        data=args.data,
        project=args.project,
        name=args.name,
        augment=args.augment,
        exist_ok=True,
        save_json=False,
        plots=False,
        batch=16,
        device=0,
        workers=0,
        verbose=False,
    )
    dump_summary(res, Path(args.project) / args.name)


if __name__ == "__main__":
    main()
