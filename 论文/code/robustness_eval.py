# -*- coding: utf-8 -*-
"""鲁棒性量化：7 种扰动（高斯噪声 sigma=10/25/50、亮度 ±30、高斯模糊 3/7）下验证集 mAP 衰减。
扰动数据集写入 data/processed_perturb/，评估结果写入 results/robustness/。
"""
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(os.environ.get("TEMP", "/tmp")) / "yolo_cfg_robust"),
)
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

PERTURBS = {
    "gauss10": ("gauss", 10),
    "gauss25": ("gauss", 25),
    "gauss50": ("gauss", 50),
    "bright_p30": ("bright", 30),
    "bright_m30": ("bright", -30),
    "blur3": ("blur", 3),
    "blur7": ("blur", 7),
}


def perturb(img, kind, param, rng):
    if kind == "gauss":
        noise = rng.normal(0, param, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
    elif kind == "bright":
        out = img.astype(np.float32) + param
    elif kind == "blur":
        out = cv2.GaussianBlur(img, (param, param), 0)
    else:
        out = img
    return np.clip(out, 0, 255).astype(np.uint8)


def build_perturbed_datasets(base: Path, pert_root: Path, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    src_imgs = base / "images/val"
    src_lbls = base / "labels/val"
    yamls = {}
    imgs = sorted(src_imgs.glob("*.jpg"))
    for cond, (kind, param) in PERTURBS.items():
        d = pert_root / cond
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        for f in imgs:
            img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            out = perturb(img, kind, param, rng)
            ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            enc.tofile(str(d / "images" / f.name))
        for f in sorted(src_lbls.glob("*.txt")):
            shutil.copy2(f, d / "labels" / f.name)
        (d / "data.yaml").write_text(
            f"path: {d.as_posix()}\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            "  0: Dent\n"
            "  1: Hole\n"
            "  2: Rusty\n",
            encoding="utf-8",
        )
        yamls[cond] = str(d / "data.yaml")
    return yamls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/improved_with_neg/weights/best.pt"))
    ap.add_argument("--base", default=str(ROOT / "data/processed"))
    ap.add_argument("--pert-root", default=str(ROOT / "data/processed_perturb"))
    ap.add_argument("--out", default=str(ROOT / "results/robustness"))
    args = ap.parse_args()

    base = Path(args.base)
    pert_root = Path(args.pert_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    yamls = build_perturbed_datasets(base, pert_root)
    model = YOLO(args.weights)

    res = model.val(
        data=str(base / "data.yaml"), project=str(out_dir), name="clean",
        exist_ok=True, save_json=False, plots=False, batch=16, device=0, workers=0, verbose=False,
    )
    results = {"clean": {"map50": float(res.box.map50), "map50_95": float(res.box.map)}}
    print("clean:", results["clean"])

    for cond, y in yamls.items():
        res = model.val(
            data=y, project=str(out_dir), name=cond,
            exist_ok=True, save_json=False, plots=False, batch=16, device=0, workers=0, verbose=False,
        )
        results[cond] = {"map50": float(res.box.map50), "map50_95": float(res.box.map)}
        print(cond, results[cond])

    base_m = results["clean"]["map50_95"]
    rows = [
        {
            "condition": cond,
            "map50": r["map50"],
            "map50_95": r["map50_95"],
            "drop": round(base_m - r["map50_95"], 4),
        }
        for cond, r in results.items()
        if cond != "clean"
    ]
    report = {"baseline": results["clean"], "conditions": rows}
    (out_dir / "robustness_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved:", out_dir / "robustness_results.json")


if __name__ == "__main__":
    main()
