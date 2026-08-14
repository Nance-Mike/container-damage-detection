# -*- coding: utf-8 -*-
"""构建负样本比例实验数据集变体：data/processed_neg300 与 data/processed_neg900。
负样本单调包含：neg300 = 原训练集 600 负样本中的前 300；neg900 = 原 600 + 未用候选前 300。
训练正样本与验证集与 data/processed 完全一致（val 固定 494 张）。
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed"
VARIANTS = {"neg300": 300, "neg900": 900}


def main():
    train_img_dir = BASE / "images/train"
    train_lbl_dir = BASE / "labels/train"
    val_img_dir = BASE / "images/val"
    val_lbl_dir = BASE / "labels/val"
    cand_dir = BASE / "negative_samples/images"

    used_neg = sorted(p.name for p in train_img_dir.glob("neg_*"))
    all_cand = sorted(p.name for p in cand_dir.glob("*.jpg"))
    unused_neg = sorted(n for n in all_cand if n not in set(used_neg))
    pos_files = sorted(p.name for p in train_img_dir.glob("*.jpg") if not p.name.startswith("neg_"))
    val_files = sorted(p.name for p in val_img_dir.glob("*.jpg"))
    print(f"正样本={len(pos_files)} 已用负样本={len(used_neg)} 未用负样本={len(unused_neg)} val={len(val_files)}")

    for var_name, n_neg in VARIANTS.items():
        if n_neg <= len(used_neg):
            neg_set = used_neg[:n_neg]
        else:
            neg_set = used_neg + unused_neg[: n_neg - len(used_neg)]
        dst = ROOT / f"data/processed_{var_name}"
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            (dst / sub).mkdir(parents=True, exist_ok=True)

        for name in pos_files:
            shutil.copy2(train_img_dir / name, dst / "images/train" / name)
            lbl_name = Path(name).with_suffix(".txt").name
            shutil.copy2(train_lbl_dir / lbl_name, dst / "labels/train" / lbl_name)
        for name in val_files:
            shutil.copy2(val_img_dir / name, dst / "images/val" / name)
            lbl_name = Path(name).with_suffix(".txt").name
            shutil.copy2(val_lbl_dir / lbl_name, dst / "labels/val" / lbl_name)
        for name in neg_set:
            shutil.copy2(cand_dir / name, dst / "images/train" / name)
            (dst / "labels/train" / (Path(name).stem + ".txt")).touch()

        (dst / "data.yaml").write_text(
            f"path: {dst.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: Dent\n"
            "  1: Hole\n"
            "  2: Rusty\n",
            encoding="utf-8",
        )
        print(f"{var_name}: train={len(pos_files) + len(neg_set)} (pos={len(pos_files)} neg={len(neg_set)}) -> {dst}")


if __name__ == "__main__":
    main()
