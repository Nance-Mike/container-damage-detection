#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 全国大学生数学建模竞赛电子版提交材料整理与打包工具。

功能：
  1. 将四类必需材料（Word 论文、PDF 论文、全部程序源码、AI 使用说明）
     统一归集到新建文件夹；
  2. 程序源码含多个文件/子目录时保持原有目录结构；
  3. 目标文件夹命名为 "2026+队号"；
  4. 归集后自动压缩为 zip，并校验大小（支撑材料限 20MB）；
  5. 压缩前校验材料完整性：四类必需材料任一缺失即列出缺失项并中止；
  6. 同名文件冲突时输出提示，不静默覆盖。

用法（项目根目录下）：
    python pack_submission.py --team 20261234567890
    python pack_submission.py --team 20261234567890 --out D:\\提交
    python pack_submission.py --check              # 只校验材料，不打包
    python pack_submission.py --team ... --force   # 目标已存在时允许覆盖（打印提示）

说明：
  - 队号为 12 位全国统一编号（承诺书口径），文件夹名 = "2026" + 队号；
  - 打包内容不含承诺书、编号专用页及任何身份信息；
  - 正文以 PDF 为准，Word 版由同一 LaTeX 源码生成（论文/make_word.py）。
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ------------------------- 四类材料默认路径 -------------------------
WORD_PAPER = BASE / "论文" / "main.docx"
PDF_PAPER = BASE / "论文" / "main.pdf"
AI_DETAIL = BASE / "AI工具使用详情.pdf"

# 程序源码：目录整体复制（保持结构）+ 散文件（按相对路径复制）
CODE_DIRS = [
    "src",
]
CODE_FILES = [
    "论文/scripts/run_val.py",
    "论文/scripts/verify_pdf.py",
    "论文/prep_ablation.py",
    "论文/prep_figures.py",
    "论文/regenerate_weibull.py",
    "论文/restyle_figures.py",
    "论文/restyle_montages.py",
    "论文/make_word.py",
    "数据探索分析.ipynb",
]
EXTRA_FILES = [
    "test_result.csv",
]

MAX_ZIP_MB = 20  # 官方支撑材料大小上限
TEAM_RE = re.compile(r"^\d{6,16}$")


def required_items() -> dict:
    return {
        "Word 论文": WORD_PAPER,
        "PDF 论文": PDF_PAPER,
        "AI 使用说明（PDF）": AI_DETAIL,
        "程序源码目录": CODE_DIRS,
    }


def validate_materials() -> list:
    """返回缺失项清单；为空表示四类材料齐全。"""
    missing = []
    items = required_items()
    for name, path in items.items():
        if isinstance(path, Path) and not path.exists():
            missing.append(f"{name}：{path.relative_to(BASE)} 不存在")
    for rel in CODE_DIRS:
        p = BASE / rel
        if not p.is_dir():
            missing.append(f"程序源码目录：{rel} 不存在")
        elif not list(p.rglob("*.py")):
            missing.append(f"程序源码目录：{rel} 下没有 .py 文件")
    for rel in CODE_FILES + EXTRA_FILES:
        p = BASE / rel
        if not p.exists():
            missing.append(f"程序源码/附加文件：{rel} 不存在")
    return missing


def copy_tree_preserve(src: Path, dst_root: Path, rel_root: Path, conflicts: list, force: bool):
    """按相对结构复制目录/文件；遇冲突记入 conflicts（不静默覆盖）。"""
    if src.is_dir():
        for item in src.rglob("*"):
            if item.is_dir():
                continue
            if "__pycache__" in item.parts or item.suffix in (".pyc", ".pyo"):
                continue
            rel = item.relative_to(rel_root)
            dst = dst_root / rel
            _copy_one(item, dst, conflicts, force)
    else:
        rel = src.relative_to(rel_root)
        _copy_one(src, dst_root / rel, conflicts, force)


def _copy_one(src: Path, dst: Path, conflicts: list, force: bool):
    if dst.exists() and not force:
        conflicts.append(f"{dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and force:
        print(f"[pack] 覆盖同名文件：{dst}")
    shutil.copy2(src, dst)


def build_manifest(folder: Path) -> str:
    lines = [
        "2026 高教社杯全国大学生数学建模竞赛 电子版提交材料文件清单",
        "=" * 64,
        "",
        "一、材料构成（四类必需材料 + 附加结果文件）",
        "  1. Word 论文：main.docx（由论文/main.tex 生成，正文以 PDF 为准）",
        "  2. PDF 论文：main.pdf（与纸质版内容一致，第一页为摘要页）",
        "  3. 全部程序源码：程序源码/（保持原目录结构）",
        "  4. AI 使用说明：AI工具使用详情.pdf",
        "  5. 附加：test_result.csv（问题二检测结果）",
        "",
        "二、文件明细",
        "",
    ]
    total = 0
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            total += size
            lines.append(f"{size:>12,}  {f.relative_to(folder).as_posix()}")
    lines.append("")
    lines.append(f"文件总数：{sum(1 for f in folder.rglob('*') if f.is_file())}")
    lines.append(f"总大小：{total / 1048576:.2f} MB")
    lines.append("")
    lines.append("三、合规说明")
    lines.append("  - 本压缩包不含承诺书、编号专用页及任何可识别参赛者身份、学校、赛区的信息；")
    lines.append("  - AI 工具使用声明已置于论文参考文献之前，详细使用情况见 AI工具使用详情.pdf；")
    lines.append("  - 支撑材料文件列表已写入论文附录。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="CUMCM 2026 电子版提交材料打包工具")
    ap.add_argument("--team", help="12 位全国统一队号（文件夹名 = 2026+队号）")
    ap.add_argument("--out", default=str(BASE.parent), help="输出父目录（默认选题D 的上一级）")
    ap.add_argument("--check", action="store_true", help="只校验四类材料是否齐全，不打包")
    ap.add_argument("--force", action="store_true", help="目标已存在时允许覆盖（仍会打印提示）")
    args = ap.parse_args()

    missing = validate_materials()
    if missing:
        print("[pack] 材料不完整，打包流程中止。缺失项清单：")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    print("[pack] 四类必需材料校验通过。")
    if args.check:
        print("[pack] --check 模式，未创建任何文件。")
        return

    if not args.team:
        print("[pack] 错误：请提供 --team（12 位全国统一队号）。", file=sys.stderr)
        sys.exit(2)
    if not TEAM_RE.match(args.team):
        print("[pack] 错误：队号应为数字（例如 2026104 或 202612345678）。", file=sys.stderr)
        sys.exit(2)
    if len(args.team) != 12:
        print(f"[pack] 提示：当前队号 {len(args.team)} 位（赛区短队号或 12 位全国统一编号均可，"
              "按你提供的队号原样执行）。")

    folder_name = "2026" + args.team
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / folder_name

    conflicts = []
    if target.exists():
        existing = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        if existing and not args.force:
            print(f"[pack] 目标文件夹已存在：{target}")
            print("[pack] 同名文件冲突（不静默覆盖，如需覆盖请加 --force）：")
            for e in existing[:50]:
                print(f"  - {e}")
            sys.exit(3)
        elif existing and args.force:
            print(f"[pack] 目标文件夹已存在，--force 模式将覆盖同名文件：{target}")
            shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)

    # ---- 1/2. 四类材料归集（程序源码保持目录结构） ----
    word_dst = target / "Word论文" / WORD_PAPER.name
    pdf_dst = target / "PDF论文" / PDF_PAPER.name
    ai_dst = target / "AI工具使用说明" / AI_DETAIL.name
    code_root = target / "程序源码"

    _copy_one(WORD_PAPER, word_dst, conflicts, args.force)
    _copy_one(PDF_PAPER, pdf_dst, conflicts, args.force)
    _copy_one(AI_DETAIL, ai_dst, conflicts, args.force)
    for rel in CODE_DIRS:
        copy_tree_preserve(BASE / rel, code_root, BASE, conflicts, args.force)
    for rel in CODE_FILES:
        copy_tree_preserve(BASE / rel, code_root, BASE, conflicts, args.force)
    for rel in EXTRA_FILES:
        _copy_one(BASE / rel, target / Path(rel).name, conflicts, args.force)

    if conflicts:
        print("[pack] 存在同名文件冲突（未覆盖）：")
        for c in conflicts:
            print(f"  - {c}")
        sys.exit(3)

    # ---- 文件清单 ----
    (target / "文件清单.txt").write_text(build_manifest(target), encoding="utf-8")

    # ---- 4. 压缩 ----
    zip_path = out_dir / f"{folder_name}.zip"
    if zip_path.exists() and not args.force:
        print(f"[pack] zip 已存在：{zip_path}（不静默覆盖，如需覆盖请加 --force）")
        sys.exit(3)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(target.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir).as_posix())

    size_mb = zip_path.stat().st_size / 1048576
    print(f"[pack] 归集完成：{target}")
    print(f"[pack] 压缩完成：{zip_path}（{size_mb:.2f} MB）")
    if size_mb > MAX_ZIP_MB:
        print(f"[pack] 错误：压缩包超过 {MAX_ZIP_MB} MB 上限，请精简材料后重试。", file=sys.stderr)
        sys.exit(4)
    print("[pack] 大小校验通过（≤ 20MB）。")


if __name__ == "__main__":
    main()
