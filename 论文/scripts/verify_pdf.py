#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify a compiled CUMCM LaTeX paper PDF (quality gate).

Usage:
    python verify_pdf.py [main.pdf] [--expect-pages-max 120]

Checks: A4 page size, page count, undefined references ("??"), required
section headings, abstract page (摘要/关键词), fonts, embedded images, and
appendix code rendering (including Chinese comments). Prints diagnostics and
exits 0 on pass / 1 on failure. Requires PyMuPDF (fitz).
"""
import argparse
import re
import sys

import fitz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default="main.pdf",
                    help="path to the compiled main.pdf")
    ap.add_argument("--expect-pages-max", type=int, default=120,
                    help="fail if the PDF has more pages than this")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    problems: list[str] = []
    info: list[str] = [f"pages: {len(doc)}"]

    # A4 page size
    rect = doc[0].rect
    is_a4 = abs(rect.width - 595.28) < 2 and abs(rect.height - 841.89) < 2
    info.append(f"page size (pt): {rect.width:.2f} x {rect.height:.2f} (A4: {is_a4})")
    if not is_a4:
        problems.append("page 1 is not A4")
    if len(doc) > args.expect_pages_max:
        problems.append(f"page count {len(doc)} exceeds max {args.expect_pages_max}")

    # Per-page text
    full = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    all_text = "\n".join(t for _, t in full)

    # Undefined references
    nq = all_text.count("??")
    info.append(f'"??" count: {nq}')
    if nq:
        problems.append('undefined references remain ("??")')

    # Required section headings
    heads = ["一、问题重述", "二、问题分析", "三、模型假设", "四、符号说明",
             "五、数据处理与探索性分析", "六、问题一：基于极值理论",
             "七、问题二：基于 YOLOv8", "八、问题三：模型多维度评估",
             "九、模型评价与推广", "参考文献", "附录A", "附录B", "附录C"]
    for h in heads:
        pages = [p for p, t in full if h.replace(" ", "") in t.replace(" ", "")]
        info.append(f"{h}: pages {pages[:3]}")
        if not pages:
            problems.append(f"heading not found: {h}")

    # Abstract page
    p1 = full[0][1]
    ok_abs = "摘" in p1 and "关键词" in p1
    info.append(f"abstract page ok: {ok_abs}")
    if not ok_abs:
        problems.append("page 1 is not a proper abstract page (摘要/关键词 missing)")

    # Fonts
    fonts = set()
    for page in doc:
        for f in page.get_fonts():
            fonts.add(f[3])
    info.append(f"fonts ({len(fonts)}): {', '.join(sorted(fonts)[:12])}")

    # Appendix code rendering: scan from the appendix start to the end
    app_start = next((i for i, (_, t) in enumerate(full) if "附录A" in t), None)
    if app_start is None:
        problems.append("appendix not found")
    else:
        t_app = "\n".join(t for _, t in full[app_start:])
        has_code = "def " in t_app and "import " in t_app
        has_cn = "项目根目录" in t_app
        info.append(f"appendix pages {app_start + 1}-{len(doc)}: "
                    f"code rendered: {has_code}; Chinese comment rendered: {has_cn}")
        if not has_code or not has_cn:
            problems.append("appendix code or Chinese comments not rendered")

    # Caption numbering sample
    caption_lines = []
    for _, t in full:
        for line in t.splitlines():
            s = line.strip()
            if re.match(r"^(图|表)\s*\d+-\d+\s", s) or s.startswith("附录"):
                if s not in caption_lines:
                    caption_lines.append(s)
    info.append("sample captions: " + "; ".join(caption_lines[:8]))

    # Embedded images
    imgs = sum(len(page.get_images(full=True)) for page in doc)
    info.append(f"embedded images: {imgs}")

    print("\n".join(info))
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" -", p)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
