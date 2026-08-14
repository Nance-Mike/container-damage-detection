#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate check for a compiled CUMCM LaTeX paper PDF.

Usage:
    python verify_pdf.py path/to/main.pdf [--expect-pages-max 120]

Checks: A4 page size, page count, undefined references ("??"), abstract page
contains 摘要/关键词, fonts embedded, images embedded. Exits 0 on pass, 1 on
failure. Requires PyMuPDF (fitz).
"""
import argparse
import sys

import fitz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf', help='path to compiled main.pdf')
    ap.add_argument('--expect-pages-max', type=int, default=120)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    problems: list[str] = []
    info: list[str] = [f'pages: {len(doc)}']

    rect = doc[0].rect
    is_a4 = abs(rect.width - 595.28) < 2 and abs(rect.height - 841.89) < 2
    info.append(f'page size (pt): {rect.width:.2f} x {rect.height:.2f} (A4: {is_a4})')
    if not is_a4:
        problems.append('page 1 is not A4')
    if len(doc) > args.expect_pages_max:
        problems.append(f'page count {len(doc)} exceeds max {args.expect_pages_max}')

    text = '\n'.join(page.get_text() for page in doc)
    nq = text.count('??')
    info.append(f'"??" count: {nq}')
    if nq:
        problems.append('undefined references remain ("??")')

    p1 = doc[0].get_text()
    ok_abs = ('摘' in p1) and ('关键词' in p1)
    info.append(f'abstract page ok: {ok_abs}')
    if not ok_abs:
        problems.append('page 1 is not a proper abstract page (摘要/关键词 missing)')

    fonts = set()
    for page in doc:
        for f in page.get_fonts():
            fonts.add(f[3])
    info.append(f'fonts ({len(fonts)}): {", ".join(sorted(fonts)[:12])}')

    imgs = sum(len(page.get_images(full=True)) for page in doc)
    info.append(f'embedded images: {imgs}')

    print('\n'.join(info))
    if problems:
        print('PROBLEMS:')
        for p in problems:
            print(' -', p)
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
