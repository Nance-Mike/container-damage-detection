# -*- coding: utf-8 -*-
"""Regenerate sample montages with unified Okabe-Ito class colors."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "论文" / "figures"
IMG = ROOT / "data" / "processed" / "images" / "train"
LBL = ROOT / "data" / "processed" / "labels" / "train"

NAMES = {0: "Dent 凹陷", 1: "Hole 破洞", 2: "Rusty 锈蚀"}
# RGB, Okabe-Ito palette A
COLORS = {
    0: (0, 114, 178),    # Steel Blue
    1: (230, 159, 0),    # Warm Orange
    2: (0, 158, 115),    # Bluish Green
}
FONT_LABEL = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 26)
FONT_TITLE = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 30)


def find_single_class_image(cls):
    for name in sorted(IMG.glob("*.jpg")):
        lp = LBL / (name.stem + ".txt")
        if not lp.exists():
            continue
        lines = [ln.strip() for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        got = sorted({int(ln.split()[0]) for ln in lines})
        if got == [cls]:
            return name, lines
    raise SystemExit(f"no single-class image for class {cls}")


def draw_panel(im, lines, cls):
    w, h = im.size
    d = ImageDraw.Draw(im)
    color = COLORS[cls]
    for ln in lines[:8]:
        cid, cx, cy, bw, bh = map(float, ln.split())
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        d.rectangle([x1, y1, x2, y2], outline=color, width=5)
        d.rectangle([x1, max(0, y1 - 34), x1 + 158, y1], fill=color)
        d.text((x1 + 6, max(0, y1 - 31)), NAMES[cls], font=FONT_LABEL, fill=(255, 255, 255))
    return im


def main():
    panels = []
    for cls in (0, 1, 2):
        path, lines = find_single_class_image(cls)
        im = Image.open(path).convert("RGB").resize((640, 640))
        panels.append(draw_panel(im, lines, cls))

    cell = 640
    pad = 12
    title_h = 46
    mont = Image.new("RGB", (cell * 3 + pad * 4, cell + pad * 2 + title_h), (255, 255, 255))
    d = ImageDraw.Draw(mont)
    for i, p in enumerate(panels):
        x = pad + i * (cell + pad)
        d.rectangle([x, pad, x + cell, pad + cell], outline=(220, 220, 220), width=1)
        mont.paste(p, (x + 1, pad + 1))
        tw = d.textlength(NAMES[i], font=FONT_TITLE)
        d.text((x + (cell - tw) / 2, pad + cell + 8), NAMES[i],
               font=FONT_TITLE, fill=COLORS[i])
    mont.save(FIG / "fig_defect_classes.jpg", quality=94)
    print("saved", FIG / "fig_defect_classes.jpg")


if __name__ == "__main__":
    main()
