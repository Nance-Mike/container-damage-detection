# -*- coding: utf-8 -*-
"""YOLOv1 精读（上）公众号封面渲染：3 套方案，900x383（头条 2.35:1）"""
from PIL import Image, ImageDraw, ImageFont

W, H = 900, 383
OUT = r"C:\Users\administrator\Desktop\2026数模国赛\选题D\yolo系列论文\paper-notes\yolov1"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"

def font(path, size):
    return ImageFont.truetype(path, size)

def vgradient(draw, top, bottom):
    """垂直渐变底色"""
    for y in range(H):
        t = y / H
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (W, y)], fill=c)

def draw_grid(draw, x0, y0, cell, n, color, width=1):
    """n x n 网格"""
    for i in range(n + 1):
        draw.line([(x0 + i * cell, y0), (x0 + i * cell, y0 + n * cell)], fill=color, width=width)
        draw.line([(x0, y0 + i * cell), (x0 + n * cell, y0 + i * cell)], fill=color, width=width)

def draw_bbox(draw, x0, y0, x1, y1, color, label, label_bg, label_fg, tick=14, lw=5):
    """检测框：细矩形 + 四角加粗刻度 + 标签 chip"""
    draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    for cx, cy, dx, dy in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=color, width=lw)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=color, width=lw)
    f = font(FONT_BOLD, 20)
    tw = draw.textlength(label, font=f)
    pad = 10
    ly0 = y0 - 34 if y0 - 34 > 6 else y1 + 8
    draw.rectangle([x0 - 1, ly0, x0 + tw + pad * 2, ly0 + 30], fill=label_bg)
    draw.text((x0 + pad - 1, ly0 + 3), label, font=f, fill=label_fg)

def tag(draw, x, y, text, fg, accent):
    """小节标签：色块 + 文字"""
    draw.rectangle([x, y + 6, x + 14, y + 20], fill=accent)
    draw.text((x + 26, y), text, font=font(FONT_BOLD, 26), fill=fg)

# ============ 方案 A：深色科技风（主推） ============
img = Image.new("RGB", (W, H))
d = ImageDraw.Draw(img)
vgradient(d, (10, 24, 48), (18, 38, 76))            # 深藏青渐变
# 右侧 7x7 网格（低透明度白线 -> 用暗蓝模拟）
grid_c = (52, 78, 128)
cell, n = 52, 7
gx, gy = W - cell * n - 40, (H - cell * n) // 2
draw_grid(d, gx, gy, cell, n, grid_c, 1)
# 高亮一个格子 + 检测框
hx, hy = gx + 2 * cell, gy + 2 * cell
d.rectangle([hx, hy, hx + cell, hy + cell], fill=(38, 62, 108))
RED = (255, 82, 66)
draw_bbox(d, hx - 26, hy - 26, hx + 2 * cell + 26, hy + 2 * cell + 26, RED, "object 0.99", RED, (255, 255, 255))
# 文案区
tag(d, 60, 74, "YOLOv1 论文精读 · 上篇", (158, 178, 214), RED)
d.text((56, 118), "只看一眼", font=font(FONT_BOLD, 104), fill=(255, 255, 255))
d.text((60, 268), "把目标检测，变成一次回归", font=font(FONT_REG, 34), fill=(172, 190, 224))
img.save(OUT + r"\cover_A_dark.png")

# ============ 方案 B：浅色学术风 ============
img = Image.new("RGB", (W, H), (248, 246, 241))      # 米白纸感
d = ImageDraw.Draw(img)
# 顶部细红条
d.rectangle([0, 0, W, 8], fill=(217, 56, 43))
# 右侧浅灰网格
grid_c = (216, 210, 198)
gx, gy = W - cell * n - 40, (H - cell * n) // 2
draw_grid(d, gx, gy, cell, n, grid_c, 1)
hx, hy = gx + 2 * cell, gy + 2 * cell
d.rectangle([hx, hy, hx + cell, hy + cell], fill=(236, 231, 220))
RED_B = (217, 56, 43)
draw_bbox(d, hx - 26, hy - 26, hx + 2 * cell + 26, hy + 2 * cell + 26, RED_B, "object 0.99", RED_B, (255, 255, 255))
# 文案区
tag(d, 60, 74, "YOLOv1 论文精读 · 上篇", (107, 101, 92), RED_B)
d.text((56, 118), "只看一眼", font=font(FONT_BOLD, 104), fill=(20, 20, 20))
d.text((60, 268), "把目标检测，变成一次回归", font=font(FONT_REG, 34), fill=(96, 91, 82))
img.save(OUT + r"\cover_B_light.png")

# ============ 方案 C：克莱因蓝撞色风 ============
img = Image.new("RGB", W and (W, H), (0, 47, 167))   # 克莱因蓝
d = ImageDraw.Draw(img)
YELLOW = (255, 216, 77)
# 右侧网格（深蓝线条）
grid_c = (28, 82, 200)
gx, gy = W - cell * n - 40, (H - cell * n) // 2
draw_grid(d, gx, gy, cell, n, grid_c, 1)
hx, hy = gx + 2 * cell, gy + 2 * cell
d.rectangle([hx, hy, hx + cell, hy + cell], fill=(20, 68, 190))
draw_bbox(d, hx - 26, hy - 26, hx + 2 * cell + 26, hy + 2 * cell + 26, YELLOW, "object 0.99", YELLOW, (0, 47, 167))
# 文案区
tag(d, 60, 74, "YOLOv1 论文精读 · 上篇", (150, 178, 240), YELLOW)
d.text((56, 118), "只看一眼", font=font(FONT_BOLD, 104), fill=(255, 216, 77))
d.text((60, 268), "把目标检测，变成一次回归", font=font(FONT_REG, 34), fill=(190, 206, 248))
img.save(OUT + r"\cover_C_blue.png")

print("done: cover_A_dark.png / cover_B_light.png / cover_C_blue.png")
