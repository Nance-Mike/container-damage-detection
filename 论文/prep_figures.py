# -*- coding: utf-8 -*-
"""Prepare figures and code copies for the competition paper.

Also sanitizes unicode math symbols (in, approx, sigma, ->) inside the copied
appendix sources so that XeLaTeX listings render cleanly; src/ stays untouched.
"""
import os
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, '论文', 'figures')
CODE = os.path.join(ROOT, '论文', 'code')


def cp(src, dst):
    shutil.copyfile(os.path.join(ROOT, src), os.path.join(FIG, dst))
    print('cp', src, '->', dst)


def resize_cp(src, dst, maxw=2200, quality=90):
    im = Image.open(os.path.join(ROOT, src)).convert('RGB')
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    im.save(os.path.join(FIG, dst), quality=quality)
    print('resize', src, '->', dst, im.size)


cp('数据探索/02_类别分布.png', 'fig_class_dist.png')
cp('数据探索/03_缺陷数量分布.png', 'fig_boxes_per_image.png')
cp('数据探索/04_尺度分布.png', 'fig_scale_dist.png')
cp('数据探索/04_亮度对比度色彩分布.png', 'fig_quality.png')
cp('数据探索/08_位置热力图.png', 'fig_heatmap.png')
cp('data/processed/augmentation_preview.jpg', 'fig_aug_preview.jpg')
cp('results/evt_improved_with_neg/weibull_fit.png', 'fig_weibull.png')
cp('runs/detect/improved_with_neg/results.png', 'fig_results_curves.png')
cp('runs/detect/improved_with_neg/BoxPR_curve.png', 'fig_pr_curve.png')
cp('runs/detect/improved_with_neg/confusion_matrix_normalized.png', 'fig_cm.png')
cp('runs/detect/improved_with_neg/val_batch1_labels.jpg', 'fig_val_labels.jpg')
cp('runs/detect/improved_with_neg/val_batch1_pred.jpg', 'fig_val_pred.jpg')


def find_img_with(classes):
    imgdir = os.path.join(ROOT, 'data/processed/images/train')
    lbldir = os.path.join(ROOT, 'data/processed/labels/train')
    for name in sorted(os.listdir(imgdir)):
        if not name.endswith('.jpg'):
            continue
        lp = os.path.join(lbldir, name[:-4] + '.txt')
        if not os.path.exists(lp):
            continue
        with open(lp, encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        got = sorted({int(l.split()[0]) for l in lines})
        if got == classes:
            return os.path.join(imgdir, name), lines
    return None, None


names = {0: 'Dent 凹陷', 1: 'Hole 破洞', 2: 'Rusty 锈蚀'}
colors = {0: (30, 80, 255), 1: (255, 60, 60), 2: (255, 150, 20)}
font_big = ImageFont.truetype(r'C:\Windows\Fonts\simhei.ttf', 30)
font_small = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 26)
panels = []
for cls in [0, 1, 2]:
    path, lines = find_img_with([cls])
    if path is None:
        raise SystemExit('no single-class image for %d' % cls)
    im = Image.open(path).convert('RGB')
    w, h = im.size
    dr = ImageDraw.Draw(im)
    for line in lines[:8]:
        cid, cx, cy, bw, bh = map(float, line.split())
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        dr.rectangle([x1, y1, x2, y2], outline=colors[int(cid)], width=4)
        dr.text((x1, max(0, y1 - 34)), names[int(cid)], font=font_small, fill=colors[int(cid)])
    panels.append(im.resize((640, 640)))
mont = Image.new('RGB', (640 * 3 + 40, 680), (255, 255, 255))
for i, p in enumerate(panels):
    mont.paste(p, (10 + i * (640 + 10), 10))
drm = ImageDraw.Draw(mont)
drm.text((20, 655), 'Dent 凹陷', font=font_big, fill=colors[0])
drm.text((670, 655), 'Hole 破洞', font=font_big, fill=colors[1])
drm.text((1320, 655), 'Rusty 锈蚀', font=font_big, fill=colors[2])
mont.save(os.path.join(FIG, 'fig_defect_classes.jpg'), quality=92)
print('generated fig_defect_classes.jpg')


negdir = os.path.join(ROOT, 'data/processed/negative_samples/images')
negs = sorted(os.listdir(negdir))[:4]
cell = 400
mont = Image.new('RGB', (cell * 2 + 30, cell * 2 + 70), (255, 255, 255))
for i, n in enumerate(negs):
    im = Image.open(os.path.join(negdir, n)).convert('RGB').resize((cell, cell))
    x, y = 10 + (i % 2) * (cell + 10), 60 + (i // 2) * (cell + 10)
    mont.paste(im, (x, y))
drm = ImageDraw.Draw(mont)
drm.text((10, 10), '负样本示例（无破损区域裁剪）', font=font_big, fill=(0, 0, 0))
mont.save(os.path.join(FIG, 'fig_neg_samples.jpg'), quality=92)
print('generated fig_neg_samples.jpg')


preddir = os.path.join(ROOT, 'runs/visualize/predict')
sel = ['1.jpg', '102.jpg', '200.jpg', '388.jpg']
cell = 600
mont = Image.new('RGB', (cell * 2 + 30, cell * 2 + 30), (255, 255, 255))
for i, n in enumerate(sel):
    im = Image.open(os.path.join(preddir, n)).convert('RGB').resize((cell, cell))
    x, y = 10 + (i % 2) * (cell + 10), 10 + (i // 2) * (cell + 10)
    mont.paste(im, (x, y))
mont.save(os.path.join(FIG, 'fig_detect_results.jpg'), quality=92)
print('generated fig_detect_results.jpg')


for f in ['preprocess.py', 'train_yolo.py', 'evt_classifier.py', 'wd_focal_loss.py',
          'evaluate.py', 'eda.py', 'build_neg_variants.py', 'eval_model.py',
          'robustness_eval.py', 'run_evt_probe.py']:
    shutil.copyfile(os.path.join(ROOT, 'src', f), os.path.join(CODE, f))
print('code copied')

# sanitize unicode symbols in the appendix code copies (comments/docstrings only)
SYMBOL_TABLE = {'∈': 'in', '≈': '~', 'σ': 'sigma', '→': '->'}
for f in os.listdir(CODE):
    p = os.path.join(CODE, f)
    data = open(p, encoding='utf-8').read()
    for src, dst in SYMBOL_TABLE.items():
        if src in data:
            print('sanitize', f, repr(src), '->', dst, 'x', data.count(src))
            data = data.replace(src, dst)
    open(p, 'w', encoding='utf-8', newline='').write(data)
print('code sanitized')

print('---- figure sizes ----')
for f in sorted(os.listdir(FIG)):
    print(f, os.path.getsize(os.path.join(FIG, f)) // 1024, 'KB')
