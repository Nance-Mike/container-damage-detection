# -*- coding: utf-8 -*-
"""Generate ablation comparison figure from the three training runs."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'C:\Users\administrator\Desktop\2026数模国赛\选题D'
RUNS = {
    '基线 YOLOv8n': 'runs/baseline/results.csv',
    '改进 YOLOv8s+Copy-Paste': 'runs/improved/results.csv',
    '改进+负样本（最终）': 'runs/detect/improved_with_neg/results.csv',
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

colors = ['#7f8fa6', '#e1b12c', '#273c75']
best_rows = []
for (label, path), color in zip(RUNS.items(), colors):
    df = pd.read_csv(os.path.join(ROOT, path))
    ep = df['epoch']
    axes[0].plot(ep, df['metrics/mAP50(B)'], label=label, color=color, lw=1.6)
    axes[0].plot(ep, df['metrics/mAP50-95(B)'], color=color, lw=1.2, ls='--')
    i = df['metrics/mAP50(B)'].idxmax()
    best_rows.append((label, df.loc[i, 'metrics/mAP50(B)'],
                      df.loc[i, 'metrics/mAP50-95(B)']))

axes[0].set_xlabel('训练轮数 Epoch')
axes[0].set_ylabel('mAP')
axes[0].set_title('训练过程 mAP 曲线（实线 mAP@0.5，虚线 mAP@0.5:0.95）')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

labels = [r[0] for r in best_rows]
m50 = [r[1] for r in best_rows]
m5095 = [r[2] for r in best_rows]
x = range(len(labels))
axes[1].bar([i - 0.2 for i in x], m50, width=0.4, color='#e1b12c', label='mAP@0.5')
axes[1].bar([i + 0.2 for i in x], m5095, width=0.4, color='#273c75', label='mAP@0.5:0.95')
for i, v in enumerate(m50):
    axes[1].text(i - 0.2, v + 0.004, '%.3f' % v, ha='center', fontsize=9)
for i, v in enumerate(m5095):
    axes[1].text(i + 0.2, v + 0.004, '%.3f' % v, ha='center', fontsize=9)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylabel('mAP')
axes[1].set_title('验证集最优 mAP 对比')
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 0.5)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
out = os.path.join(ROOT, '论文', 'figures', 'fig_ablation.png')
plt.savefig(out, dpi=160)
print('saved', out)
