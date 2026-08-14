# -*- coding: utf-8 -*-
"""Validate the final model and dump per-class metrics to JSON."""
import json
import os

os.makedirs(os.environ.get('YOLO_CONFIG_DIR', os.path.join(os.environ['TEMP'], 'yolo_cfg_val3')),
            exist_ok=True)

from ultralytics import YOLO

m = YOLO('runs/detect/improved_with_neg/weights/best.pt')
res = m.val(data='data/processed/data.yaml', project='runs/detect/results/eval_final2', name='val',
            exist_ok=True, save_json=True, plots=False, batch=16, device=0,
            workers=0, verbose=False)

out = {
    'map50': float(res.box.map50),
    'map50_95': float(res.box.map),
    'instances': int(res.box.nc),
    'precision': [float(v) for v in res.box.p],
    'recall': [float(v) for v in res.box.r],
    'ap50': [float(v) for v in res.box.ap50],
    'ap50_95': [float(v) for v in res.box.ap],
    'f1': [float(v) for v in res.box.f1],
    'speed': {k: float(v) for k, v in res.speed.items()},
}
print('save_dir:', res.save_dir)
os.makedirs(res.save_dir, exist_ok=True)
with open(os.path.join(res.save_dir, 'val_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
