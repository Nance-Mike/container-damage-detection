# -*- coding: utf-8 -*-
"""
数据预处理脚本：集装箱缺陷检测项目
"""
import os
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np
import yaml


def cv2_imread_cn(filepath, flags=cv2.IMREAD_COLOR):
    """支持中文路径的 imread（cv2.imread 不支持中文路径）"""
    return cv2.imdecode(np.fromfile(str(filepath), dtype=np.uint8), flags)


def cv2_imwrite_cn(filepath, img):
    """支持中文路径的 imwrite"""
    ext = Path(filepath).suffix
    result, encoded = cv2.imencode(ext, img)
    if result:
        encoded.tofile(str(filepath))
    return result

# 设置随机种子，保证可复现性
random.seed(42)
np.random.seed(42)

# 类别定义: 0=Dent(凹陷), 1=Hole(破洞), 2=Rusty(锈蚀)
CLASSES = {0: 'Dent', 1: 'Hole', 2: 'Rusty'}

def get_dominant_class(label_path):
    """获取标注文件中出现次数最多的类别，作为分层采样的依据"""
    if not label_path.exists():
        return -1 # 表示无目标背景图
    
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        return -1
    
    # YOLO 格式的第一列为类别 ID
    classes = [int(line.strip().split()[0]) for line in lines if line.strip()]
    if not classes:
        return -1
        
    counts = Counter(classes)
    return counts.most_common(1)[0][0]

def split_dataset(src_dir, dst_dir, val_ratio=0.15):
    """1. 训练/验证集划分"""
    print(">>> 开始划分数据集...")
    src_images_dir = src_dir / 'images' / 'train'
    src_labels_dir = src_dir / 'labels' / 'train'
    
    dst_images_train = dst_dir / 'images' / 'train'
    dst_images_val = dst_dir / 'images' / 'val'
    dst_labels_train = dst_dir / 'labels' / 'train'
    dst_labels_val = dst_dir / 'labels' / 'val'
    
    # 创建目标目录
    for d in [dst_images_train, dst_images_val, dst_labels_train, dst_labels_val]:
        d.mkdir(parents=True, exist_ok=True)
    
    image_files = list(src_images_dir.glob('*.jpg'))
    if not image_files:
        print(f"警告：未能在 {src_images_dir} 找到图片。")
        return []

    # 按主要类别进行分组，用于分层抽样
    class_groups = defaultdict(list)
    for img_path in image_files:
        label_path = src_labels_dir / (img_path.stem + '.txt')
        dom_class = get_dominant_class(label_path)
        class_groups[dom_class].append(img_path)
        
    train_files = []
    val_files = []
    
    # 对每个类别的图片集进行按比例划分
    for cls, files in class_groups.items():
        random.shuffle(files)
        val_size = int(len(files) * val_ratio)
        val_files.extend(files[:val_size])
        train_files.extend(files[val_size:])
        
    print(f"总图片数: {len(image_files)}, 分配给训练集: {len(train_files)}, 分配给验证集: {len(val_files)}")
    
    # 定义复制文件的辅助函数
    def copy_files(files, target_img_dir, target_lbl_dir):
        for img_path in files:
            shutil.copy(img_path, target_img_dir / img_path.name)
            label_path = src_labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.copy(label_path, target_lbl_dir / label_path.name)
            else:
                # 若源图片无标注文件，则在目标路径生成一个空文件
                open(target_lbl_dir / (img_path.stem + '.txt'), 'w').close()
                
    copy_files(train_files, dst_images_train, dst_labels_train)
    copy_files(val_files, dst_images_val, dst_labels_val)
    
    # 自动生成 data.yaml 配置文件
    yaml_data = {
        'path': str(dst_dir.absolute()).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'names': CLASSES
    }
    
    with open(dst_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        
    print(f">>> 数据集划分完成，配置文件 {dst_dir / 'data.yaml'} 已生成。\n")
    return train_files

def parse_labels(label_path):
    """解析 YOLO 标注文件，返回解析后的 [x_center, y_center, w, h] 列表"""
    bboxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bboxes.append([float(x) for x in parts[1:5]])
    return bboxes

def check_overlap(crop_box, bboxes, img_size=640):
    """判断裁剪区域 (x1, y1, x2, y2) 是否与任何原有的 YOLO 标注框重叠"""
    cx1, cy1, cx2, cy2 = crop_box
    for b in bboxes:
        bx_c, by_c, bw, bh = b
        # 将归一化坐标转换为原图的绝对坐标
        bx1 = (bx_c - bw/2) * img_size
        by1 = (by_c - bh/2) * img_size
        bx2 = (bx_c + bw/2) * img_size
        by2 = (by_c + bh/2) * img_size
        
        # 判断两个矩形是否有交集
        if not (cx2 <= bx1 or cx1 >= bx2 or cy2 <= by1 or cy1 >= by2):
            return True
    return False

def generate_negative_samples(src_dir, dst_dir, train_files, num_samples=600, crop_size=320):
    """2. 负样本生成：从训练图片中裁剪不含标注框的区域"""
    print(f">>> 开始生成负样本，目标数量约: {num_samples}...")
    neg_dir = dst_dir / 'negative_samples'
    neg_images_dir = neg_dir / 'images'
    neg_labels_dir = neg_dir / 'labels'
    
    neg_images_dir.mkdir(parents=True, exist_ok=True)
    neg_labels_dir.mkdir(parents=True, exist_ok=True)
    
    src_labels_dir = src_dir / 'labels' / 'train'
    
    count = 0
    # 随机打乱以增加负样本来源的多样性
    random.shuffle(train_files)
    
    for img_path in train_files:
        if count >= num_samples:
            break
            
        label_path = src_labels_dir / (img_path.stem + '.txt')
        bboxes = parse_labels(label_path)
        
        img = cv2_imread_cn(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 每张图尝试进行随机裁剪
        for _ in range(5):
            if count >= num_samples:
                break
                
            x1 = random.randint(0, max(0, w - crop_size))
            y1 = random.randint(0, max(0, h - crop_size))
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            
            # 如果该裁剪区域与任何已有缺陷标注不重合，则是干净的负样本
            if not check_overlap((x1, y1, x2, y2), bboxes, img_size=640):
                crop_img = img[y1:y2, x1:x2]
                out_name = f"neg_{img_path.stem}_{x1}_{y1}.jpg"
                cv2_imwrite_cn(str(neg_images_dir / out_name), crop_img)
                # 生成对应的空 txt 标注文件
                open(neg_labels_dir / f"neg_{img_path.stem}_{x1}_{y1}.txt", 'w').close()
                count += 1
                
    print(f">>> 负样本生成完成，实际共生成 {count} 张负样本。\n")

def preview_augmentations(sample_img_path, dst_dir):
    """3. 数据增强预览：展示增强效果对比图并保存"""
    print(f">>> 开始生成数据增强预览图，样本: {sample_img_path.name}...")
    img = cv2_imread_cn(sample_img_path)
    if img is None:
        print("图片加载失败，跳过预览。")
        return
        
    h, w = img.shape[:2]
    
    # 0. 原图
    orig = img.copy()
    
    # 1. 水平翻转
    flipped = cv2.flip(img, 1)
    
    # 2. 亮度调整 (全局变亮)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.5, 0, 255)
    bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 3. HSV 随机扰动
    hsv_jitter = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv_jitter[:, :, 0] = (hsv_jitter[:, :, 0] + random.randint(-20, 20)) % 180
    hsv_jitter[:, :, 1] = np.clip(hsv_jitter[:, :, 1] + random.randint(-40, 40), 0, 255)
    hsv_jitter[:, :, 2] = np.clip(hsv_jitter[:, :, 2] + random.randint(-40, 40), 0, 255)
    hsv_jitter = cv2.cvtColor(hsv_jitter.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 4. 随机裁剪并缩放回原大小
    crop_size = int(w * 0.7)
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    cropped = img[y1:y1+crop_size, x1:x1+crop_size]
    cropped_resized = cv2.resize(cropped, (w, h))
    
    # 用于在图上添加文字标签的辅助函数
    def add_text(image, text):
        res = image.copy()
        cv2.putText(res, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return res
        
    row1 = np.hstack([add_text(orig, "Original"), add_text(flipped, "Horizontal Flip"), add_text(bright, "Brightness")])
    row2 = np.hstack([add_text(hsv_jitter, "HSV Jitter"), add_text(cropped_resized, "Random Crop"), np.zeros_like(orig)])
    
    preview = np.vstack([row1, row2])
    
    out_path = dst_dir / 'augmentation_preview.jpg'
    cv2_imwrite_cn(str(out_path), preview)
    print(f">>> 数据增强预览图已保存至: {out_path}\n")

if __name__ == '__main__':
    # 定义输入和输出路径
    src_dir = Path(__file__).resolve().parents[1] / "数据集3713"
    dst_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    
    if not src_dir.exists():
        print(f"错误: 数据集源路径不存在 {src_dir}")
    else:
        print("=== 集装箱缺陷检测数据预处理开始 ===")
        # 1. 划分数据集 (15% 验证集，分层抽样)
        train_files = split_dataset(src_dir, dst_dir, val_ratio=0.15)
        
        # 2. 生成负样本 (无目标裁剪)
        if train_files:
            generate_negative_samples(src_dir, dst_dir, train_files, num_samples=600)
            
            # 3. 数据增强预览 (取一张训练集图片)
            preview_augmentations(train_files[0], dst_dir)
            
        print("=== 数据预处理全部完成 ===")
