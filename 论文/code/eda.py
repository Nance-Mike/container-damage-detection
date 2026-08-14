import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from collections import defaultdict
import itertools

# 设置中文字体，解决matplotlib中文显示问题
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


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

class DatasetEDA:
    def __init__(self, data_root, output_dir):
        """
        初始化EDA类
        :param data_root: 数据集根目录
        :param output_dir: 结果输出目录
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.img_train_dir = self.data_root / 'images' / 'train'
        self.img_test_dir = self.data_root / 'images' / 'test'
        self.lbl_train_dir = self.data_root / 'labels' / 'train'
        self.lbl_test_dir = self.data_root / 'labels' / 'test'
        
        self.classes = {0: 'Dent(凹陷)', 1: 'Hole(破洞)', 2: 'Rusty(锈蚀)'}
        self.colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)} # BGR格式用于cv2绘制
        
        # 存储提取出的标签信息用于分析
        self.all_labels = []
        self.boxes_per_image = []
        self.co_occurrences = []

    def run_all(self):
        """运行所有EDA步骤"""
        print("开始进行数据探索分析 (EDA)...")
        self.parse_labels()
        self.basic_statistics()
        self.target_size_analysis()
        self.image_quality_analysis()
        self.visualize_bboxes(num_samples=10)
        print(f"\\n分析完成，所有结果图表已保存至: {self.output_dir}")

    def parse_labels(self):
        """解析所有的标注文件，提取边界框信息"""
        print("正在解析标注文件...")
        for label_file in self.lbl_train_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            self.boxes_per_image.append(len(lines))
            
            # 用于记录当前图片包含的类别，用于共现分析
            classes_in_img = set()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    self.all_labels.append({
                        'class_id': cls_id,
                        'class_name': self.classes[cls_id],
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'area': width * height * 640 * 640, # 实际像素面积
                        'aspect_ratio': width / height if height > 0 else 0
                    })
                    classes_in_img.add(cls_id)
            
            if classes_in_img:
                self.co_occurrences.append(tuple(sorted(list(classes_in_img))))

        self.df_labels = pd.DataFrame(self.all_labels)

    def basic_statistics(self):
        """基础统计"""
        print("\\n--- 基础统计 ---")
        train_imgs = list(self.img_train_dir.glob('*.jpg'))
        test_imgs = list(self.img_test_dir.glob('*.jpg'))
        print(f"训练集图片数量: {len(train_imgs)}")
        print(f"测试集图片数量: {len(test_imgs)}")
        
        if self.df_labels.empty:
            print("未找到任何标注信息！")
            return

        # 1. 各类别标注框总数和占比
        class_counts = self.df_labels['class_name'].value_counts()
        print("\\n各类别缺陷数量及占比:")
        for name, count in class_counts.items():
            print(f"  {name}: {count} ({count/len(self.df_labels)*100:.2f}%)")
            
        plt.figure(figsize=(8, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('各类别缺陷数量统计')
        plt.ylabel('数量')
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close()

        # 2. 每张图片中缺陷数量的分布（直方图）
        plt.figure(figsize=(8, 6))
        sns.histplot(self.boxes_per_image, bins=range(0, max(self.boxes_per_image)+2), discrete=True)
        plt.title('每张图片的缺陷数量分布')
        plt.xlabel('缺陷数量')
        plt.ylabel('图片数量')
        plt.savefig(self.output_dir / 'boxes_per_image_dist.png')
        plt.close()

        # 3. 统计多类别共现情况
        co_counts = pd.Series(self.co_occurrences).value_counts()
        print("\\n多类别共现情况:")
        for combo, count in co_counts.items():
            combo_names = " + ".join([self.classes[c] for c in combo])
            print(f"  {combo_names}: {count}张图片")

    def target_size_analysis(self):
        """目标尺寸分析"""
        print("\\n--- 目标尺寸分析 ---")
        if self.df_labels.empty:
            return

        # 1. 按COCO标准划分目标大小
        # 小目标(<32^2=1024), 中目标(1024~9216), 大目标(>9216)
        small = self.df_labels[self.df_labels['area'] < 1024]
        medium = self.df_labels[(self.df_labels['area'] >= 1024) & (self.df_labels['area'] <= 9216)]
        large = self.df_labels[self.df_labels['area'] > 9216]
        
        print("按COCO标准划分目标尺寸:")
        print(f"  小目标 (<32^2): {len(small)} ({len(small)/len(self.df_labels)*100:.2f}%)")
        print(f"  中目标 (32^2~96^2): {len(medium)} ({len(medium)/len(self.df_labels)*100:.2f}%)")
        print(f"  大目标 (>96^2): {len(large)} ({len(large)/len(self.df_labels)*100:.2f}%)")

        # 2. 各类别的宽高比分布
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df_labels, x='class_name', y='aspect_ratio')
        plt.title('各类别的宽高比 (Aspect Ratio) 分布')
        plt.ylabel('宽高比 (宽/高)')
        plt.ylim(0, 5) # 限制Y轴便于观察
        plt.savefig(self.output_dir / 'aspect_ratio_boxplot.png')
        plt.close()

        # 3. 各类别标注框面积分布（箱线图）
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df_labels, x='class_name', y='area')
        plt.title('各类别缺陷的面积分布')
        plt.ylabel('面积 (像素^2)')
        plt.yscale('log') # 使用对数刻度更好展示
        plt.savefig(self.output_dir / 'area_boxplot.png')
        plt.close()

    def image_quality_analysis(self):
        """图像质量分析"""
        print("\\n--- 图像质量分析 (采样100张) ---")
        train_imgs = list(self.img_train_dir.glob('*.jpg'))
        sample_imgs = random.sample(train_imgs, min(100, len(train_imgs)))
        
        brightness_list = []
        contrast_list = []
        
        for img_path in sample_imgs:
            img = cv2_imread_cn(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                brightness = np.mean(img)
                contrast = np.std(img)
                brightness_list.append(brightness)
                contrast_list.append(contrast)
                
        print(f"  亮度均值: {np.mean(brightness_list):.2f}, 范围: [{np.min(brightness_list):.2f}, {np.max(brightness_list):.2f}]")
        print(f"  对比度均值: {np.mean(contrast_list):.2f}, 范围: [{np.min(contrast_list):.2f}, {np.max(contrast_list):.2f}]")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(brightness_list, ax=axes[0], kde=True)
        axes[0].set_title('图像亮度分布 (100张采样)')
        axes[0].set_xlabel('平均亮度 (灰度值)')
        
        sns.histplot(contrast_list, ax=axes[1], kde=True)
        axes[1].set_title('图像对比度分布 (100张采样)')
        axes[1].set_xlabel('对比度 (标准差)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'image_quality.png')
        plt.close()

    def visualize_bboxes(self, num_samples=10):
        """随机可视化训练图片及其标注框"""
        print(f"\\n--- 正在生成 {num_samples} 张示例图片的可视化 ---")
        train_imgs = list(self.img_train_dir.glob('*.jpg'))
        sample_imgs = random.sample(train_imgs, min(num_samples, len(train_imgs)))
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        for img_path in sample_imgs:
            img = cv2_imread_cn(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            label_path = self.lbl_train_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_c, y_c, bw, bh = map(float, parts[1:5])
                            
                            # 反归一化
                            x1 = int((x_c - bw/2) * w)
                            y1 = int((y_c - bh/2) * h)
                            x2 = int((x_c + bw/2) * w)
                            y2 = int((y_c + bh/2) * h)
                            
                            color = self.colors.get(cls_id, (255, 255, 255))
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            
                            # 绘制标签文本
                            label = self.classes[cls_id]
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out_path = viz_dir / f"viz_{img_path.name}"
            cv2_imwrite_cn(out_path, img)

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "数据集3713"
    output_dir = project_root / "数据探索"
    
    eda = DatasetEDA(data_root, output_dir)
    eda.run_all()
