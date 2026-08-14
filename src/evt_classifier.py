# -*- coding: utf-8 -*-
"""
基于极值理论 (EVT) 的集装箱缺陷分类器
解决问题1：在全缺陷训练集中识别无缺陷图片

数学推导详见: src/evt_derivation.tex

工作流程:
    1. 用训练好的 YOLO 模型对验证集推理，提取每张图的最大检测置信度
    2. 用 Weibull 分布拟合置信度分布
    3. 对测试图片计算缺陷概率 P_defect
    4. 通过最优阈值 tau* 进行二分类

使用方法:
    E:\\python2025\\python.exe src/evt_classifier.py \\
        --weights runs/baseline/weights/best.pt \\
        --val-dir data/processed/images/val \\
        --test-dir 数据集3713/images/test \\
        --save-dir results/evt
"""
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import weibull_min
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 随机种子
np.random.seed(42)


def cv2_imread_cn(filepath, flags=cv2.IMREAD_COLOR):
    """支持中文路径的 imread"""
    return cv2.imdecode(np.fromfile(str(filepath), dtype=np.uint8), flags)


def extract_max_confidences(model, image_dir: str, conf_threshold: float = 0.01) -> dict:
    """
    对目录中的图片批量推理，提取每张图的最大检测置信度

    Args:
        model: Ultralytics YOLO 模型
        image_dir: 图片目录路径
        conf_threshold: 最低置信度阈值（保留更多检测以获取分布信息）

    Returns:
        dict: {image_name: max_confidence}
    """
    print(f"正在对 {image_dir} 进行推理...")
    results = model.predict(
        source=image_dir,
        conf=conf_threshold,
        iou=0.5,
        verbose=False,
        save=False,
        device=0,
    )

    confidences = {}
    for r in results:
        img_name = Path(r.path).stem
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            max_conf = float(boxes.conf.max().cpu())
        else:
            max_conf = 0.0
        confidences[img_name] = max_conf

    print(f"  推理完成: {len(confidences)} 张图片")
    print(f"  置信度范围: [{min(confidences.values()):.4f}, {max(confidences.values()):.4f}]")
    print(f"  置信度均值: {np.mean(list(confidences.values())):.4f}")
    return confidences


def fit_weibull(scores: np.ndarray) -> tuple:
    """
    用 Weibull 分布拟合置信度分数序列

    数学依据:
        由 Fisher-Tippett-Gnedenko 定理，有界随机变量的极值分布为 Weibull 型。
        YOLO 置信度 ∈ [0,1]，因此最大置信度适合 Weibull 拟合。

    Args:
        scores: (N,) 最大置信度数组

    Returns:
        (shape_k, loc, scale_lambda): Weibull 参数
    """
    # 过滤零值（无检测的图片不参与拟合）
    valid_scores = scores[scores > 0]
    if len(valid_scores) < 10:
        raise ValueError(f"有效样本太少: {len(valid_scores)}")

    # scipy.stats.weibull_min.fit 返回 (c, loc, scale)
    # c = shape parameter k
    # scale = lambda
    shape_k, loc, scale_lambda = weibull_min.fit(valid_scores, floc=0)

    print(f"\n=== Weibull 分布拟合结果 ===")
    print(f"  形状参数 k = {shape_k:.4f}")
    print(f"  位置参数 loc = {loc:.6f}")
    print(f"  尺度参数 lambda = {scale_lambda:.4f}")
    print(f"  有效样本数: {len(valid_scores)}")

    return shape_k, loc, scale_lambda


def calculate_defect_probability(score: float, k: float, loc: float, lam: float) -> float:
    """
    计算单个样本的缺陷概率

    公式: P_defect(x) = 1 - exp(-(s_x / lambda)^k)
          即 Weibull CDF

    Args:
        score: 最大检测置信度 s_x
        k: Weibull 形状参数
        loc: 位置参数
        lam: Weibull 尺度参数

    Returns:
        P_defect ∈ [0, 1]
    """
    return float(weibull_min.cdf(score, k, loc=loc, scale=lam))


def batch_defect_probabilities(scores: np.ndarray, k: float, loc: float, lam: float) -> np.ndarray:
    """批量计算缺陷概率"""
    return weibull_min.cdf(scores, k, loc=loc, scale=lam)


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    通过网格搜索找到使 F1 分数最大化的最优阈值 tau*

    tau* = argmax_tau F1(tau)

    Args:
        probs: (N,) 缺陷概率数组
        labels: (N,) 真实标签 (1=有缺陷, 0=无缺陷)

    Returns:
        最优阈值 tau*
    """
    best_f1 = 0.0
    best_tau = 0.5
    thresholds = np.arange(0.01, 1.0, 0.01)

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    print(f"\n=== 最优阈值搜索 ===")
    print(f"  tau* = {best_tau:.2f}")
    print(f"  对应 F1 = {best_f1:.4f}")
    return best_tau


def classify_test_set(
    model, test_dir: str, k: float, loc: float, lam: float, threshold: float
) -> dict:
    """
    对测试集进行批量分类

    Args:
        model: YOLO 模型
        test_dir: 测试图片目录
        k, loc, lam: Weibull 参数
        threshold: 分类阈值 tau*

    Returns:
        dict: {image_name: {"max_conf": float, "p_defect": float, "label": int}}
    """
    confidences = extract_max_confidences(model, test_dir)

    results = {}
    defect_count = 0
    normal_count = 0

    for img_name, max_conf in confidences.items():
        p_defect = calculate_defect_probability(max_conf, k, loc, lam)
        label = 1 if p_defect >= threshold else 0
        results[img_name] = {
            "max_conf": max_conf,
            "p_defect": p_defect,
            "label": label,
        }
        if label == 1:
            defect_count += 1
        else:
            normal_count += 1

    print(f"\n=== 测试集分类结果 ===")
    print(f"  总计: {len(results)} 张")
    print(f"  有缺陷: {defect_count} 张 ({defect_count/len(results)*100:.1f}%)")
    print(f"  无缺陷: {normal_count} 张 ({normal_count/len(results)*100:.1f}%)")

    return results


def plot_weibull_fit(scores: np.ndarray, k: float, loc: float, lam: float, save_path: str):
    """绘制 Weibull 分布拟合效果图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid_scores = scores[scores > 0]

    # 左图: 直方图 + 拟合 PDF
    x = np.linspace(0.001, 1.0, 200)
    pdf = weibull_min.pdf(x, k, loc=loc, scale=lam)

    axes[0].hist(valid_scores, bins=50, density=True, alpha=0.7, color='steelblue', label='实测分布')
    axes[0].plot(x, pdf, 'r-', linewidth=2, label=f'Weibull 拟合 (k={k:.2f}, $\\lambda$={lam:.2f})')
    axes[0].set_xlabel('最大检测置信度 $s$')
    axes[0].set_ylabel('概率密度')
    axes[0].set_title('Weibull 分布拟合效果')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: CDF（即 P_defect 映射）
    cdf = weibull_min.cdf(x, k, loc=loc, scale=lam)
    axes[1].plot(x, cdf, 'b-', linewidth=2, label='$P_{defect}(s)$ = Weibull CDF')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='$\\tau$ = 0.5')
    axes[1].set_xlabel('最大检测置信度 $s$')
    axes[1].set_ylabel('缺陷概率 $P_{defect}$')
    axes[1].set_title('置信度 → 缺陷概率映射')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Weibull 拟合图已保存: {save_path}")


def plot_roc_curve(probs: np.ndarray, labels: np.ndarray, save_path: str):
    """绘制 ROC 曲线"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('EVT 分类器 ROC 曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="EVT 极值理论缺陷分类器")
    parser.add_argument("--weights", type=str, required=True, help="YOLO 模型权重路径")
    parser.add_argument("--val-dir", type=str, default="data/processed/images/val",
                        help="验证集图片目录")
    parser.add_argument("--test-dir", type=str, default="数据集3713/images/test",
                        help="测试集图片目录")
    parser.add_argument("--save-dir", type=str, default="results/evt",
                        help="结果保存目录")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="推理最低置信度阈值")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("基于极值理论 (EVT) 的集装箱缺陷分类器")
    print("=" * 60)

    # 1. 加载模型
    weights_path = PROJECT_ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    print(f"\n加载模型: {weights_path}")
    model = YOLO(str(weights_path))

    # 2. 对验证集提取最大置信度
    val_dir = PROJECT_ROOT / args.val_dir
    print(f"\n--- Step 1: 提取验证集最大置信度 ---")
    val_confidences = extract_max_confidences(model, str(val_dir), conf_threshold=args.conf)
    val_scores = np.array(list(val_confidences.values()))

    # 验证集全部为有缺陷样本，标签全为 1
    val_labels = np.ones(len(val_scores))

    # 3. 拟合 Weibull 分布
    print(f"\n--- Step 2: Weibull 分布拟合 ---")
    k, loc, lam = fit_weibull(val_scores)

    # 4. 绘制拟合效果图
    plot_weibull_fit(val_scores, k, loc, lam, str(save_dir / "weibull_fit.png"))

    # 5. 计算验证集的缺陷概率
    val_probs = batch_defect_probabilities(val_scores, k, loc, lam)
    print(f"\n验证集缺陷概率统计:")
    print(f"  均值: {val_probs.mean():.4f}")
    print(f"  最小值: {val_probs.min():.4f}")
    print(f"  <0.5 的样本数: {(val_probs < 0.5).sum()} / {len(val_probs)}")

    # 6. 确定最优阈值
    # 注意：验证集全为正样本，我们使用启发式阈值
    # 取验证集缺陷概率的第5百分位数作为阈值（允许5%的假阴性）
    threshold = max(np.percentile(val_probs, 5), 0.1)
    print(f"\n最优阈值 (验证集第5百分位): tau* = {threshold:.4f}")

    # 7. 对测试集分类
    print(f"\n--- Step 3: 测试集分类 ---")
    test_dir = PROJECT_ROOT / args.test_dir
    test_results = classify_test_set(model, str(test_dir), k, loc, lam, threshold)

    # 8. 保存结果
    results_summary = {
        "weibull_params": {"k": float(k), "loc": float(loc), "lambda": float(lam)},
        "threshold": float(threshold),
        "val_stats": {
            "num_samples": len(val_scores),
            "mean_conf": float(val_scores.mean()),
            "mean_p_defect": float(val_probs.mean()),
        },
        "test_classification": {
            img: {"label": r["label"], "p_defect": round(r["p_defect"], 4)}
            for img, r in test_results.items()
        },
    }

    summary_path = save_dir / "evt_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {summary_path}")

    # 9. 生成问题1的分类 CSV
    import csv
    cls_csv_path = save_dir / "problem1_classification.csv"
    with open(cls_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "has_defect", "p_defect", "max_confidence"])
        for img_name, r in test_results.items():
            writer.writerow([img_name, r["label"], f"{r['p_defect']:.4f}", f"{r['max_conf']:.4f}"])
    print(f"问题1分类结果: {cls_csv_path}")

    print("\n" + "=" * 60)
    print("EVT 分类器执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
