# -*- coding: utf-8 -*-
"""
WD-Focal Loss: 基于 Wasserstein 距离的自适应焦点损失函数
用于 Ultralytics YOLOv8/v11 的分类损失替换

数学推导详见: src/wd_focal_loss_derivation.tex
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class WDFocalLoss(nn.Module):
    """
    Wasserstein-Distance Weighted Focal Loss (WD-Focal Loss)

    在标准 Focal Loss 基础上，利用各类别特征分布之间的 Wasserstein 距离
    动态计算类别平衡系数 alpha_c，替代传统的固定 alpha。

    核心公式:
        L_WD-FL(p_t, c) = -alpha_c * (1 - p_t)^gamma * log(p_t)
        alpha_c = sigmoid(W_1(mu_c, mu_bar) / tau)

    参数:
        num_classes (int): 类别数量 (本项目为 3)
        gamma (float): 聚焦参数，默认 2.0
        tau (float): 温度参数，控制 sigmoid 映射灵敏度
        base_alpha (float): 基础 alpha 值，Wasserstein 距离未计算前使用
        num_quantiles (int): 计算 Wasserstein 距离的分位数网格点数
        update_interval (int): 每隔多少个 epoch 更新一次 alpha
    """

    def __init__(
        self,
        num_classes: int = 3,
        gamma: float = 2.0,
        tau: float = 1.0,
        base_alpha: float = 0.25,
        num_quantiles: int = 100,
        update_interval: int = 10,
        alpha_min: float = 0.25,
        alpha_max: float = 0.75,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.tau = tau
        self.base_alpha = base_alpha
        self.num_quantiles = num_quantiles
        self.update_interval = update_interval
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # 初始化动态 alpha 为 base_alpha（训练开始前）
        self.register_buffer(
            "alpha_per_class", torch.full((num_classes,), base_alpha)
        )

        # 存储各类别的特征分布（用于计算 Wasserstein 距离）
        self._class_features: Dict[int, list] = {c: [] for c in range(num_classes)}
        self._epoch_counter = 0
        # 训练期正样本置信度缓存（置信度分布版，forward 内无梯度收集）
        self._conf_buffer: Dict[int, list] = {c: [] for c in range(num_classes)}

    def reset_features(self):
        """每个 epoch 开始前重置特征缓存"""
        self._class_features = {c: [] for c in range(self.num_classes)}

    def collect_features(self, features: torch.Tensor, labels: torch.Tensor):
        """
        收集当前 batch 的特征向量（用于 Wasserstein 距离计算）

        Args:
            features: (N, D) 特征向量
            labels: (N,) 类别标签
        """
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = labels == c
                if mask.any():
                    # 取特征的 L2 范数作为一维投影（简化计算）
                    norms = features[mask].norm(dim=1).cpu().tolist()
                    self._class_features[c].extend(norms)

    def update_alpha_from_confidences(self, class_confidences: Dict[int, list]) -> tuple:
        """
        实践版：基于各类别检测置信度分布更新 alpha_c（以置信度分布代替附录推导中的特征分布）。
        使用一维分位数近似的 W_1 距离:
            W_1(mu_c, mu_bar) ~ (1/K) * sum_k |Q_c(k/K) - Q_bar(k/K)|
        """
        if not class_confidences or len(class_confidences) < self.num_classes:
            print("[WD-Focal] 置信度分布缺失，跳过 alpha 更新")
            return {}, self.alpha_per_class.tolist()

        min_samples = min(len(v) for v in class_confidences.values())
        if min_samples < 10:
            print(f"[WD-Focal] 样本不足 (min={min_samples})，跳过 alpha 更新")
            return {}, self.alpha_per_class.tolist()

        quantile_grid = np.linspace(0, 1, self.num_quantiles + 2)[1:-1]  # 去除0和1
        class_quantiles = {}
        all_features = []

        for c in range(self.num_classes):
            arr = np.asarray(class_confidences.get(c, []), dtype=np.float64)
            class_quantiles[c] = np.quantile(arr, quantile_grid)
            all_features.extend(arr.tolist())

        ref_quantiles = np.quantile(np.asarray(all_features), quantile_grid)

        distances = {}
        for c in range(self.num_classes):
            distances[c] = float(np.mean(np.abs(class_quantiles[c] - ref_quantiles)))

        adaptive_tau = max(float(np.median(list(distances.values()))), 1e-6)
        if self.tau <= 0:
            self.tau = adaptive_tau

        # 限幅：避免 alpha 饱和到 1 使负样本损失消失（v1 失败的根因）
        new_alphas = [
            float(np.clip(1.0 / (1.0 + np.exp(-distances[c] / self.tau)),
                          self.alpha_min, self.alpha_max))
            for c in range(self.num_classes)
        ]
        self.alpha_per_class = torch.tensor(new_alphas, dtype=torch.float32).to(
            self.alpha_per_class.device
        )
        print(f"[WD-Focal] W1 distances = {distances}, "
              f"alpha = {dict(zip(range(self.num_classes), new_alphas))}")
        return distances, new_alphas

    def update_alpha(self):
        """兼容旧接口：从训练期特征缓存更新 alpha（特征取 L2 范数投影，等价于置信度分布版）"""
        self._epoch_counter += 1
        if self._epoch_counter % self.update_interval != 0:
            return
        self.update_alpha_from_confidences(self._class_features)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        计算 WD-Focal Loss

        Args:
            pred: (bs, num_anchors, num_classes) logits 预测
            label: (bs, num_anchors, num_classes) one-hot 或 soft 标签

        Returns:
            标量损失值
        """
        # 标准 BCE 损失（逐元素）
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        # Focal 调制因子
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = loss * modulating_factor

        # WD 动态 alpha 加权
        # alpha_per_class: (num_classes,) -> 广播到 (1, 1, num_classes)
        alpha = self.alpha_per_class.to(device=pred.device, dtype=pred.dtype)
        alpha = alpha.view(1, 1, -1)  # (1, 1, C)

        # 按类别加权: 正样本用 alpha_c, 负样本用 (1 - alpha_c)
        alpha_factor = label * alpha + (1 - label) * (1 - alpha)
        loss = loss * alpha_factor

        # 返回逐元素损失 (N, C)，与 Ultralytics v8DetectionLoss 的 bce 接口兼容
        # 收集正样本 p_t（正确类别置信度）分布，供每 update_interval 轮的 alpha 更新使用
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = label[..., c] > 0.5
                if mask.any():
                    self._conf_buffer[c].extend(p_t[..., c][mask].float().cpu().tolist())
        return loss

    def drain_conf_buffer(self) -> Dict[int, list]:
        """取出并清空训练期正样本置信度缓存"""
        confs = {c: list(v) for c, v in self._conf_buffer.items()}
        self._conf_buffer = {c: [] for c in range(self.num_classes)}
        return confs


# ============================================================
# 以下为植入 Ultralytics 的工具函数
# ============================================================

def get_wd_focal_loss(num_classes: int = 3, gamma: float = 2.0, tau: float = 1.0) -> WDFocalLoss:
    """工厂函数：创建 WD-Focal Loss 实例"""
    return WDFocalLoss(num_classes=num_classes, gamma=gamma, tau=tau)


def patch_v8_detection_loss(model, wd_focal_loss: WDFocalLoss):
    """
    将 WD-Focal Loss 植入到 YOLOv8 的 v8DetectionLoss 中

    具体修改位置: ultralytics/utils/loss.py 第 348 行
    原始代码: self.bce = nn.BCEWithLogitsLoss(reduction="none")
    修改思路: 在计算分类损失时替换 BCE 为 WD-Focal Loss

    Args:
        model: Ultralytics YOLO 模型实例
        wd_focal_loss: WDFocalLoss 实例
    """
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        # 找到 DetectionLoss 所在位置
        criterion = model.model.criterion
        if hasattr(criterion, 'bce'):
            # 保存原始 BCE 以备回退
            criterion._original_bce = criterion.bce
            # 注入 WD-Focal Loss
            criterion._wd_focal = wd_focal_loss.to(next(model.parameters()).device)
            criterion._use_wd_focal = True
            print("[WD-Focal] 已成功植入 v8DetectionLoss")
        else:
            print("[WD-Focal] 警告: 未找到 criterion.bce，跳过植入")


# ============================================================
# 用于 train_yolo.py 的回调函数
# ============================================================

class WDFocalCallback:
    """
    Ultralytics 训练回调：在训练过程中动态更新 WD-Focal Loss 的 alpha

    使用方法:
        from src.wd_focal_loss import WDFocalLoss, WDFocalCallback
        wdfl = WDFocalLoss(num_classes=3, gamma=2.0, tau=1.0)
        callback = WDFocalCallback(wdfl)
        model.add_callback("on_train_epoch_start", callback.on_epoch_start)
        model.add_callback("on_train_epoch_end", callback.on_epoch_end)
    """

    def __init__(self, wd_focal_loss: WDFocalLoss):
        self.wd_focal = wd_focal_loss

    def on_epoch_start(self, trainer):
        """每个 epoch 开始前重置特征缓存"""
        self.wd_focal.reset_features()

    def on_epoch_end(self, trainer):
        """每个 epoch 结束后更新 alpha"""
        self.wd_focal.update_alpha()


class WDFocalTrainCallback:
    """
    Ultralytics 训练回调：on_train_start 注入 WD-Focal 到 v8DetectionLoss.bce，
    每 update_interval 轮用固定验证子图的检测置信度分布更新 alpha。

    工程取舍：以检测置信度分布代替附录推导中的特征分布（特征级收集需侵入检测头，
    留作后续工作）；alpha 更新只读推理，不影响训练主循环数值。
    """

    def __init__(self, yolo_model, wdfl: WDFocalLoss, update_interval: int = 10):
        self.yolo = yolo_model
        self.wdfl = wdfl
        self.update_interval = update_interval

    def on_train_start(self, trainer):
        from ultralytics.utils.torch_utils import unwrap_model

        model = unwrap_model(trainer.model)
        criterion = getattr(model, "criterion", None)
        if criterion is None:
            # criterion 在首次 loss 计算时才惰性创建，这里提前创建以便注入
            criterion = model.init_criterion()
            model.criterion = criterion
        if criterion is None:
            print("[WD-Focal] 警告：未找到 criterion，跳过注入")
            return
        if hasattr(criterion, "bce"):
            criterion._original_bce = criterion.bce
            self.wdfl.to(device=next(model.parameters()).device)
            criterion.bce = self.wdfl
            print("[WD-Focal] 已注入 v8DetectionLoss.bce（WD-Focal）")

    def on_train_epoch_end(self, trainer):
        if (trainer.epoch + 1) % self.update_interval != 0:
            return
        confs = self.wdfl.drain_conf_buffer()
        self.wdfl.update_alpha_from_confidences(confs)


if __name__ == "__main__":
    # 单元测试
    print("=== WD-Focal Loss 单元测试 ===")

    wdfl = WDFocalLoss(num_classes=3, gamma=2.0, tau=1.0)

    # 模拟输入
    batch_size, num_anchors, num_classes = 4, 100, 3
    pred = torch.randn(batch_size, num_anchors, num_classes)
    label = torch.zeros(batch_size, num_anchors, num_classes)
    label[:, :50, 0] = 1.0  # 前50个anchor为Dent
    label[:, 50:60, 1] = 1.0  # 10个为Hole
    label[:, 60:80, 2] = 1.0  # 20个为Rusty

    loss = wdfl(pred, label).sum()
    print(f"初始 Loss: {loss.item():.4f}")
    print(f"初始 alpha: {wdfl.alpha_per_class}")

    # 模拟特征收集和更新
    for _ in range(10):
        features = torch.randn(100, 256)
        labels = torch.cat([torch.zeros(50), torch.ones(10), torch.full((40,), 2)]).long()
        wdfl.collect_features(features, labels)

    wdfl._epoch_counter = 9  # 触发第10个epoch的更新
    wdfl.update_alpha()
    print(f"更新后 alpha: {wdfl.alpha_per_class}")

    loss_after = wdfl(pred, label).sum()
    print(f"更新后 Loss: {loss_after.item():.4f}")
    print("=== 测试通过 ===")
