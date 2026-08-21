import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import os

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(os.environ.get("TEMP", "/tmp"), "yolo_cfg_train"),
)
from ultralytics import YOLO
import pandas as pd
import torch
import torch.nn as nn

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    """YOLO 训练配置参数数据类"""
    model_name: str = "yolov8n.pt"  # 预训练模型权重（仅用于加载，不传给 train()）
    data: str = str(PROJECT_ROOT / "data" / "processed" / "data.yaml")  # 数据配置文件
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = "0"
    patience: int = 20
    lr0: float = 0.01
    lrf: float = 0.01
    cos_lr: bool = True
    seed: int = 42
    augment: bool = True
    mosaic: float = 1.0
    copy_paste: float = 0.0
    name: str = "baseline"
    workers: int = 0
    exist_ok: bool = True
    project: str = str(PROJECT_ROOT / "runs")
    loss: str = "none"            # none | wd_focal | rusty_w
    resume: bool = False
    copy_paste_set: bool = False  # 是否显式指定了 copy_paste（不传给 train()）
    pretrained: str = "yolov8s.pt"  # yaml 模型（如 P2）挂载的预训练权重

    # 不传给 model.train() 的字段
    _exclude_keys = {"model_name", "loss", "resume", "copy_paste_set"}

    def to_train_args(self) -> Dict[str, Any]:
        """将配置转换为 YOLO model.train() 接受的参数字典"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k not in self._exclude_keys}

def train_baseline(config: TrainConfig) -> None:
    """
    基线模型训练 (YOLOv8n)
    """
    print(f"--- 开始基线模型训练: {config.name} ---")
    model = load_model(config)

    setup_custom_loss(model, config)
    
    # 获取训练参数（排除 model_name 等非训练参数）
    train_args = config.to_train_args()
    if config.resume:
        train_args["resume"] = True
    print(f"训练参数: {train_args}")

    # 开始训练
    results = model.train(**train_args)

    # 打印 mAP50-95
    val_map = results.box.map
    print(f"--- 训练完成，基线模型 {config.name} 的最终 mAP50-95: {val_map:.4f} ---")


def train_improved(config: TrainConfig, custom_epochs: Optional[int] = None) -> None:
    """
    改进模型训练 (支持 YOLOv8s/m, 启用更多数据增强)
    """
    if config.name == "baseline":
        config.name = "improved"
    config.mosaic = 1.0
    if not config.copy_paste_set:
        config.copy_paste = 0.5
    if custom_epochs:
        config.epochs = custom_epochs
    else:
        config.epochs = 200 # 改进模型默认200轮
        
    print(f"--- 开始改进模型训练: {config.name} ({config.model_name}) ---")
    
    model = load_model(config)

    setup_custom_loss(model, config)
    
    train_args = config.to_train_args()
    if config.resume:
        train_args["resume"] = True
    print(f"训练参数: {train_args}")
    
    results = model.train(**train_args)
    
    val_map = results.box.map
    print(f"--- 训练完成，改进模型 {config.name} 的最终 mAP50-95: {val_map:.4f} ---")


def load_model(config: TrainConfig) -> YOLO:
    """加载模型：yaml 配置（如 yolov8s-p2.yaml）加载后挂载 COCO 预训练权重，pt 直接加载"""
    model_path = config.model_name
    if model_path.endswith(".yaml"):
        model = YOLO(model_path)
        if config.pretrained:
            pretrained = config.pretrained
            if not Path(pretrained).is_absolute():
                pretrained = str(PROJECT_ROOT / pretrained)
            if Path(pretrained).exists():
                model.load(pretrained)
                print(f"[train] yaml 模型已挂载预训练权重: {pretrained}")
            else:
                print(f"[train] 警告：预训练权重不存在，按随机初始化训练: {pretrained}")
        return model
    return YOLO(model_path)


def setup_custom_loss(model: YOLO, config: TrainConfig) -> None:
    """按 config.loss 注入自定义分类损失（用 on_train_start 回调，确保 criterion 已就绪）"""
    if config.loss == "wd_focal":
        from wd_focal_loss import WDFocalLoss, WDFocalTrainCallback
        # tau=0 表示自适应：每次更新取各类 W1 距离的中位数（与附录推导一致）
        wdfl = WDFocalLoss(num_classes=3, gamma=2.0, tau=0.0, base_alpha=0.25)
        cb = WDFocalTrainCallback(
            yolo_model=model,
            wdfl=wdfl,
            update_interval=10,
        )
        model.add_callback("on_train_start", cb.on_train_start)
        model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
        print("[train] WD-Focal 损失与回调已注册")
    elif config.loss == "rusty_w":
        # Rusty 类加权：以 Dent 为基准的逆频率 pos_weight（3934/3158~1.246），仅改分类损失
        pos_weight = torch.tensor([1.0, 1.0, 3934.0 / 3158.0])

        def _inject_rusty(trainer):
            from ultralytics.utils.torch_utils import unwrap_model

            model = unwrap_model(trainer.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None:
                criterion = model.init_criterion()
                model.criterion = criterion
            if hasattr(criterion, "bce"):
                criterion._original_bce = criterion.bce
                pw = pos_weight.to(device=next(model.parameters()).device, dtype=torch.float32)
                criterion.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
                print(f"[train] Rusty 类加权已注入（pos_weight={pw.tolist()}）")

        model.add_callback("on_train_start", _inject_rusty)
        print("[train] Rusty 类加权回调已注册")

def predict_and_export(weights_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45) -> None:
    """
    加载最佳权重对 test 集进行推理，生成 CSV 结果和可视化图片
    """
    print(f"--- 开始测试集推理 ---")
    model = YOLO(weights_path)
    
    # 测试集路径（原始数据集中的 test 目录）
    test_dir = PROJECT_ROOT / "数据集3713" / "images" / "test"
    if not test_dir.exists():
        print(f"错误: 测试集目录不存在 {test_dir}")
        return
        
    visualize_dir = PROJECT_ROOT / "runs" / "visualize"
    visualize_dir.mkdir(parents=True, exist_ok=True)
    
    # 进行推理
    results = model.predict(
        source=str(test_dir),
        conf=conf_thres,
        iou=iou_thres,
        save=True,  # 保存可视化图片
        project=str(PROJECT_ROOT / "runs"),
        name="visualize/predict",
        exist_ok=True # 允许覆盖
    )
    
    # 提取结果并保存为 test_result.csv
    out_data = []
    for result in results:
        img_name = Path(result.path).stem  # image_id 不含扩展名（如 1.jpg -> 1）
        boxes = result.boxes
        for box in boxes:
            # 获取类别和坐标 (归一化后的中心点 xywh)
            cls_id = int(box.cls[0].item())
            xywhn = box.xywhn[0].cpu().numpy() # x_center, y_center, width, height (归一化)
            out_data.append([img_name, cls_id, xywhn[0], xywhn[1], xywhn[2], xywhn[3]])
            
    df = pd.DataFrame(out_data, columns=["image_id", "class_id", "x_center", "y_center", "width", "height"])
    out_csv = PROJECT_ROOT / "test_result.csv"
    df.to_csv(out_csv, index=False)
    print(f"--- 推理完成 ---")
    print(f"结果已保存到: {out_csv}")
    print(f"可视化结果位于: {PROJECT_ROOT / 'runs' / 'visualize' / 'predict'}")

def main():
    parser = argparse.ArgumentParser(description="YOLO 模型训练与推理脚本")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "improved", "predict"], 
                        help="运行模式: baseline, improved 或 predict")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                        help="预训练模型权重路径或名称")
    parser.add_argument("--weights", type=str, default="", 
                        help="用于推理的模型权重路径 (例如 runs/baseline/weights/best.pt)")
    parser.add_argument("--epochs", type=int, default=0, 
                        help="覆盖默认的训练轮数")
    parser.add_argument("--loss", type=str, default="none", choices=["none", "wd_focal", "rusty_w"],
                        help="分类损失方案: none(默认) / wd_focal / rusty_w")
    parser.add_argument("--copy-paste", type=float, default=None,
                        help="Copy-Paste 增强比例（覆盖 improved 默认 0.5）")
    parser.add_argument("--imgsz", type=int, default=0, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=0, help="批次大小")
    parser.add_argument("--data", type=str, default="", help="数据配置文件路径")
    parser.add_argument("--resume", action="store_true", help="从 runs/{project}/{name}/weights/last.pt 断点续训")
    parser.add_argument("--pretrained", type=str, default="", help="yaml 模型（如 P2）挂载的预训练权重名")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="推理置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, 
                        help="推理 NMS IoU 阈值")
    parser.add_argument("--name", type=str, default="", 
                        help="实验名称 (例如 improved_with_neg)")
    
    args = parser.parse_args()
    
    if args.mode in ("baseline", "improved"):
        # 若传入的是裸权重名且项目根下存在本地文件，则优先用本地权重（避免联网下载）
        if not Path(args.model).is_absolute() and not Path(args.model).exists():
            local = PROJECT_ROOT / args.model
            if local.exists():
                args.model = str(local)
        config = TrainConfig(model_name=args.model)
        if args.name:
            config.name = args.name
        if args.epochs > 0:
            config.epochs = args.epochs
        if args.imgsz > 0:
            config.imgsz = args.imgsz
        if args.batch > 0:
            config.batch = args.batch
        if args.data:
            config.data = str(PROJECT_ROOT / args.data) if not Path(args.data).is_absolute() else args.data
        if args.copy_paste is not None:
            config.copy_paste = args.copy_paste
            config.copy_paste_set = True
        config.loss = args.loss
        config.resume = args.resume
        if args.pretrained:
            config.pretrained = args.pretrained
        if config.resume:
            # 断点续训必须从 runs/{project}/{name}/weights/last.pt 加载，而非裸权重名
            resume_pt = Path(config.project) / config.name / "weights/last.pt"
            if resume_pt.exists():
                config.model_name = str(resume_pt)
                print(f"[train] 断点续训加载: {resume_pt}")
            else:
                print(f"[train] 警告：未找到续训检查点 {resume_pt}，将从头训练")
        if args.mode == "baseline":
            train_baseline(config)
        else:
            train_improved(config, custom_epochs=args.epochs if args.epochs > 0 else None)
        
    elif args.mode == "predict":
        if not args.weights:
            print("错误: 预测模式需要提供 --weights 参数")
            return
        # 使用相对于项目根目录的路径或绝对路径
        weights_path = str(PROJECT_ROOT / args.weights) if not Path(args.weights).is_absolute() else args.weights
        predict_and_export(weights_path, conf_thres=args.conf, iou_thres=args.iou)

if __name__ == "__main__":
    main()
