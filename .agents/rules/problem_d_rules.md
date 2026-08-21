# 选题D：集装箱智能破损检测 — Agent 规则

## 项目概览
- **任务类型**：工业缺陷检测（图像分类 + 目标检测）
- **数据集**：3713 张集装箱图片（训练 3300 + 测试 413），640×640 像素
- **缺陷类别**：Dent(0), Hole(1), Rusty(2)
- **当前状态（2026-08-21）**：建模、论文撰写、赛后补充实验、评审整改、AI 自查与
  提交材料打包全部完成（消融/负样本比例/Rusty 加权/P2/TTA/WD-Focal 端到端，见
  results/实验记录.md）；配图优化（Okabe-Ito 方案A）与「理论/公式 ↔ 参考文献」审查
  修订完成——refs.tex 共 18 条、引用键全部有对应条目；main.pdf 88 页（含 AI 工具
  使用声明与附录 A.12 支撑材料列表），verify_pdf 门禁全绿（??=0、16 图、附录代码
  渲染正常）；Word 版 main.docx、AI工具使用详情.pdf（DeepSeek-V4-Flash，匿名）与
  打包脚本 pack_submission.py 已生成，电子版提交包 20262026104.zip 已校验
  （9.08MB，27 文件，zip 内 CSV 0 个 .jpg）；git HEAD 为 83607a3 且与 origin/main
  同步，本轮打包改动尚未 commit/push（待用户拍板，远端 443 暂不可达）；最终模型
  保持 improved_with_neg（mAP50-95≈0.205、EVT k=3.064），test_result.csv 未按 TTA
  重出（已拍板不采用，2026-08-14，增益 0.011 优化不大）；工程卫生已闭环：git init
  + .gitignore、src 相对路径化、run_evt_probe.py 移至 src/、uv.lock / yolo26n.pt /
  .idea 跟踪配置 / val_scores.npy 已清理（pyproject.toml 已按用户决定删除，依赖
  说明见 README.md / CLAUDE.md）

## Python 环境
- **Python 解释器**：`E:\python2025\python.exe`
- 所有 Python 脚本应使用该解释器运行
- Ultralytics 脚本若报 `settings.json` 权限错误，先设置环境变量 `YOLO_CONFIG_DIR` 指向可写目录

## 输出规范
### test_result.csv 格式
```
image_id,class_id,x_center,y_center,width,height
```
- image_id: 图片文件名（不含扩展名）
- class_id: 0/1/2
- x_center, y_center, width, height: 归一化坐标 (0~1)
- **已统一（2026-08-21）**：按提交要求首列不含 `.jpg` 后缀（用户拍板）；
  src/train_yolo.py 与 论文/code/train_yolo.py 同步改为 `Path(result.path).stem`，
  931 行检测框不变（Dent 463 / Hole 131 / Rusty 337）

## 关键注意事项
1. **类别不平衡**：Hole(1) 占 11.8% 且以中小目标为主；Copy-Paste 已配置（p=0.5）但本数据无分割掩膜、消融证实无实际增益（2026-08-14），小目标 Hole 的有效提升路径为 P2 头（AP50-95 +0.021）与 TTA（+0.011，已拍板不重出 test_result.csv）
2. **负样本**：原训练集无负样本；已从无标注区域裁剪 1200 幅候选（`data/processed/negative_samples/`），其中 600 幅已并入训练集（`data/processed/images/train/` 内 `neg_*`）；600 为分离度甜点（neg300/600/900 的 EVT k 对比见实验记录）
3. **可复现性**：所有实验设置 `seed=42`，保存完整配置
4. **评估指标**：mAP@0.5, mAP@0.5:0.95, 各类别 P/R/F1
5. **最终模型替换门槛**：新模型 mAP50-95 ≥ 0.215 且逐类无回退才替换最终模型并重出 test_result.csv；否则新结果仅作消融对比

## 目录约定
- `src/`：核心代码（权威源；`论文/code/` 为附录副本，仅 ASCII 数学符号转换差异，由 `论文/prep_figures.py` 重新生成）
- `data/processed/`：处理后的数据集
- `runs/`：训练输出与模型权重（最终权重在 `runs/detect/improved_with_neg/weights/best.pt`）
- `results/`：评估报告与可视化（含 `results/evt*` 问题1判别结果、`results/evt_pseudo_neg_results.txt` 伪负样本校准输出、`results/ai_self_check/` AI 自查报告）
- `论文/`：LaTeX 论文工程（`main.tex` 编译入口，`figures/` 图表，`code/` 附录源码副本；编译：`cd 论文; .\build.ps1`；门禁：`python 论文/scripts/verify_pdf.py 论文/main.pdf`；Word 版由 `论文/make_word.py` 生成）
- `pack_submission.py`：电子版材料归集与打包（输出 `20262026104/` 与 `20262026104.zip`）
- `AI工具使用详情.pdf/.tex`：AI 工具使用说明（支撑材料，内容匿名）
- `数据探索/`：EDA 输出图表
- `参考论文/`：参考资料（只读，入库）；`yolo系列论文/`：参考资料（只读，**仅本地保留、不入库**，2026-08-20 用户决定）
