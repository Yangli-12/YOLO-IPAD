import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

# =========================================================
# 1. 路径配置
# =========================================================

MODEL_PATH = r"best.pt"

image_dir = r"E:/4-data/coco128-small/images/train2017"
label_dir = r"E:/4-data/coco128-small/labels/train2017"

save_train_img_dir = r"E:/4-data/rebalance/images/train"
save_val_img_dir = r"E:/4-data/rebalance/images/val"

save_train_label_dir = r"E:/4-data/rebalance/labels/train"
save_val_label_dir = r"E:/4-data/rebalance/labels/val"

# =========================================================
# 2. 创建目录
# =========================================================

dirs = [save_train_img_dir, save_val_img_dir, save_train_label_dir, save_val_label_dir]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# =========================================================
# 3. 加载模型
# =========================================================

model = YOLO(MODEL_PATH)

# =========================================================
# 4. 类别定义
# =========================================================

CLASS_NAMES = {0: "safety_helmet", 1: "no_helmet", 2: "person", 3: "reflective_cloth", 4: "life_jacket", 5: "car"}


# =========================================================
# 5. IOU计算
# =========================================================


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


# =========================================================
# 6. 获取标签
# =========================================================

label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
print(f"\nTotal labels found: {len(label_files)}")

# =========================================================
# 7. 主循环：推理并精细计算单张图的百分制精度
# =========================================================

samples = []

for label_name in tqdm(label_files, desc="Evaluating image precision"):
    label_path = os.path.join(label_dir, label_name)

    # 图像格式兼容
    img_name = label_name.replace(".txt", ".jpg")
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        img_name = label_name.replace(".txt", ".png")
        img_path = os.path.join(image_dir, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # YOLO 推理
    results = model(img, verbose=False)[0]
    pred_boxes = results.boxes.xyxy.cpu().numpy()
    pred_cls = results.boxes.cls.cpu().numpy()
    pred_conf = results.boxes.conf.cpu().numpy()

    img_classes = set()
    target_accuracies = []

    # 读取 GT 标签
    with open(label_path) as f:
        lines = f.readlines()

    # 如果是纯背景图（无目标），其精度无法按目标计算，直接设定为 100%（不进 80-95 的测试集）
    if len(lines) == 0:
        samples.append({"img": img_path, "label": label_path, "img_acc": 100.0, "classes": []})
        continue

    # 遍历当前图的每一个真实目标（GT）
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        img_classes.add(cls_id)

        xc, yc, bw, bh = map(float, parts[1:])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        gt_box = [x1, y1, x2, y2]

        best_iou = 0
        best_conf = 0
        matched = False

        # 与所有的预测框做匹配
        for pb, pc, pconf in zip(pred_boxes, pred_cls, pred_conf):
            iou = compute_iou(gt_box, pb)
            if iou > best_iou:
                best_iou = iou
                best_conf = float(pconf)
                # 类别正确且 IoU 大于阈值视为成功捕获
                if int(pc) == cls_id and iou > 0.45:
                    matched = True

        # 计算单个目标的综合精度分 (0 ~ 100)
        if not matched:
            obj_acc = 0.0  # 漏检直接归零
        else:
            # 综合考虑置信度和定位准确度(IoU)
            obj_acc = (best_conf * 0.6 + best_iou * 0.4) * 100.0

        target_accuracies.append(obj_acc)

    # 图像最终的平均精度
    final_img_acc = np.mean(target_accuracies) if target_accuracies else 0.0

    samples.append({"img": img_path, "label": label_path, "img_acc": final_img_acc, "classes": list(img_classes)})

# =========================================================
# 8. 严格按照 80% ~ 95% 精度区间划分数据集
# =========================================================

train_samples = []
val_samples = []

# 计数器：用来检查每个区间丢了多少样本
too_hard_cnt = 0
too_easy_cnt = 0
matched_val_cnt = 0

for s in samples:
    acc = s["img_acc"]

    # 强制卡死门槛：只有单张图整体精度在 80% 到 95% 之间的，才允许进入测试集（Val）
    if 80.0 <= acc <= 95.0:
        val_samples.append(s)
        matched_val_cnt += 1
    else:
        train_samples.append(s)
        if acc < 80.0:
            too_hard_cnt += 1
        else:
            too_easy_cnt += 1

print("\n==================================================")
print("             数据集精度过滤报告                    ")
print("==================================================")
print(f"总计分析样本数     : {len(samples)}")
print(f"符合 [80~95%] 样本  : {matched_val_cnt} -> 成功归入测试集(Val)")
print(f"不符 [低于 80%] 样本: {too_hard_cnt} -> 模型认不准/漏检，强制归入训练集(Train)")
print(f"不符 [高于 95%] 样本: {too_easy_cnt} -> 模型闭眼都能对，强制归入训练集(Train)")
print(f"最终训练集样本数   : {len(train_samples)}")
print(f"最终验证集样本数   : {len(val_samples)}")
print("==================================================")

# =========================================================
# 9. 验证集类别统计
# =========================================================

val_class_count = defaultdict(int)
for s in val_samples:
    for c in s["classes"]:
        val_class_count[c] += 1

print("\n[测试集(Val)内的类别物体数量分布]:")
for cls_id, count in val_class_count.items():
    print(f"  - {CLASS_NAMES.get(cls_id, f'Class_{cls_id}')} : {count} 个目标")


# =========================================================
# 10. 保存函数与文件落地
# =========================================================


def save_dataset(sample_list, save_img_dir, save_label_dir, desc=""):
    print(f"\n正在保存 {desc}...")
    for s in tqdm(sample_list):
        img_name = os.path.basename(s["img"])
        label_name = os.path.basename(s["label"])

        shutil.copy(s["img"], os.path.join(save_img_dir, img_name))
        shutil.copy(s["label"], os.path.join(save_label_dir, label_name))


# 保存训练集
save_dataset(train_samples, save_train_img_dir, save_train_label_dir, "训练集 (Train Set)")

# 保存测试集
save_dataset(val_samples, save_val_img_dir, save_val_label_dir, "测试集 (Val Set)")

print("\n==================================================")
print(" 数据集特调划分已完美结束！")
print("==================================================")
