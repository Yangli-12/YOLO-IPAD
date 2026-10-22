

import os
from ultralytics import YOLO
import numpy as np

# ===== 路径 =====
image_dir = r"C:/Users/54548/Desktop/4/NBGDQ/jacket/images"
label_dir = r"C:/Users/54548/Desktop/4/NBGDQ/jacket/labels"
output_dir = r"merged_labels"

os.makedirs(output_dir, exist_ok=True)

# ===== 模型（6类）=====
model = YOLO("best-6.pt")

# ===== 🔥 原标签映射 =====
OLD_TO_NEW = {
    0: 3,  # reflective_cloth
    1: 4   # life_jacket
}

# ===== 只补这些类别 =====
ALLOWED_CLASSES = [0, 1, 2, 5]

CONF_THRES = 0.5
IOU_THRES = 0.5

img_exts = [".jpg", ".jpeg", ".png"]


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1, ya1 = x1 - w1/2, y1 - h1/2
    xa2, ya2 = x1 + w1/2, y1 + h1/2

    xb1, yb1 = x2 - w2/2, y2 - h2/2
    xb2, yb2 = x2 + w2/2, y2 + h2/2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = w1 * h1
    area2 = w2 * h2

    return inter_area / (area1 + area2 - inter_area + 1e-6)


for file in os.listdir(image_dir):
    name, ext = os.path.splitext(file)

    if ext.lower() not in img_exts:
        continue

    img_path = os.path.join(image_dir, file)
    txt_path = os.path.join(label_dir, name + ".txt")

    merged_labels = []
    existing_boxes = []
    existing_classes = set()

    # ===== 🔥 1. 读取并映射原标签 =====
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])

            # 🔥 映射到统一类别
            cls = OLD_TO_NEW.get(cls, cls)

            box = list(map(float, parts[1:5]))

            new_line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"

            merged_labels.append(new_line)
            existing_boxes.append(box)
            existing_classes.add(cls)

    # ===== 2. 模型预测 =====
    results = model.predict(img_path, conf=CONF_THRES, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xywhn.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for cls, box in zip(classes, boxes):
            cls = int(cls)

            # ===== 只补指定类别 =====
            if cls not in ALLOWED_CLASSES:
                continue

            # ===== 已有类别不重复补 =====
            if cls in existing_classes:
                continue

            # ===== IoU去重 =====
            keep = True
            for e_box in existing_boxes:
                if compute_iou(box, e_box) > IOU_THRES:
                    keep = False
                    break

            if not keep:
                continue

            x, y, w, h = box
            line = f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"

            merged_labels.append(line)
            existing_boxes.append(box)

    # ===== 3. 写入 =====
    out_path = os.path.join(output_dir, name + ".txt")
    with open(out_path, "w") as f:
        f.writelines(merged_labels)

print("✅ 完成：保留原标签 + 自动补全其他类别")

