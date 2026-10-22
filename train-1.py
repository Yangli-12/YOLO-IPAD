import os

import cv2

from ultralytics import YOLO

# =========================
# 配置部分
# =========================
VIDEO_ROOT = r"D:\BaiduNetdiskDownload\video1"  # 输入视频根目录
SAVE_ROOT = r"E:\4-data"  # 输出目录
MODEL_PATH = r"best-all3.pt"  # 权重

IMG_SIZE = 640
CONF_THRES = 0.25

FRAME_INTERVAL = 5  # ⭐ 每5帧取1帧（防止数据爆炸）
MIN_BOX_SIZE = 10  # ⭐ 最小目标尺寸过滤

# 是否只保留某些类别（None = 全部）
TARGET_CLASSES = None  # e.g. [0] 只保留person

# =========================
# 初始化模型
# =========================
model = YOLO(MODEL_PATH)
model = model.cuda()

# =========================
# 创建统一输出目录
# =========================
save_img_dir = os.path.join(SAVE_ROOT, "images")
save_txt_dir = os.path.join(SAVE_ROOT, "labels")

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_txt_dir, exist_ok=True)


# =========================
# 获取所有视频路径（递归）
# =========================
def get_all_videos(root):
    video_list = []
    for root_dir, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".ts")):
                video_list.append(os.path.join(root_dir, f))
    return video_list


# =========================
# 保存 YOLO 标签
# =========================
def save_yolo_txt(txt_path, boxes, img_w, img_h):
    with open(txt_path, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            xc = (x1 + x2) / 2 / img_w
            yc = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# =========================
# 全局ID（防止重名）
# =========================
global_id = 0


# =========================
# 处理视频
# =========================
def process_video(video_path):
    global global_id

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frame_id = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ⭐ 帧采样
        if frame_id % FRAME_INTERVAL != 0:
            frame_id += 1
            continue

        h, w = frame.shape[:2]

        # 推理
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]

        boxes_to_save = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 类别过滤
                if TARGET_CLASSES is not None and cls not in TARGET_CLASSES:
                    continue

                # ⭐ 置信度过滤（提升伪标签质量）
                if conf < CONF_THRES:
                    continue

                # ⭐ 小目标过滤
                if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                    continue

                boxes_to_save.append([cls, x1, y1, x2, y2])

        # ⭐ 只保存有目标的帧
        if len(boxes_to_save) > 0:
            img_name = f"{video_name}_{global_id:08d}.jpg"
            txt_name = f"{video_name}_{global_id:08d}.txt"

            img_path = os.path.join(save_img_dir, img_name)
            txt_path = os.path.join(save_txt_dir, txt_name)

            cv2.imwrite(img_path, frame)
            save_yolo_txt(txt_path, boxes_to_save, w, h)

            global_id += 1
            save_count += 1

        frame_id += 1

    cap.release()
    print(f"[✓] 完成: {video_name} | 保存 {save_count} 帧")


# =========================
# 主函数
# =========================
if __name__ == "__main__":
    video_list = get_all_videos(VIDEO_ROOT)
    print(f"共找到 {len(video_list)} 个视频")

    for v in video_list:
        process_video(v)

    print("全部处理完成 ✅")
