import cv2
import time
from ultralytics import YOLO

# =========================
# 1. 加载模型
# =========================
# 修改为你的权重路径
model = YOLO(r"best.pt")

# =========================
# 2. 打开视频流
# =========================
# 摄像头：
# cap = cv2.VideoCapture(0)

# 如果是视频文件：
cap = cv2.VideoCapture("1.mp4")

# 设置分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# 3. FPS计算
# =========================
prev_time = 0

# =========================
# 4. 实时检测循环
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("无法读取视频流")
        break

    # =========================
    # YOLO推理
    # =========================
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.5,
        iou=0.45,
        verbose=False
    )

    # 绘制检测结果
    annotated_frame = results[0].plot()

    # =========================
    # FPS计算
    # =========================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # 显示FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =========================
    # 显示窗口
    # =========================
    cv2.imshow("YOLO26 Real-time Detection", annotated_frame)

    # 按 q 退出
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# =========================
# 5. 释放资源
# =========================
cap.release()
cv2.destroyAllWindows()