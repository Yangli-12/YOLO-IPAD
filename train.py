from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r'best.pt').cuda()  # 此处以 m 为例，只需写yolov11m即可定位到m模型
    model = YOLO(r'C:/Users/54548/Desktop/4/code/ultralytics-main/yolo26-our.yaml').cuda()  # 此处以 m 为例，只需写yolov11m即可定位到m模型

    model.train(data='coco128.yaml',
                resume=False,
                imgsz=640,
                epochs=100,
                single_cls=False,
                batch=16,
                workers=8,
                device='0',
                )

