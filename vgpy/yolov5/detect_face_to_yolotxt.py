import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from numpy import random
import numpy as np
import time
import os

from .models.experimental import attempt_load
from .utils.datasets import LoadImages
from .utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, set_logging

from .utils.torch_utils import select_device, load_classifier

from ..utils.img_utils import get_cut_img_from_im0 as cut
from ..utils.img_utils import cv2_read_zh_img, preprocess_frame_to_yolo_one_cam

# python detect_face.py --weights "weights/face_fastface4l.pt" --source "D:/Facial_V1.3/Dataset/*.jpg" --save-txt --name "face_location"
def detect_face2yolotxt(img_dir):
    parser = argparse.ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'weights', 'face_fastface4l.pt')

    parser.add_argument('--weights', nargs='+', type=str, default=weights_path, help='model.pt path(s)')    
    # parser.add_argument('--source', type=str, default=img_dir, help='source')

    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    # print(opt)


    # source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    source, weights, save_txt, imgsz = img_dir, opt.weights, opt.save_txt, opt.img_size
    save_txt = True
    opt.exist_ok = True
    opt.project = img_dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
        ### 經過幾個小時的 debug，如果要在別的資料夾使用yolov5的東西 就要加上以下這段才能順利讀到model
    import sys
    hubconf_dir = './vgpy/yolov5'
    sys.path.insert(0, hubconf_dir)        
        ### 以上這段 拯救蒼生

    model = attempt_load(weights, map_location=device)  # load FP32 model
    del sys.path[0]
    
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # 若source是路徑
    if isinstance(source, str):
        # Directories
        save_dir = Path(opt.project)
        # 清空 labels 資料夾
        import shutil
        shutil.rmtree(save_dir / 'labels')
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        img_files = Path(source).glob('*.jpg')
        for img_file in img_files:
            img_path = os.path.join(source, img_file)
            img = cv2_read_zh_img(img_path)
            img, im0s = preprocess_frame_to_yolo_one_cam(img, img_size=imgsz, stride=stride)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image            
                s, im0 = '%g: ' % i, im0s[i].copy()
                p = Path(img_path)  # to Path

                txt_path = str(save_dir / 'labels' / p.stem) # img.txt

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xyxy, conf) if opt.save_conf else (cls, *xyxy)  # label format
                            with open(txt_path + '.txt', 'w') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')


        if save_txt:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

        return

    # 若source是圖片
    else:
        img, im0s = preprocess_frame_to_yolo_one_cam(source, img_size=imgsz, stride=stride)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for i, det in enumerate(pred):
            if len(det) == 0:
                return False
            else:
                return True

