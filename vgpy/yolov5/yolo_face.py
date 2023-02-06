import argparse
from pathlib import Path
import os

import torch
import torch.backends.cudnn as cudnn

from numpy import random
import numpy as np


from .models.experimental import attempt_load
from .utils.datasets import LoadStreams
from .utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from .utils.torch_utils import select_device, time_synchronized

from ..utils.img_utils import get_cut_img_from_im0 as cut
from ..yolov5.utils.datasets import letterbox

RESIZE_SHAPE = (64,64)

class Yolo_Rectangle:
    def __init__(self, xyxy, conf, label, resize_img, img):
        self.xyxy = xyxy
        self.yolo_conf = conf
        self.name = label
        self.cut_img_not_resize = img
        self.cut_img_resize = resize_img
        self.has = None
        self.conf = None # conf_thre 預測結果 信心超過多少 才標註為 有穿戴
    


from ..utils.img_utils import preprocess_frame_to_yolo_one_cam
class FaceYoloDetector():
    def __init__(self, weights='face_fastface4l.pt', conf_thres=0.6):
        parser = argparse.ArgumentParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, 'weights', weights)
        parser.add_argument('--weights', nargs='+', type=str, default=weights_path, help='model.pt path(s)')

        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

        parser.add_argument('--conf-thres', type=float, default=conf_thres, help='object confidence threshold')

        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        
        opt = parser.parse_args()

        weights, imgsz = opt.weights, opt.img_size
        

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

        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        self.opt = opt
        self.imgsz = imgsz
        self.stride = stride
        self.half = half
        self.device = device
        self.model = model
        self.names = names


    def detect_frame(self, img):
        s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in img], 0)  # shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

        if len(img.shape) == 3:
            img = [img]

        img, im0s = preprocess_frame_to_yolo_one_cam(img, img_size=self.imgsz, stride=self.stride, auto=rect)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=None, agnostic=self.opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                return [Yolo_Rectangle(xyxy, conf, self.names[int(cls)], cut(im0, xyxy, RESIZE_SHAPE), cut(im0, xyxy)) \
                        for *xyxy, conf, cls in reversed(det)], im0
            else:
                return [], im0