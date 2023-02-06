import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn

from numpy import random
import numpy as np
import cv2

from .models.experimental import attempt_load
from .utils.datasets import LoadStreams
from .utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from .utils.torch_utils import select_device, time_synchronized

from ..utils.img_utils import get_cut_img_from_im0 as cut
from ..yolov5.utils.datasets import letterbox
from .utils.calculate import init_fence_point_to_int

RESIZE_SHAPE = (64,64)

class Yolo_Rectangle:
    def __init__(self, xyxy, conf, label, im0):
        self.xyxy = xyxy # 後續要用來給其它 function使用的xyxy (可能是原始的 也可能是轉換後的)
        self.ori_xyxy = xyxy # 原始 yolo預測出來的 bbox
        self.yolo_conf = conf
        self.name = label
        self.im0 = im0
        self.cut_img_not_resize = None
        self.cut_img_resize = None
        self.has = None
        self.conf = None # conf_thre 預測結果 信心超過多少 才標註為 有穿戴
        self.transform_xyxy = None # 經過轉換後，非原始yolo預測的 bbox (可以用來畫框及裁剪)
    
    def cut_img(self):
        cut_xyxy = None
        if self.transform_xyxy is not None:
            cut_xyxy = self.transform_xyxy
        else:
            cut_xyxy = self.ori_xyxy
        self.cut_img_not_resize = cut(self.im0, cut_xyxy)
        self.cut_img_resize = cut(self.im0, cut_xyxy, RESIZE_SHAPE)

    def larger_bbox(self, left, top, right, down):
        # 可以自行選擇要增加 多少的bbox 範圍
        x0, y0, x1, y1 = self.xyxy
        img_h, img_w = self.im0.shape[:2]
        new_xyxy = (max(0, x0-left), max(0, y0-top),
                    min(img_w, x1+right), min(img_h, y1+down))  
        
        self.transform_xyxy = new_xyxy
        self.xyxy = new_xyxy
        

from ..utils.img_utils import preprocess_frame_to_yolo_one_cam
class PartYoloDetector():
    def __init__(self, part_list, detect_parts, weights, conf_thres=0.6):
        parser = argparse.ArgumentParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # weights_path = os.path.join(current_dir, 'weights', 'nose_wrist_20000.pt')
        weights_path = os.path.join(current_dir, 'weights', weights)
        parser.add_argument('--weights', nargs='+', type=str, default=weights_path, help='model.pt path(s)')

        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

        parser.add_argument('--conf-thres', type=float, default=conf_thres, help='object confidence threshold')

        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        
        opt = parser.parse_args()

        # self.classes = [i for i,c in enumerate(part_list) if c in detect_parts]
        self.update_detect_part(part_list, detect_parts)

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
        names = part_list[:len(names)]

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

    def update_detect_part(self, part_list, detect_parts):
        self.classes = [i for i,c in enumerate(part_list) if c in detect_parts]

    def detect_frame(self, img, fence_setting=None, first_flag=False):
        s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in img], 0)  # shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        
        img_copy = img.copy()
        if fence_setting:
            fence_enable, fence_roi = fence_setting 
            if fence_enable == 'true':
                fences_int = init_fence_point_to_int(fence_roi)
                points = np.array(fences_int[0], np.int32)
                points = points.reshape((-1, 1, 2))
                # ROI 圖像分割
                mask = np.zeros(img_copy[0].shape, np.uint8)
                # 画多边形
                mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
                img[0] = cv2.bitwise_and(mask2, img_copy[0])

        img, im0s = preprocess_frame_to_yolo_one_cam(img, img_size=self.imgsz, stride=self.stride, auto=rect)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        if first_flag == True:  #首張測試圖就不限制類別
            tmp_class=[0]
        else:
            tmp_class=self.classes
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=tmp_class, agnostic=self.opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '%g: ' % i, img_copy[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string


                yolo_rectangles = []
                for *xyxy, conf, cls in reversed(det):
                    which_part = self.names[int(cls)]
                    rect = Yolo_Rectangle(
                            xyxy, conf, self.names[int(cls)], im0
                        )
                    if which_part == 'body':
                        # 身體的框要特別加大，所以在 class Yolo_Rectangle中多寫了一個方法來處裡放大
                        # 參數順序為 左 上 右 下 分別要加大多少
                        rect.larger_bbox(100, 40, 100, 40)

                    rect.cut_img()
                    yolo_rectangles.append(rect)

                return yolo_rectangles, im0
            else:
                return [], im0

        