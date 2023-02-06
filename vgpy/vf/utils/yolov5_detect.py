import time
t_start = time.time()

import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized

from utils.plots import plot_one_box, plot_all_fense, ResizeWithAspectRatio

from utils.calculate import *

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()


weights, imgsz = opt.weights, opt.img_size

# Initialize
set_logging()
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()  # to FP16

cudnn.benchmark = True  # set True to speed up constant image size inference

# Get names and colors
class_names = model.module.names if hasattr(model, 'module') else model.names # 取得模型所有類別
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

print('model 載入時間 %.2f 秒' % (time.time() - t_start) )


##### 全域變數 #####
IMG_WIDTH = None
IMG_HEIGHT = None
class Detector():
    def __init__(self, web_controller, source='0'):
        global IMG_WIDTH, IMG_HEIGHT
        self.result_img = None
        self.new_img_f = 0
        self.dataset = LoadStreams(source, img_size=imgsz, stride=stride)    
        self.fences_int = None    

        self.width = int(self.dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        IMG_WIDTH = self.width
        IMG_HEIGHT = self.height
        
        # alarm_thread 用的變數
        self.alarm_flag = 0
        self.alarm_distance = 0

        self.web_controller = web_controller
        self.count = 0
        self.start_time = None


    def update_fences(self, new_fences): # 從網頁更新 圍籬 範圍後 ， 要更新近來
        self.fences_int = init_fence_point_to_int(new_fences)

    def detect(self, ptsArr, least_safe_distance=0):
        ##### 初始化變數 #####
        rm_class = ['Danger'] # 要移除的類別
        select_class = [_class for _class in class_names if _class not in rm_class] # 留下來的類別

        result_img, alarm_flag, min_distance = None, None, None
        ##### 初始化變數 #####

        self.fences_int = init_fence_point_to_int(ptsArr) # 圍籬的點座標 轉換成 int
        self.start_time = time.time()
        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0] # (1, 3, 384, 640)

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            result_img = None
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '%g: ' % i, im0s[i].copy(), self.dataset.count                
                # 變數 s 的 範例 「0: 480x640 1 cup, 1 tv, 1 laptop, 1 mouse, 1 keyboard, Done. (0.010s)」            
                s += '%gx%g ' % img.shape[2:]  # print string
                result_img = im0.copy()

                ############ 將偵測後的影像 顯示到螢幕上 ###########
                # if im0 is not None:
                #     # cv2.imshow("AUO", im0)
                #     cv2.imshow("AUO", ResizeWithAspectRatio(im0, width=600))                    
                #     cv2.waitKey(1)

                # 畫出 所有 虛擬圍籬的 點 與 線
                plot_all_fense(self.fences_int, img=result_img, color=(0,0,255), line_thickness=2)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()                    
                    alarm_rectangle = [] # (dist, xyxy) # 存下每個 bbox 到 每個圍籬的最短距離 與 該 bbox 之 xyxy

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {class_names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in reversed(det):
                        label = f'{class_names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, result_img, label=label, color=colors[int(cls)], line_thickness=3)

                        predicted_class = class_names[int(cls)]

                        if predicted_class not in select_class:
                            continue

                        box_middle_point = get_bbox_middle_point(xyxy, IMG_WIDTH, IMG_HEIGHT)
                        box_middle_point = tuple(int(p) for p in box_middle_point) # 將 Bounding Box 的中心點座標轉成整數
                        
                        for fence_points in self.fences_int:
                            # 取得目前的這個 Bbox 與 目前的圍籬的各個點比較後 的 最短距離 + 是圍籬的哪個點
                            shortest_dist, nearest_point = get_bbox_to_fence_shortest_distance_and_point(box_middle_point, fence_points)
                            # nearest_point: 該圍籬的某個點， 其與 bbox 最接近的， shortest_dist 為此兩點的距離

                            alarm_rectangle.append( tuple([shortest_dist, xyxy]) )

                            # 將 最近圍籬點 連到該 bbox 之 中心
                            cv2.circle(result_img, nearest_point, 5, (0, 0, 255), -1)
                            cv2.line(result_img, pt1=nearest_point, pt2=box_middle_point, color=(255,0,0), thickness=1)

                            # 最近圍籬點 與 bbox 中心， 兩點之連線的中點要 放入 距離多遠 的文字
                            sum_x = nearest_point[0] + box_middle_point[0]
                            sum_y = nearest_point[1] + box_middle_point[1]
                            dist_text_point = (int(sum_x/2), int(sum_y/2))
                            cv2.putText(result_img,'{:.0f}(cm)'.format(shortest_dist), 
                                        dist_text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), thickness=2)

                    ############ 將距離圍籬最近的物件 框起來 ###########
                    if len(alarm_rectangle):
                        alarm_rectangle = sorted(alarm_rectangle, key=lambda x:x[0])
                        short_x0, short_y0, short_x1, short_y1 = alarm_rectangle[0][1]
                        cv2.rectangle(result_img, pt1=(short_x0+5, short_y0+5), 
                                    pt2=(short_x1, short_y1), color=(0,0,255), thickness=6)
    
                        ############ 檢測距離是否超標 ###########
                        min_distance = alarm_rectangle[0][0]
                        if min_distance < least_safe_distance: #確認最短距離是否超標
                            self.alarm_flag = 1
                            self.alarm_distance = min_distance
                            cv2.putText(result_img, 'Alarm Distance: {:.0f} (cm)'.format(min_distance),(int(self.width*0.65),15), 
                                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(76,46,255),thickness=2)
                        else:
                            self.alarm_flag = 0
                    

                    # if result_img is not None:
                    #     # cv2.imshow("AUO", im0)
                    #     cv2.imshow("AUO_Detected", ResizeWithAspectRatio(result_img, width=800))                    
                    #     cv2.waitKey(1)

            ########### 偵測完一張影像 ###########
            # self.count += 1
            # print('fps:', (self.count / (time.time()-self.start_time) ))

            self.result_img = result_img
            self.new_img_f = 1
            self.web_controller.setting_control(im0, result_img)

            ############ 將偵測後的影像 顯示到螢幕上 ###########
            # if result_img is not None:
            #     cv2.imshow("AUO", result_img)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         pass
            


                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')

                

                    

