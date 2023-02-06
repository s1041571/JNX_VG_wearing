import time
import argparse
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
from threading import Thread
from datetime import timedelta, date

from .models.experimental import attempt_load
from .utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device

from .. vf.vf_backend import ALARM_IMG
from ..vf.utils.plots import plot_all_fense
from ..vf.utils.calculate import *
from ..vf.utils.speech_out_play import play_mp3
from ..vf.utils.LINE_Notify import line_notify_message_ndarray_img
from ..utils.mail_utils import send_email, send_email_ndarray_img
from ..utils.img_utils import preprocess_frame_to_yolo_one_cam, put_zh_text_opencv
from ..utils.json_utils import load_json, save_json
from ..config import color
from vgpy.global_object import GlobalVar
from ..utils.Linkpost import Linkpost

# from ..utils.relay_control import RelayControl
# import serial

parser = argparse.ArgumentParser()
# parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS') #設定濾除物件重疊率
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
imgsz = opt.img_size

# Initialize
set_logging()
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA


##### 全域變數 #####
IMG_WIDTH = None
IMG_HEIGHT = None
select_class = []

from ..vf import vf_backend as vfbackend

WEIGHTS_DIR = os.path.join(current_dir, 'weights')
class Detector():
    def __init__(self, vfvar: vfbackend.VfVar, gvar:GlobalVar, conf_thres=0.6, weights="yolov5s.pt"):
        global IMG_WIDTH, IMG_HEIGHT
        self.conf_thres = conf_thres

        weights_path = os.path.join(current_dir, 'weights', weights)
        self.weights = weights_path
        self.select_class = vfvar.current_model_class # Wen新增 載入上次
        self.classes = []   # Wen add, 記錄當前frame內出現的class
        self.model_reload()

        self.result_img = None
        self.new_img_f = 0
        self.fences_int = None    

        # TODO 之後要改成可以跟著 攝影機的大小切換
        self.width, self.height = gvar.cam_wh
        # self.width = 1920
        # self.height = 1080 
        
        IMG_WIDTH = self.width
        IMG_HEIGHT = self.height
        
        # alarm_thread 用的變數
        self.alarm_var = self.AlarmVar()
        self.alarm_level = 0
        self.alarm_play_time = time.time()
        self.alarm_thread = None
        
        self.count = 0
        self.start_time = None

        ##### 初始化變數 #####
        self.vfvar = vfvar
        self.gvar = gvar
        ## Wen 更新 ================================================================#
        
        # 物件模式
        self.N_FRAMES = None          #最近 [1]個frame中，有[0]個超過門檻值的frame，就會發警告
        self.obj_max_count = 0        #範圍內最多能容許的物件數
        self.obj_frame_count = []     #計算frame數

        if self.vfvar.obj_detect_mode:
            self.update_fences(self.vfvar.cam_obj_areas) # 圍籬的點座標 轉換成 int
            self.alarm_setting = self.vfvar.obj_alarm_set
            self.N_FRAMES = (self.alarm_setting['alarm_threhold'].get('ng_frame'), self.alarm_setting['alarm_threhold'].get('total_frame')) 
            self.obj_max_count = self.alarm_setting['alarm_threhold'].get('obj_max_count')
        else:
            self.update_fences(self.vfvar.cam_vf_areas) 
            self.alarm_setting = self.vfvar.vf_alarm_set
    
        self.log_save_day = self.alarm_setting.get('log').get('save_day')  if self.alarm_setting.get('log') else 30
        self.config_path =  os.path.join(os.getcwd(), 'vgpy', 'vf', 'config')
        self.notify_thread = None   # Wen新增 因應XR展開發
        # self.relay_control = self.RelayControl()  
        self.relay_last_id = None
        # self.serialPort1 = serial.Serial('COM4', 9600)
        # self.serialPort2 = serial.Serial('COM5', 9600)
        # self.onMsg  = [ b'\xA0\x01\x01\xA2', b'\xA0\x02\x01\xA3', b'\xA0\x03\x01\xA4', b'\xA0\x04\x01\xA5' ]
        # self.offMsg = [ b'\xA0\x01\x00\xA1', b'\xA0\x02\x00\xA2', b'\xA0\x03\x00\xA3', b'\xA0\x04\x00\xA4' ]
        #===========================================================================#

    def update_fences(self, new_fences): # 從網頁更新 圍籬 範圍後，要更新進來
        self.fences_int = init_fence_point_to_int(new_fences)

    def update_alarm_setting(self, new_setting):    #前端更新異常發報設定檔，要更新
        self.alarm_setting = new_setting
        self.log_save_day = self.alarm_setting.get('log').get('save_day') if self.alarm_setting.get('log') else 30
        if self.vfvar.obj_detect_mode:
            self.N_FRAMES = (self.alarm_setting.get('alarm_threhold').get('ng_frame'), self.alarm_setting.get('alarm_threhold').get('total_frame')) 
            self.obj_max_count = self.alarm_setting.get('alarm_threhold').get('obj_max_count')



    def reload_model_weights(self,new_weights,new_classes): #從網頁 切換模型 後，要進來重載 (Wen新增)
        self.weights = os.path.join(WEIGHTS_DIR, new_weights)
        self.select_class = new_classes
        self.model_reload()

    def model_reload(self):
        global imgsz, class_names, select_class, colors, model
        t_start = time.time()
        # Load model
        import sys
        sys.path.insert(0, './vgpy/yolov5') # 要用相對路徑 加入 yolo 資料夾的所在位置，不然 attempt_load 會報錯
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        del sys.path[0]

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

        ##### 初始化變數 #####
        rm_class = [ 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush' ]# 要移除的類別
        # rm_class = ['person']

        # TODO 控制要辨識的類別
        if self.select_class:
            select_class = self.select_class
        else:
            select_class = [_class for _class in class_names if _class not in rm_class] # 留下來的類別
        print('model 載入時間 %.2f 秒' % (time.time() - t_start) )


    def detect_img(self, img):    
        result_img, min_distance = None, None
        
        self.start_time = time.time()

        im0 = img.copy()

        if self.vfvar.obj_detect_mode:
            try:
                points = np.array(self.fences_int[0], np.int32)
                points = points.reshape((-1, 1, 2))
                # ROI 圖像分割
                mask = np.zeros(im0.shape, np.uint8)
                # 画多边形
                mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
                img = cv2.bitwise_and(mask2, im0)
                # print("self.vfvar.obj_detect_mode")
            except:
                img = np.zeros(im0.shape, np.uint8) # 全黑
        
        img, _ = preprocess_frame_to_yolo_one_cam([img])

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 012 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0] # (1, 3, 384, 640)
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        

        # Process detections
        s = '%g: ' % 0
        # 變數 s 的 範例 「0: 480x640 1 cup, 1 tv, 1 laptop, 1 mouse, 1 keyboard, Done. (0.010s)」            
        s += '%gx%g ' % img.shape[2:]  # print string
        result_img = im0.copy()

        ############ 將偵測後的影像 顯示到螢幕上 ###########
        # if im0 is not None:
        #     # cv2.imshow("AUO", im0)
        #     cv2.imshow("AUO", ResizeWithAspectRatio(im0, width=600))                    
        #     cv2.waitKey(1)

        # 畫出 所有 虛擬圍籬的 點 與 線
        plot_all_fense(self.fences_int, img=result_img, color=(255,0,255), line_thickness=4)
        det = pred[0]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()                    
            alarm_rectangle = [] # (dist, xyxy) # 存下每個 bbox 到 每個圍籬的最短距離 與 該 bbox 之 xyxy
            self.classes = []
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {class_names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):
                label = f'{class_names[int(cls)]} {conf:.2f}'
                predicted_class = class_names[int(cls)]

                if predicted_class not in select_class:  #如果沒有在欲辨識清單內，則By pass
                    self.alarm_var.alarm_distance = None
                    continue
                
                if predicted_class not in self.classes: #記錄這frame內出現的class
                    self.classes.append(predicted_class)
                plot_one_box(xyxy, result_img, label=label, color=colors[int(cls)], line_thickness=3)   #物件畫框


                if not self.vfvar.obj_detect_mode:
                    box_middle_point = get_bbox_middle_point(xyxy, IMG_WIDTH, IMG_HEIGHT)
                    box_middle_point = tuple(int(p) for p in box_middle_point) # 將 Bounding Box 的中心點座標轉成整數
                    xy_point = tuple(int(p) for p in xyxy)
                    box_point_list = [(xy_point[0],xy_point[1]), (xy_point[0],xy_point[3]),(xy_point[2],xy_point[1]),(xy_point[2],xy_point[3]),box_middle_point]
                    for fence_points in self.fences_int:
                        # # 取得目前的這個 Bbox 與 目前的圍籬的各個點比較後 的 最短距離 + 是圍籬的哪個點
                        # shortest_dist, nearest_point = get_bbox_to_fence_shortest_distance_and_point(box_middle_point, fence_points)
                        # # nearest_point: 該圍籬的某個點， 其與 bbox 最接近的， shortest_dist 為此兩點的距離
                        shortest_dist = 0
                        for i, box_point in enumerate(box_point_list) :
                            dist, point = get_bbox_to_fence_shortest_distance_and_point(box_point, fence_points, self.gvar.cam_wh)
                            if shortest_dist == 0:
                                shortest_dist, nearest_point, box_point_index = dist, point, i
                            elif dist < shortest_dist :
                                shortest_dist, nearest_point, box_point_index = dist, point, i

                        alarm_rectangle.append( tuple([shortest_dist, xyxy]) )

                        # 將 最近圍籬點 連到該 bbox 之 中心
                        cv2.circle(result_img, nearest_point, 5, (0, 0, 255), -1)
                        cv2.line(result_img, pt1=nearest_point, pt2=box_point_list[box_point_index], color=(255,0,0), thickness=3)

                        # 最近圍籬點 與 bbox 中心， 兩點之連線的中點要 放入 距離多遠 的文字
                        sum_x = nearest_point[0] + box_point_list[box_point_index][0]
                        sum_y = nearest_point[1] + box_point_list[box_point_index][1]
                        dist_text_point = (int(sum_x/2), int(sum_y/2))
                        cv2.putText(result_img,'{:.0f}(cm)'.format(shortest_dist), 
                                dist_text_point, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color.亮綠, thickness=5)
                else:
                    alarm_rectangle.append(xyxy)

            ############ 將距離圍籬最近的物件 框起來 ###########
            if len(alarm_rectangle):
                font_size = int(self.gvar.cam_wh[0]/1920*100)
                if not self.vfvar.obj_detect_mode:
                    alarm_rectangle = sorted(alarm_rectangle, key=lambda x:x[0])
                    short_x0, short_y0, short_x1, short_y1 = list(map(int, alarm_rectangle[0][1]))
                    cv2.rectangle(result_img, pt1=(short_x0+5, short_y0+5), 
                                pt2=(short_x1, short_y1), color=(0,0,255), thickness=6)

                    ############ 檢測距離是否超標 ###########
                    min_distance = alarm_rectangle[0][0]
                    # if min_distance < self.vfvar.least_safe_distance: #確認最短距離是否超標
                    if 1==1:
                        self.alarm_var.alarm_distance = min_distance
                        self.alarm_level = self.get_alarm_level()
                        font_scale = 2.5
                        if self.alarm_level is not None:
                            text, text_color = None, None 
                            # if self.alarm_level != self.relay_last_id: #目前燈號與上個燈號不同 
                                # self.relay_control.relay_turnOFF()     #關掉當前燈號
                            if self.alarm_level == 1:
                                text = "危險"
                                text_color = color.危險
                                # self.relay_control.relay_turnON(1)
                                # self.relay_last_id = 1
                            elif self.alarm_level == 2:
                                text = "警戒"
                                text_color = color.警戒
                                # self.relay_control.relay_turnON(2)
                                # self.relay_last_id = 2
                            elif self.alarm_level == 3:
                                text = "小心"
                                text_color = color.注意
                                # self.relay_control.relay_turnON(3)
                                # self.relay_last_id = 3
                            text = text+': {:.0f} (cm)'.format(min_distance)
                            result_img = put_zh_text_opencv(
                                result_img, text,
                                (int(self.gvar.cam_wh[0]*0.6), 20),
                                text_color, font_size
                            )
                            if not self.alarm_var.alarm_running:
                                self.alarm_play(result_img, self.classes)
                        # cv2.putText(
                        #     result_img, 'Alarm Distance: {:.0f} (cm)'.format(min_distance),
                        #     (int(self.width*0.45), int(15*font_scale*3)),
                        #     cv2.FONT_HERSHEY_SIMPLEX, font_scale , (76,46,255), thickness=5)
                        else:
                            if self.relay_last_id != None:  #未辨識到警示等級距離&有燈號狀態
                                self.relay_last_id == None
                                # self.relay_control.relay_turnOFF()    #關掉所有警示燈
                else:
                    print("object mode")
                    font_scale = 2
                    result_img = put_zh_text_opencv(
                        result_img, 'Object Qty: {}'.format(len(alarm_rectangle)),
                        (35,40), color.亮綠, font_size
                    )
                    result_img = put_zh_text_opencv(
                        result_img, 'INTRUSION: {}'.format(",".join(self.classes)),
                        (35, 140), color.危險, font_size
                    )
                    #rect數超標 append 1, 未超標 append 0
                    if len(alarm_rectangle) >= self.obj_max_count:
                        self.obj_frame_count.append(1) 
                    else:
                        self.obj_frame_count.append(0)

                    if len(self.obj_frame_count) > self.N_FRAMES[1]:
                        if sum(self.obj_frame_count) >= self.N_FRAMES[0]: 
                            self.alarm_play(result_img, self.classes)
                        self.obj_frame_count = self.obj_frame_count[-self.N_FRAMES[1]:] # 只保留最近 N frame的結果


        else: # 如果沒偵測到東西
            self.alarm_var.alarm_distance = None
            if self.relay_last_id != None:
                self.relay_last_id == None
                # self.relay_control.relay_turnOFF() #關掉警示燈


        ########### 偵測完一張影像 ###########
        # self.count += 1
        # print('fps:', (self.count / (time.time()-self.start_time) ))

        ########### 將偵測後的影像 顯示到螢幕上 ###########
        # if result_img is not None:
        #     cv2.imshow("AUO", result_img)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         pass

        self.result_img = result_img
        self.new_img_f = 1
        return result_img, im0

        ############ 將偵測後的影像 顯示到螢幕上 ###########
        # if result_img is not None:
        #     cv2.imshow("AUO", result_img)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         pass
    
    def get_alarm_level(self):
        distance = self.alarm_var.alarm_distance
        level_set = self.alarm_setting['level'] #取得alarm level setting
        alarm_level = None
        if distance < level_set['lv1']:
            alarm_level = 1
        elif distance < level_set['lv2']:
            alarm_level = 2
        elif distance < level_set['lv3']:
            alarm_level = 3
        return alarm_level

    class AlarmVar:
        def __init__(self):
            self.notify_disable = False
            # self.line_disable = False
            # self.mail_disable = False
            self.alarmDate = date.today()-timedelta(days=1)
            self.previous_alarm_time = None
            self.alarm_running = False
            self.alarm_distance = None
            self.audio_text = None
            self.audio_speed = None
            #===== XR展 新增 ==================#
            self.alarm_notify_running = False   #是否正在執行通報執行緒flag
            self.notify_img = None              #通報發送照片
            #==================================#
            self.count = dict()

    def alarm_play(self, img, classes):
        if time.time() - self.alarm_play_time > 10: # 超過 10秒 沒有進來 alarm_play，就清空 count
            for k,v in self.alarm_var.count.items():
                self.alarm_var.count[k] = 0

        # # line發出警告後，要維持安全距離超過 30秒 後 才可重複發報Line
        if self.alarm_var.notify_disable and (time.time() - self.alarm_var.previous_alarm_time) > 30:  
            self.alarm_var.notify_disable = False
        # # Mail 與 LINE 同上機制(避免重複寄送郵件)
        # if self.alarm_var.mail_disable and (time.time() - self.alarm_var.previous_alarm_time) > 30:  
        #     self.alarm_var.mail_disable = False

        self.alarm_play_time = time.time()

        def daemon_alarm_play(var: Detector.AlarmVar):
            history_json = vfbackend.get_history_mp3_json()

            audio_text = var.audio_text
            speed = var.audio_speed
            audio_is_exist = False
            audio_index = 0
            for idx, text_speed_tuple in enumerate(history_json['history']):
                t, s = text_speed_tuple
                if t == audio_text and s == speed:
                    audio_is_exist = True
                    audio_index = idx
                    break

            if audio_text is not None: # 有符合 任一個距離的範圍
                if not audio_is_exist:
                    # audio_index = len(history_json['history'])
                    # filepath = os.path.join(vfbackend.AUDIO_DIR, f'{audio_index}.mp3')
                    # text_to_mp3_file_and_play(audio_text, speed, filepath)
                    # print("audio_is_not_exist Create")
                    
                    # history_json['history'].append((audio_text, speed))
                    # vfbackend.save_history_mp3_json(history_json)
                    pass
                else:
                    filepath = os.path.join(vfbackend.AUDIO_DIR, f'{audio_index}.mp3')
                    play_mp3(filepath)
            self.alarm_var.alarm_running = False
        
        def get_alarm_audio_text():            
            audio_text = None
            speed = 1
            distance = self.alarm_var.alarm_distance
            if self.alarm_level == 1:
                audio_text = "危險，危險，危險"
                speed = 1.4                
                # print ("一級警報，距離：{:.0f}公分".format(distance))                                

            elif self.alarm_level == 2:
                audio_text = "警戒，警戒"
                speed = 1.2
                # print ("二級警報，距離：{:.0f}公分".format(distance))

            elif self.alarm_level == 3:
                audio_text = "注意，注意"
                speed = 1.2
                # print ("三級警報，距離：{:.0f}公分".format(distance))
            
            return audio_text, speed
        
        def daemon_run_notify(alarm_var: Detector.AlarmVar, classes):  #通報執行緒的執行func
            mode = self.vfvar.obj_detect_mode
            if mode:
                msg_title, msg_text = "《V·Guard 物件警報系統 警報通知》", \
                "虛擬圍籬偵測到有違禁品，請盡速前往察看注意！"
            else:
                msg_title, msg_text = "《V·Guard 測距警報系統 - 警報通知》", \
                "虛擬圍籬偵測到人員，可能發生危險，請盡速前往察看注意！"
            channel_set = self.alarm_setting['channel']
            if 'link' in channel_set['channel']:
                Linkpost(camFrame=alarm_var.notify_img, msg_title=msg_title, msg_text=msg_text)
                print("Link 已傳送")
            if 'line' in channel_set['channel']:
                line_notify_message_ndarray_img(msg_text, alarm_var.notify_img)
                print('LINE Notify MessageImg:', msg_text)
                # alarm_var.line_disable = True
            if 'mail' in channel_set['channel']:
                mail_status = send_email_ndarray_img(channel_set['mail_group'], msg_title, msg_text, None, [alarm_var.notify_img])
                print('E-mail 發送結果:'+mail_status)
                # alarm_var.mail_disable = True
            
            # 儲存影像
            img_filename = time.strftime("%Y%m%d_%H%M%S", time.localtime())+'.jpg'
            img_dir_path = os.path.join(os.getcwd(), 'vgpy', 'vf', 'img', 'alarm_msg')
            cv2.imwrite(os.path.join(img_dir_path,img_filename), alarm_var.notify_img) 
            # 儲存alarm log
            self.update_alarm_log(img_filename, classes)
            alarm_var.alarm_notify_running = False


        def run_alarm_process(audio_text=None, speed=None):
            vfbackend.save_alarm_img(self.result_img)
            if audio_text is None:
                self.alarm_var.audio_text, self.alarm_var.audio_speed = get_alarm_audio_text()
            else:
                self.alarm_var.audio_text, self.alarm_var.audio_speed = audio_text, speed
            # time.sleep(0.5)
            # 開執行序單獨播放語音，不讓辨識停駐
            self.alarm_thread = Thread(target=daemon_alarm_play, args=(self.alarm_var, ), daemon=True)
            self.alarm_thread.start()

        def run_notify_process(img, classes):  #啟動通報執行緒
            self.alarm_var.notify_img = img
            self.notify_thread = Thread(target=daemon_run_notify, args=(self.alarm_var, classes), daemon=True, name="run_alarm_notify")
            self.notify_thread.start()

        if not self.vfvar.obj_detect_mode:
            audio_text, speed = get_alarm_audio_text()
            count_key = (audio_text, speed)
            if count_key not in self.alarm_var.count:
                self.alarm_var.count[count_key] = 1
            else:
                self.alarm_var.count[count_key] += 1

            for k,v in self.alarm_var.count.items():
                if v >= 15:
                    if self.alarm_level == 1 and not self.alarm_var.notify_disable: #卡事故清場後，再次發生異常的空窗時間
                        if not self.alarm_var.alarm_notify_running:
                            self.alarm_var.alarm_notify_running = True
                            self.alarm_var.notify_disable = True
                            run_notify_process(img, classes)
                            print('發報訊息')
                        self.alarm_var.previous_alarm_time = time.time()

                    if not self.alarm_var.alarm_running:
                        self.alarm_var.alarm_running = True
                        run_alarm_process(audio_text, speed)
                    else:
                        pass
                        # 如果 最新偵測到的距離所屬的危急程度 跟 正在播放的危急程度不同，就要覆蓋播放
                        # if speed != self.alarm_var.audio_speed or audio_text != self.alarm_var.audio_text:
                        #     run_alarm_process(audio_text, speed)
                    
                    self.alarm_var.count[k] = 0
                    break

        else:
            if not self.alarm_var.notify_disable: #卡事故清場後，再次發生異常的空窗時間
                if not self.alarm_var.alarm_notify_running:
                    self.alarm_var.alarm_notify_running = True
                    self.alarm_var.notify_disable = True
                    run_notify_process(img, classes)
                    print('發報訊息')
                self.alarm_var.previous_alarm_time = time.time()

            if not self.alarm_var.alarm_running:
                self.alarm_var.alarm_running = True
                run_alarm_process('注意！注意！區域有違禁品', 1)
                print('呼叫 run_alarm_process')
            else:
                pass
            print("alarm_play obj finish")


    
    def update_alarm_log(self, img_filename, classes):
        img_dir_path = os.path.join(os.getcwd(), 'vgpy', 'vf', 'img', 'alarm_msg')
        channel_set = self.alarm_setting['channel']
        now_time= time.strftime("%Y%m%d", time.localtime())
        mode = "obj" if self.vfvar.obj_detect_mode else "vf"
        item_json = { 
            'time':time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'channel': ','.join(channel_set['channel']).upper(),
            'cam_id': self.gvar.cam_id,
            'weight': self.vfvar.current_model_weights,
            'classes':  ','.join(classes),
            'img': img_filename
        }
        wr_json ={ 
            now_time: [item_json]
        }
        if os.path.exists(os.path.join(self.config_path, 'alarm_msg_log.json')):
            current_json = load_json(self.config_path, 'alarm_msg_log.json')
            lastDate = date.today()-timedelta(days = self.log_save_day)
            for logDate in current_json[mode].copy().keys():
                if date(int(logDate[0:4]), int(logDate[4:6]), int(logDate[6:8])) < lastDate:
                    for item in current_json[mode][logDate]:
                        del_filename = item['img']
                        os.remove(os.path.join(img_dir_path,del_filename))
                    del current_json[mode][logDate]
                else:
                    break
            if current_json[mode].get(now_time):
                current_json[mode][now_time].append(item_json)
            else:
                current_json[mode].update(wr_json)
            save_json(current_json, self.config_path, 'alarm_msg_log.json')
        else:
            save_json(wr_json, self.config_path, 'alarm_msg_log.json')

        # if os.path.exists(os.path.join(self.config_path, 'alarm_msg_log.json')):
        #     current_json = load_json(self.config_path, 'alarm_msg_log.json')
        #     i = 0
        #     if len(current_json[mode]) > self.log_save_day:
        #         n =len(current_json[mode].copy())- self.log_save_day #要刪除的數量
        #         for key in current_json[mode].copy().keys():
        #             if(i < n):
        #                 for item in current_json[mode][key]:
        #                     del_filename = item['img']
        #                     os.remove(os.path.join(img_dir_path,del_filename))
        #                 del current_json[key]
        #             else:
        #                 break
        #             i+=1
        #     if current_json[mode].get(now_time):
        #         current_json[mode][now_time].append(item_json)
        #     else:
        #         current_json[mode].update(wr_json)
        #     save_json(current_json, self.config_path, 'alarm_msg_log.json')
        # else:
        #     save_json(wr_json, self.config_path, 'alarm_msg_log.json')

