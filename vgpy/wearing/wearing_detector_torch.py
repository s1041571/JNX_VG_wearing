from ..yolov5.yolo_head_wrist import PartYoloDetector
from ..utils.img_utils import put_zh_text_opencv, plot_yolo_box_to_yolo_rectangle, ImageSaverTime, cv2_save_zh_img
from ..config import color
from .config import WearingConfig
from .config import PART_LIST, PRED_SAVE_DIR_HEAD, ALARM_AUDIO_DIR, current_dir, SHUFFLE_NET_MODEL_DIR, ALARM_IMG_DIR, CONFIG_DIR
from .alarm_mp3_maker import AlarmMP3Maker
import pyttsx3

import numpy as np
import cv2
import torch

import os
import time
from threading import Thread
from ..utils.mail_utils import send_email, send_email_ndarray_img
from ..vf.utils.LINE_Notify import line_notify_message_ndarray_img
from .utils.plots import plot_all_fense
from .utils.calculate import init_fence_point_to_int
from ..utils.json_utils import load_json, save_json
from ..global_object import GlobalVar
from ..utils.Linkpost import Linkpost

from .torch_model import get_eval_shuffleNet_v2_model, ParallelWearingModel

wearing_config = None
device = torch.device('cuda')
dtype = torch.float32

SAVE_PREDICTION = True
PLOT_YOLO_RECTANGLE = True
if SAVE_PREDICTION and not os.path.isdir(PRED_SAVE_DIR_HEAD):
    os.makedirs(PRED_SAVE_DIR_HEAD)
    print('create dir', PRED_SAVE_DIR_HEAD)


from vgpy.utils.logger import create_logger    
logger = create_logger()


class MergePredictor:
    def __init__(self, clfs, clfs_names, thres, part_clf_indexs):
        """
        part_clf_indexs: 每個部位要辨識的裝備，在 clfs 中的哪些 index
        {
            "頭部": [0, 1],
            "手部": [2, 3],
        }
        """        
        self.set_conf_thres(thres) # 各預測類別 的 信心門檻 Ex: [0.5, 0.5]
        self.part_clf_indexs = part_clf_indexs

        test_img = torch.ones((1,3,64,64)) # 64*64 為我們 ShuffleNet_v2 模型的 input size
        test_img = test_img.to(device)

        self.merge_model = ParallelWearingModel(clfs)
        self.merge_model(test_img)        
        logger.info('穿戴:MergePredictor初始化辨識完成')

    def set_conf_thres(self, thres):
        self.conf_thres = np.array(thres).reshape(-1, 1)

    def predict(self, rects):
        # 輸入全部的照片，一次做預測，然後再將結果分類
        X = np.array([rect.cut_img_resize for rect in rects])
        # X = X[:10]
        X = X[:,:,:,::-1].transpose(0,3,1,2)
        X = torch.tensor(X.copy(), device=device, dtype=dtype)

        pred = self.merge_model(X)
        pred = np.array(pred)
        pred = pred.reshape(pred.shape[:2]) # 預測結果的維度會是 (M, N, 1) : M個clf，N個yolo rectangle，要變成(M,N)
        has_matrix = pred > self.conf_thres
        # 把預測結果 儲存到 Yolo_Rectangle 物件中
        for i, rect in enumerate(rects):
            model_idxs = self.part_clf_indexs[rect.name]
            rect.conf = list(pred[model_idxs, i])
            rect.has = list(has_matrix[model_idxs, i])


class Plotter:
    def __init__(self, name, label_names, fontsize, leading, has_color, no_color):
        """            
            fontsize    字大小
            leading     行距
            has_color   有穿戴的時候 字的顏色
            no_color    沒穿戴的時候 字的顏色
        """
        self.name = name
        self.label_names = label_names
        self.fontsize = fontsize
        self.leading = leading
        self.has_color = has_color
        self.no_color = no_color
        

    def put_text_to_rect(self, rect, img):
        xyxy = rect.xyxy
        fontsize = self.fontsize
        for i, zipo in enumerate(zip(rect.conf, rect.has, self.label_names)):
            conf, has, cls_name = zipo
            text_base_y = (xyxy[1] - 50) + (fontsize + self.leading)*i # 用來決定字 的 y 的位置            
            head = '有 ' if has else '無 '

            clf_result_text = "%s%s%.2f" % (head, cls_name, conf)

            if not has:
                img = put_zh_text_opencv(img, clf_result_text, (xyxy[2], text_base_y), self.no_color, fsize=fontsize)
            else:
                img = put_zh_text_opencv(img, clf_result_text, (xyxy[2], text_base_y), self.has_color, fsize=fontsize)        
        return img


    def plot_result(self, rects, img):
        for rect in rects:
            img = self.put_text_to_rect(rect, img)
        return img


def save_result(rects, part, cls_names):
    for rect in rects:
        for cls, has in zip(cls_names, rect.has):            
            if has:                
                save_dir = os.path.join(PRED_SAVE_DIR_HEAD, part, f"{cls}_has")
            else:                
                save_dir = os.path.join(PRED_SAVE_DIR_HEAD, part, f"{cls}_no")
            saver.save(rect.cut_img_not_resize, save_dir, '', verbose=True)


################ 讀取有要偵測的部位 的 有要偵測的類別 的模型 + 建立畫圖及預測的物件################
part_list = PART_LIST
N_FRAME_TO_SAVE = None
detect_parts = None
part_plotter_dict = dict()
saver = ImageSaverTime()

def load_config():
    global detect_parts, N_FRAME_TO_SAVE, saver, wearing_config

    if wearing_config is not None:
        wearing_config.load_config()
    else:
        wearing_config = WearingConfig()

    detect_parts = wearing_config.get_detect_part()
    # 處理 有選擇要偵測部位，但沒勾選要偵測的類別的話，把該偵測部位移除
    detect_parts = [part for part in detect_parts if len(wearing_config.get_detect_class(part))>0]
    

    N_FRAME_TO_SAVE = wearing_config.get_sampling_rate()
    saver.max_save_num = wearing_config.get_max_quantity()

    

from .wearing_api import WearingVar
class WearingDetector:
    def __init__(self, wearingvar:WearingVar, globalvar:GlobalVar):
        self.merge_predictor = None
        self.frame = 0
        self.N_FRAMES = (6, 8) # 最近 8個frame中，有6個frame沒穿戴，就會發警告
        self.yolo_detector = None
        self.part_detect_classes = None
        self.is_add_part = False # 記錄此次收集照片，YOLO是否有被強制加入要辨識的部位，收集完成後要根據該變數去決定是否要刪除強制加入的部位        
        self.no_wearing_count = dict() # 記錄每個裝備最近 N 個 frame 有無配戴
        self.no_wearing_img = dict() # 記錄每個裝備最近 N 個 frame 有無配戴 的照片
        self.mp3maker = AlarmMP3Maker(ALARM_AUDIO_DIR)
        self.wait_time = 0 # 根據 有幾個裝備沒穿戴 決定要等待多久不呼叫警報的 thread
        self.play_start_time = 0 # 啟動一個警報的 thread的起始時間
        self.wait_time_ratio = 2.0 # 由此處的數值，控制每個未穿戴警報 要等待幾秒
        self.wearingvar = wearingvar # Wen新增 為取得使用者設定要通報的頻道有哪些
        self.globalvar = globalvar # Jack新增，為了在subprocess中取得攝影機 id
        self.previous_alarm_time = time.time() # Wen新增 記錄最近一次發送警報訊息的時間
        self.alarm_disable = False   # Wen新增
        # if self.wearingvar.alarm_channel.get('log_save_day'):         # Wen新增 alarm_log 預設30天以後的資料將不再保存
        #     self.log_save_day = self.wearingvar.alarm_channel['log_save_day']
        # else:
        #     self.log_save_day = 30

        # self.line_disable = None    # Wen新增
        # self.mail_disable = None    # Wen新增
        self.fps_calculator = []
        self.fps_count_frame = 20 # 計算最近多少個 frame 的 fps

        # self.alarm_notify_enable = None #Wen新增 告知前端是否發送警報(為顯示警報清除按鈕用)
        self.init()
        self.yolo_detector_init(self.wearingvar.yolo_conf)
        self.no_wearing_count_init()
        self.no_wearing_img_init()

        #======== XR展 新增 ============#
        self.alarm_notify_running = False
        self.notify_thread = None
        self.notify_img = None
        # self.relay_control = RelayControl()
        self.relay_enable = False
        #===============================#
    def init(self):
        logger.info('穿戴:config初始載入中...')
        load_config()
        # 把 load_config() 讀到的 merge_predictor 存到 WearingDetector 的屬性
        self.init_merge_predictor()
        
        # 更新yolo模型 要辨識哪些部位
        if self.yolo_detector is not None:
            self.yolo_detector.update_detect_part(part_list, detect_parts)

        self.part_detect_classes = dict() # 儲存每個身體部位 要辨識哪些裝備
        for part in detect_parts:
            classes = wearing_config.get_detect_class(part) # 此部位 要辨識哪些裝備
            self.part_detect_classes[part] = classes

        # #FIXME:更改設定(重選部位)為重load yolo_detecter 懷疑detect_parts(辨識部位)未更新 -211029_Wen
        # self.yolo_detector = PartYoloDetector(
        #     part_list, detect_parts, weights='nose_wrist_body_foot.pt', conf_thres=self.wearingvar.yolo_conf)

        self.no_wearing_count_init() # 更換config後，要重新初始化 count
        self.no_wearing_img_init() # 更換config後，要重新初始化 count
        # print('wearing config loaded or reloaded')
        # logging.debug('wearing config loaded or reloaded')
        logger.info('穿戴:config初始載入成功')


    def update_config(self):
        load_config()
        logger.info('穿戴:config更新成功')

    def get_merge_predictor_conf_thres(self):
        # 取得個裝備預測的信心門檻值
        merge_class_confs = []
        for part in detect_parts:            
            class_confs = wearing_config.get_part_wearing_confs_in_detect_class(part)    
            merge_class_confs.extend(class_confs)
        return merge_class_confs

    def init_merge_predictor(self):
        global part_plotter_dict
        clf_list = []
        clf_name_list = []
        part_clf_indexs = dict()
        model_index = 0
        
        for part in detect_parts:
            detect_class = wearing_config.get_detect_class(part)
            part_clf_indexs[part] = []

            for cls in detect_class:
                model_path = os.path.join(current_dir, SHUFFLE_NET_MODEL_DIR, part, f'{cls}.ptst')
                model = get_eval_shuffleNet_v2_model('cuda')
                model.load_state_dict(torch.load(model_path))
                clf_list.append(
                    model
                )
                part_clf_indexs[part].append(model_index)
                clf_name_list.append(cls)
                model_index += 1

            # 畫出EfficientNet的辨識出的字體格式
            part_plotter_dict[part] = Plotter(
                part, detect_class, fontsize=30, leading=10, has_color=color.深綠, no_color=color.淡紅)
        
        merge_class_confs = self.get_merge_predictor_conf_thres()
        self.merge_predictor = MergePredictor(clf_list, clf_name_list, merge_class_confs, part_clf_indexs)
        logger.info('穿戴:MergePredictor初始化成功')
        

    def update_merge_predictor_confs(self):        
        self.merge_predictor.set_conf_thres(self.get_merge_predictor_conf_thres())
        logger.info('穿戴:MergePredictor更新辨識門檻值成功')


    def yolo_detector_init(self, yolo_conf):
        self.yolo_detector = PartYoloDetector(
                part_list, detect_parts, weights='nose_wrist_body_foot.pt', conf_thres=yolo_conf)
        logger.debug('yolo detector loaded')
    

    def __eff_predict_plot_save(self, im0, rectangles):
        yolo_batch = dict()
        result_img = im0.copy()

        if len(detect_parts) == 0: # 沒有設定要辨識哪些部位
            pass # TODO 如果沒勾選任何要偵測的部位，應該可以跳過下面的部分不執行，直接在這邊回傳
            
        for part in detect_parts:
            yolo_batch[part] = []

        # 按照yolo預測的結果 把框分類， 頭部一群； 手部一群
        for rect in rectangles:
            yolo_batch[rect.name].append(rect)


        self.merge_predictor.predict(rectangles)

        for part in part_list:
            # 該部位有列入要偵測的部位，且yolo有偵測到此部位
            if part in detect_parts and len(yolo_batch[part]):
                part_yolo_batch = yolo_batch[part] # yolo有偵測到 迴圈目前的這個 部位(part)

                plotter = part_plotter_dict[part]
                result_img = plotter.plot_result(part_yolo_batch, result_img)

                # 是否框出 yolo 預測的框
                if PLOT_YOLO_RECTANGLE:
                    for rect in part_yolo_batch:                                                
                        result_img = plot_yolo_box_to_yolo_rectangle(rect, result_img, color=(157, 207, 140), zh=True)

                # 把預測結果儲存下來
                if SAVE_PREDICTION and self.frame % N_FRAME_TO_SAVE == 0:
                    save_result(part_yolo_batch, part=part,
                        cls_names=wearing_config.get_detect_class(part))
        
        return result_img

    
    # def detect_img(self, img0, yolo_conf, print_spf=False):
    def detect_img(self, img0, yolo_conf, print_spf=False, first_flag=False):
        # yolo 取得部位後 輸入 eff 做裝備有無穿戴的辨識
        # print_spf 決定要不要計算 每個 frame 的處理時間是幾秒，並且print出來
        if self.yolo_detector is None:
            self.yolo_detector = PartYoloDetector(
                part_list, detect_parts, weights='nose_wrist_body_foot.pt', conf_thres=yolo_conf)
        
        #判斷虛擬圍籬是否開啟 再畫框
        if self.wearingvar.fence_setting:
            fence_enable, fence_roi = self.wearingvar.fence_setting
            if fence_enable == 'true':
                fences_int = init_fence_point_to_int(fence_roi)
                plot_all_fense(fences_int, img=img0, color=(0,0,255), line_thickness=2)

        img0 = [img0]

        #載入首張無意義照片進行辨識，縮短第一張辨識的等待時間
        if first_flag == True:
            rectangles, im0 = self.yolo_detector.detect_frame(img0, first_flag=first_flag)
            result_img = self.__eff_predict_plot_save(im0, rectangles)   
            return result_img, im0
        
        rectangles, im0 = self.yolo_detector.detect_frame(img0, self.wearingvar.fence_setting)

        if print_spf:
            self.fps_calculator.append(time.time())
            self.fps_calculator = self.fps_calculator[-self.fps_count_frame:]

        if len(rectangles):
            result_img = self.__eff_predict_plot_save(im0, rectangles)
            self.no_wearing_alarm(rectangles, result_img) # 偵測每個yolo框是否有穿戴裝備，達到一定次數沒穿戴，就會發警報
            self.frame+=1

            if print_spf:
                spf = (self.fps_calculator[-1] - self.fps_calculator[0]) / len(self.fps_calculator)
                print(f"最近 {len(self.fps_calculator)}個 frame 的 spf = {spf}")
        else:
            # self.relay_control.relay_turnOFF()
            result_img = im0

        # #判斷虛擬圍籬是否開啟 再畫框
        # if self.wearingvar.fence_setting:
        #     fence_enable, fence_roi = self.wearingvar.fence_setting
        #     if fence_enable == 'true':
        #         fences_int = init_fence_point_to_int(fence_roi)
        #         plot_all_fense(fences_int, img=result_img, color=(0,0,255), line_thickness=2)

        return result_img, im0
        

    def detect_img_part(self, img0, which_part, yolo_conf):
        # 收集照片用的function，將指定部位的照片裁剪後回傳 （就沒有輸入到efficient net 做辨識）
        if self.yolo_detector is None:
            self.yolo_detector = PartYoloDetector(
                part_list, detect_parts, weights='nose_wrist_body_foot.pt', conf_thres=yolo_conf)
        
        # #判斷虛擬圍籬是否開啟 再畫框
        # if self.wearingvar.fence_setting:
        #     fence_enable, fence_roi = self.wearingvar.fence_setting
        #     if fence_enable == 'true':
        #         fences_int = init_fence_point_to_int(fence_roi)
        #         plot_all_fense(fences_int, img=img0, color=(0,0,255), line_thickness=2)

        # img0 = [img0]

        rectangles, im0 = self.yolo_detector.detect_frame([img0], self.wearingvar.fence_setting)

        #判斷虛擬圍籬是否開啟 再畫框
        if self.wearingvar.fence_setting:
            fence_enable, fence_roi = self.wearingvar.fence_setting
            if fence_enable == 'true':
                fences_int = init_fence_point_to_int(fence_roi)
                plot_all_fense(fences_int, img=img0, color=(0,0,255), line_thickness=2)


        # img_with_box = im0.copy()
        img_with_box = img0
        # 在原始圖上標上Bonding-Box的框
        for rect in rectangles:                        
            img_with_box = plot_yolo_box_to_yolo_rectangle(rect, img_with_box, color.深綠, zh=True)

        result_imgs = []
        for rect in rectangles:
            if rect.name == which_part:
                # result_imgs.append(rect.cut_img_not_resize) # 回傳裁剪特定部位的照片 (沒有resize)
                result_imgs.append(rect.cut_img_resize)     # 回傳裁剪特定部位的照片 (有resize)
                                                            # 沒Resize時，Shape不一樣，進行np.array再轉換時，
                                                            # 就會出錯ValueError: could not broadcast input array from shape

        return np.array(result_imgs), img_with_box


    def add_collect_part(self, which_part):
        self.is_add_part = False
        # 要收集照片時，要強制 YOLO 辨識要蒐集的那個部位
        if which_part not in self.yolo_detector.classes:
            part_index = self.yolo_detector.names.index(which_part)            
            # self.yolo_detector.classes.append(part_index)
            self.yolo_detector.classes = [part_index] #收集照片 只辨識該部位的classes
            self.is_add_part = True

    def remove_collect_part(self):
        if self.is_add_part: # 如果開始收集照片時，有強制加入辨識部位的話，辨識完成後要刪除
            del self.yolo_detector.classes[-1]
            self.is_add_part = False

    def no_wearing_alarm(self, rectangles, img):
        # if self.relay_enable and (time.time() - self.previous_alarm_time) > 10:
        #     self.relay_enable = False
        #     self.relay_control.relay_turnOFF()

        ##避免重複發報 距離上次發報 30 秒後才可再發報
        # if self.line_disable and (time.time() - self.previous_alarm_time) > 30:
        #     self.line_disable = False
        # if self.mail_disable and (time.time() - self.previous_alarm_time) > 30:
        #     self.mail_disable = False`

        no_wearing_count_updated = self.__no_wearing_count_updated_init()
        # 傳入 eff net 辨識過後的 yolo rectangles，裡面會包含每個框是否有穿戴
        # for rec in rectangles: # 此方法是每個yolo偵測到的框，都會算入沒穿戴的count，導致人多的時候可能會很難發報
        #     classes = self.part_detect_classes[rec.name]
        #     for cls, has in zip(classes, rec.has):
        #         no_wearing_count_updated[cls] = True
        #         self.no_wearing_count[cls].append(not has) # 有穿的時候 append 0，沒穿的時候 append 1
        #         self.no_wearing_img[cls].append(cv2.resize(img,(640,360)))
        #         if len(self.no_wearing_count[cls]) > self.N_FRAMES[1]:
        #             self.no_wearing_count[cls] = self.no_wearing_count[cls][-self.N_FRAMES[1]:] # 只保留最近 N frame的結果

        # 此方法是會統整一個frame中 只要有其中一個yolo框沒穿裝備，那個裝備就會 count+1
            # 先按照rec.name 分群，例如 頭部一群、手部一群
        each_part_rectangles = dict()
        each_part_wearing_matrix = dict()
        for rec in rectangles:
            has_list = [bool(has) for has in rec.has] # 某一部位的各個裝備有無穿戴
            if rec.name not in each_part_rectangles:
                each_part_rectangles[rec.name] = [rec]                
                each_part_wearing_matrix[rec.name] = [has_list]
            else:
                each_part_rectangles[rec.name].append(rec)
                each_part_wearing_matrix[rec.name].append(has_list)

            # 計算每個部位，所有裝備的穿戴情況，只要有一個沒穿，就代表該frame沒有穿此裝備
        for part, matrix in each_part_wearing_matrix.items():
            matrix = np.array(matrix)
            is_wearing = matrix.sum(axis=0) == matrix.shape[0] # 每個部位有無穿戴

            classes = self.part_detect_classes[part]
            for cls, has in zip(classes, is_wearing):
                no_wearing_count_updated[cls] = True
                self.no_wearing_count[cls].append(not has) # 有穿的時候 append 0，沒穿的時候 append 1
                resize_img = cv2.resize(img,(480,270))
                self.no_wearing_img[cls].append(resize_img)
                if len(self.no_wearing_count[cls]) > self.N_FRAMES[1]:
                    self.no_wearing_count[cls] = self.no_wearing_count[cls][-self.N_FRAMES[1]:] # 只保留最近 N frame的結果
                    self.no_wearing_img[cls] = self.no_wearing_img[cls][-self.N_FRAMES[1]:] # 只保留最近 N frame的結果
                
        for cls, updated in no_wearing_count_updated.items():
            if not updated: # 代表當前這個 frame 沒有辨識到此裝備有無穿戴，所以就當作他有穿，不然會一直 alarm
                self.no_wearing_count[cls].append(False)
                resize_img = cv2.resize(img,(480,270))
                self.no_wearing_img[cls].append(resize_img)
                self.no_wearing_count[cls] = self.no_wearing_count[cls][-self.N_FRAMES[1]:] # 只保留最近 N frame的結果
                self.no_wearing_img[cls] = self.no_wearing_img[cls][-self.N_FRAMES[1]:] # 只保留最近 N frame的結果

        def daemon_run_notify(var):
            # 先儲存照片至地端
            save_img_list = []
            for img in var.notify_img:
                img_path = saver.save(img, ALARM_IMG_DIR)
                img_filename = img_path.split('\\')[-1]
                save_img_list.append(img_filename)
            msg_title, msg_text = "《V·Guard 穿戴偵測系統 警報通知》", \
                                "系統偵測到有人員未確實穿戴防護用具，請注意並落實安全保命條款！"
            if 'link' in self.wearingvar.alarm_channel[0]:
                Linkpost(camFrame=var.notify_img[1], msg_title=msg_title, msg_text=msg_text)
                print("Link 已傳送")
            if 'line' in self.wearingvar.alarm_channel[0]:
                line_notify_message_ndarray_img(msg_text, var.notify_img[1])
                logger.info('穿戴:LINE Notify MessageImg:' + msg_text)
                
            if 'mail' in self.wearingvar.alarm_channel[0]:
                mail_status = send_email_ndarray_img(self.wearingvar.alarm_channel[1], \
                        msg_title, msg_text, None, var.notify_img[1:])                
                logger.info('穿戴:E-mail 發送結果:'+mail_status)

            # var.mail_disable = True
            # 儲存alarm_log
            self.update_alarm_log(save_img_list)
            var.alarm_notify_running = False

        def run_notify_process(img):
            self.notify_img = img
            self.notify_thread = Thread(target=daemon_run_notify, args=(self, ), daemon=True, name="run_alarm_notify")
            self.notify_thread.start()

        no_wearing_alarm_mp3_filename = [] # 儲存沒穿戴的裝備 對應到的 警報的mp3檔的檔名
        alarm_flag=0    #暫時寫的燈號flag
        for wearing_name, count_list in self.no_wearing_count.items():
            count = sum(count_list)
            if count >= self.N_FRAMES[0]: # 達到要發警報的門檻
                # self.mp3maker.make_mp3(wearing_name)
                # alarm_mp3_filename = self.mp3maker.get_wearing_alarm_mp3_name(wearing_name)
                # no_wearing_alarm_mp3_filename.append(f'{alarm_mp3_filename}.mp3')
                # 語音離線版
                alarm_content = wearing_name + '沒有穿戴'
                no_wearing_alarm_mp3_filename.append(alarm_content)
                
                # if not self.relay_enable:
                #     self.relay_enable = True
                #     self.relay_control.relay_turnON()
                alarm_flag=1    #觸發燈號flag

                # if not self.alarm_disable:
                nowTime = time.time()
                if nowTime - self.previous_alarm_time > 30:
                    if not self.alarm_notify_running:
                        print("距離上次通報已經過30秒")
                        self.alarm_notify_running = True
                        self.previous_alarm_time = nowTime
                        # self.alarm_disable = True
                        run_notify_process(self.no_wearing_img[wearing_name])

                # if 'line' in self.wearingvar.alarm_channel[0] and not self.line_disable:
                #     # line_notify_message_ndarray_img(text, self.result_img)
                #     print('LINE 警告發報~~~~')
                #     self.line_disable = True
                
                # if 'mail' in self.wearingvar.alarm_channel[0] and not self.mail_disable:
                #      if not self.alarm_notify_running:
                #         self.alarm_notify_running = True
                #         run_notify_process(self.no_wearing_img[wearing_name])

                # self.previous_alarm_time = time.time()

        # #暫時寫的燈號觸發機制
        # if alarm_flag == 1:
        #     self.relay_control.relay_turnON()
        # else:
        #     self.relay_control.relay_turnOFF()
            
            
        if len(no_wearing_alarm_mp3_filename) > 0:
            if time.time()-self.play_start_time > self.wait_time:
                self.wait_time = 0
                self.play_start_time = 0

            if self.wait_time == 0:
                self.wait_time = self.wait_time_ratio * len(no_wearing_alarm_mp3_filename)
                self.play_start_time = time.time()
                alarm_thread = Thread(target=play_no_wearing_alarm,
                    args=(no_wearing_alarm_mp3_filename, ALARM_AUDIO_DIR,), daemon=True)
                alarm_thread.start()

    def update_alarm_log(self, img_list):
        now_time= time.strftime("%Y%m%d", time.localtime())
        item_json = { 
            'time':time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'channel': ','.join(self.wearingvar.alarm_channel[0]).upper(),
            'cam_id': self.globalvar.cam_id,
            # 'cam_id': "",
            'parts': ",".join(self.part_detect_classes.keys()),
            'classes': ','.join([val for val_list in self.part_detect_classes.values() for val in val_list]),
            'img': img_list
        }
        wr_json ={ 
            now_time: [item_json]
        }
        if os.path.exists(os.path.join(CONFIG_DIR, 'alarm_msg_log.json')):
            current_json = load_json(CONFIG_DIR, 'alarm_msg_log.json')
            i = 0
            if len(current_json) > self.wearingvar.alarm_channel[2]:
                n =len(current_json.copy())- self.log_save_day #要刪除的數量
                for key in current_json.copy().keys():
                    if(i < n):
                        for item in current_json[key]:
                            del_filelist= item['img']
                            for file in del_filelist:
                                os.remove(os.path.join(ALARM_IMG_DIR ,file))
                        del current_json[key]
                    else:
                        break
                    i+=1
            if current_json.get(now_time):
                current_json[now_time].append(item_json)
            else:
                current_json.update(wr_json)
            save_json(current_json, CONFIG_DIR, 'alarm_msg_log.json')
        else:
            save_json(wr_json, CONFIG_DIR, 'alarm_msg_log.json')
            
    def no_wearing_count_init(self):
        global detect_parts
        # 沒穿戴裝備的計數 要初始化
        self.no_wearing_count = dict()        
        for part in detect_parts:
            for cls in wearing_config.get_detect_class(part):
                self.no_wearing_count[cls] = [] # 儲存某裝備最近 N frame的配戴情況，如 [0,0,1,1,0]

    def no_wearing_img_init(self):
        global detect_parts
        # 沒穿戴裝備的計數 要初始化
        self.no_wearing_img = dict()        
        for part in detect_parts:
            for cls in wearing_config.get_detect_class(part):
                self.no_wearing_img[cls] = [] # 儲存某裝備最近 N frame的配戴情況，如 [0,0,1,1,0]
                # # img_dir = os.path.join(ALARM_SAVE_FRAME_DIR, cls)
                # if not os.path.exists(img_dir):
                #     os.mkdir(img_dir)

    def __no_wearing_count_updated_init(self):
        global detect_parts
        # 沒穿戴裝備的計數 要初始化
        no_wearing_count_updated = dict()
        for part in detect_parts:
            for cls in wearing_config.get_detect_class(part):
                no_wearing_count_updated[cls] = False # 儲存某裝備 當下這個 frame 有無辨識到
        return no_wearing_count_updated


def play_no_wearing_alarm(mp3_list, audio_dir):
    # from playsound import playsound
    try:
        for mp3 in mp3_list:
            # mp3_path = os.path.join(audio_dir, mp3)
            # playsound(mp3_path)
            # 語音離線版
            engine = pyttsx3.init()
            engine.say(mp3)
            engine.runAndWait()
            
    except Exception as e:
        logger.error(f'穿戴:播放沒有穿戴的音檔失敗, {e}')
        