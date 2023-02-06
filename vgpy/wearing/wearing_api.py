from time import time
from flask import Blueprint, request, render_template, jsonify, send_from_directory

from multiprocessing import Process
from datetime import datetime
import os
import fnmatch
import numpy as np
from collections import OrderedDict

from . import wearing_backend_torch as wbackend
from .config import WearingConfig
from . config import TMP_TRAIN_CHECK_DIR, TMP_TRAIN_DIR, CAMERA_COLLECT_YOLO_CONF
from ..global_object import StreamingQueues, GlobalVar
from ..global_function import remove_imagefile_extention

from enum import Enum, auto

from vgpy.utils.logger import create_logger
logger = create_logger()

current_dir = os.path.dirname(os.path.abspath(__file__))


# 穿戴辨識的 process 會用到的狀態等
class WearingVar:
    class SIGNAL(Enum):
        CHANGE_SETTING = auto() # 更改設定檔時，要更新 detector 中的 config等
        START_COLLECT = auto() # 開始收集照片
        END_COLLECT = auto() # 結束收集照片
        ALARM_CLEAR = auto() # 清除警報
        UPDATE_FENCE = auto() # 更新圍籬
        UPDATE_WEARINGVAR = auto() # 更新整個 WEARINGVAR進來 subprocess
        UPDATE_EFF_MODEL_CONF = auto() # 更新 新設定的各裝備的信心門檻值到 subprocess中的merge_predictor
    def __init__(self):
        """
        @attr        
            yolo_conf: yolo在偵測各個部位的門檻值，
                例如偵測頭部要超過多少，才認為有偵測到頭，
                要先偵測到部位，才有接著用 EfficientNet 辨識是什麼物品。

            

            click_camera_collect: 訓練頁面，Image Collection by Camera中按下 Start時變為 True
                此時 main.py中的 /stream 就會開始按照指定的頻率收集攝影機中的照片

            collect_data: 攝影機收集到的資料的檔名+路徑會放在這
            alarm_channel: 要警報的頻道與收件群組
            fence_setting: (是否開啟虛擬圍籬, 虛擬圍籬座標)
            train_args_setting: 模型訓練or優化相關參數設定
        """
        self.yolo_conf = None
        self.click_camera_collect = False
        self.collect_data = None
        self.alarm_channel = wbackend.get_alarm_setting()
        self.fence_setting = (wbackend.get_fence_enable(), wbackend.get_cam_areas())
        self.train_args_setting =wbackend.get_train_arg_dict()

### 穿戴 的 全域變數
from multiprocessing import Process, Queue
progress_queue = None
class WearingMain:
    def __init__(self, q:StreamingQueues, wearingvar:WearingVar, globalvar:GlobalVar) -> None:
        self.streaming_queues = q
        self.wearingvar = wearingvar
        self.globalvar = globalvar
        self.training_progress = None
        self.waiting_for_training = False

    def wearing_app_init(self):
        app_wearing = Blueprint('wearing', __name__)
        q = self.streaming_queues
        globalvar = self.globalvar
        wearingvar = self.wearingvar

        # 把稽核後的照片「剪下」到 稽核(check) 資料夾
        @app_wearing.route("/move2check", methods=['GET'])
        def move2check():
            try:
                filepath = request.args.get('filepath')
                return wbackend.move2check(filepath)
            except Exception as e:                
                logger.error(f'穿戴:檔案剪下 到 check 失敗, {e}')
                return '檔案剪下 到 check 失敗'
        
        # 把已稽核的照片 「剪下」 回 原本的資料夾，例如 從 "口罩_has/check" 回到 "口罩_has/"
        @app_wearing.route("/undo_move2check", methods=['GET'])
        def undo_move2check():
            try:
                filepath = request.args.get('filepath')
                return wbackend.undo_move2check(filepath)
            except Exception as e:
                logger.error(f'穿戴:檔案 從check 復原失敗, {e}')
                return '檔案 從check 復原失敗'
        
        # 刪除被拖出框外的照片
        @app_wearing.route("/remove_file/<path>", methods=['GET'])
        def remove_file(path):
            try:
                wbackend.remove_file(path)
                logger.info('穿戴: 拖曳刪除影像檔案')
                return '檔案刪除成功'
            except Exception as e:
                logger.error(f'穿戴:刪除被拖出框外的照片失敗, {e}')
                return '檔案刪除失敗'
        
        # # 預覽稽核照片使用
        # @app_wearing.route('/file_check/<string:filename>',methods=['GET'])
        # def get_model_file(filename):
        #     return send_from_directory(os.path.join(test_path), filename) #給定資料夾路徑

        #=================== 穿戴辨識頁面 (wear_identify.html) ===================#
        @app_wearing.route("/index",methods=['GET'])
        def wear_identify():
            logger.info('穿戴: 進入辨識頁')
            return render_template('wear_identify.html', cam_data=(globalvar.cam_id, int(globalvar.cam_num)))
        
        # @app_wearing.route("/index/alarm_response",methods=['GET','POST'])
        # def response_alarm_disable():
        #     yield "data: %s \n\n" % ("True")
        #     return Response(response_alarm_disable(), content_type='text/event-stream')

        @app_wearing.route("/alarm_clear",methods=['POST'])
        def alarm_clear():
            if request.method =='POST':
                q.c_queue.put(WearingVar.SIGNAL.ALARM_CLEAR)
                logger.info('穿戴: 手動清除異常警報')
                return '已清除警報！！'

        #=================== 穿戴設定頁面 (wear_setting.html) ===================#
        @app_wearing.route("/wear_setting",methods=['GET'])
        def wear_setting():
            #設定檔 資料架構如下 前面為index,後面以list 組資料
            set_dict = {
                1: wbackend.get_detect_part_list(),     #部位
                2: [wearingvar.fence_setting[0]],       #圍籬功能啟用/禁用
                3: [wbackend.get_sample_rate(), wbackend.get_max_quantity()],     #sample rate ＆ max quantity
                4: wearingvar.alarm_channel         #通報頻道 & Mail收件群組
            }
            fence_coord= wbackend.get_canvas_fence_area_str()   #圍籬區域座標
            mail_group = globalvar.mail_group   #Email 收件群組清單
            #辨識類別 資料架構如下 類別對應物品
            part_list = wbackend.PART_LIST
            class_names = wbackend.get_part_clslist_dict()
            class_confs = wbackend.get_part_confs_dict()
            part_detect_class = wbackend.get_part_detect_class()
            detect_class_checkbox_type = dict() # 記錄每個部位 的每個類別 要顯示哪種 checkbox，(有三種 checked, unchecked, disable)
            for part in part_list:
                detect_class_checkbox_type[part] = dict()

            # 判斷 設定頁 勾選要辨識那些部位的checkbox 能不能勾 及 預設有沒有打勾
            for part in part_list:
                detect_cls = part_detect_class[part]
                trained_model_list = wbackend.get_part_model_list(part)
                for cls in class_names[part]:
                    if cls not in trained_model_list: # 沒訓練過的模型
                        detect_class_checkbox_type[part][cls] = 'disable'
                    else:
                        if cls in detect_cls: # 訓練過的模型，且選擇 要辨識
                            detect_class_checkbox_type[part][cls] = 'checked'
                        else: # 訓練過的模型，且選擇 不辨識
                            detect_class_checkbox_type[part][cls] = 'unchecked'
            model_dict = wbackend.get_train_arg_dict()
            logger.info('穿戴: 進入設定頁')
            return render_template('wear_setting.html',set1=set_dict, part_list=part_list,
                            class_names=class_names, class_confs=class_confs,
                            detect_class_checkbox_type=detect_class_checkbox_type,
                            fence_set=fence_coord, mail_group_dict=mail_group, model_dict=model_dict)

        @app_wearing.route("/save_class_setting", methods=['POST'])
        def save_class_setting():
            try:
                data = request.get_json()
                ori_detect_part_list = wbackend.get_detect_part_list()
                ori_part_detect_class = wbackend.get_part_detect_class()
                part_detect_class = data['part_detect_class']
                part_confs = data['part_confs']
                for part in part_confs.keys():
                    detect_class = part_detect_class[part]
                    confs = list(map(int, part_confs[part]))
                    wbackend.save_class_setting(part, detect_class, confs)
                sample_rate = int(data['sample_rate'])
                max_quantity = int(data['max_quantity'])
                detect_part_list = data['detect_part_list']
                fence_enable = (str(data['fence_enable'])).lower()     #因應前端條件式判斷方便 改為小寫 
                alarm_channel_list = data['alarm_channel_list']
                wbackend.save_wearing_setting(detect_part_list, sample_rate, max_quantity, fence_enable, alarm_channel_list)
                wearingvar.fence_setting = (fence_enable, wbackend.get_cam_areas())
                wearingvar.alarm_channel = wbackend.get_alarm_setting()
                q.c_queue.put((WearingVar.SIGNAL.UPDATE_WEARINGVAR, wearingvar))

                q.c_queue.put(WearingVar.SIGNAL.UPDATE_EFF_MODEL_CONF)
                if(ori_detect_part_list != detect_part_list or ori_part_detect_class != part_detect_class):
                    #如果辨識部位或裝備有變更在更新載入模型
                    q.c_queue.put(WearingVar.SIGNAL.CHANGE_SETTING)
                    q.r_queue.get(block=True) # 等待 subprocess 設定改辨完成
                logger.info('穿戴設定值變更成功')
                return '成功儲存 穿戴偵測系統設定值！'
            except:
                logger.error('穿戴設定值變更失敗')
                return '儲存失敗 穿戴偵測系統設定值'


        @app_wearing.route("/save_train_setting", methods=['POST'])
        def save_train_setting():
            try:
                data = request.get_json()
                wbackend.save_train_args_setting(data)
                wearingvar.train_args_setting = data
                q.c_queue.put((WearingVar.SIGNAL.UPDATE_WEARINGVAR, wearingvar))
                logger.info('穿戴: 模型訓練參數設定值 變更成功')
                return '成功儲存 模型訓練參數設定值！'
            except:
                logger.error('穿戴: 模型訓練參數設定值 變更失敗')
                return '儲存失敗 模型訓練參數設定值！'

        @app_wearing.route("/save_fence_coord", methods=['POST'])
        def save_fence_coord():
            try:
                data = request.get_json()
                area_json = data['area_json']
                sample_rate = int(data['sample_rate'])
                max_quantity = int(data['max_quantity'])
                detect_part_list = data['detect_part_list']
                fence_enable = (str(data['fence_enable'])).lower()     #因應前端條件式判斷方便 改為小寫 
                alarm_channel_list = data['alarm_channel_list']
                wbackend.save_wearing_setting(detect_part_list, sample_rate, max_quantity, fence_enable, alarm_channel_list)
                wbackend.save_fence_coord(area_json, globalvar.cam_wh)
                wearingvar.fence_setting = (wbackend.get_fence_enable(), wbackend.get_cam_areas())
                q.c_queue.put((WearingVar.SIGNAL.UPDATE_WEARINGVAR, wearingvar))
                logger.info('穿戴: 虛擬圍籬設定成功')
                return '圍籬區域儲存成功!'
            except:
                logger.error('穿戴: 虛擬圍籬設定失敗')
                return '圍籬區域儲存失敗'

        @app_wearing.route("/api/class", methods=['POST', 'DELETE'])
        def api_class():
            try:
                if request.method == 'POST':
                    part_name = request.form.get('part_name')
                    zh_name = request.form.get('zh_name')
                    en_name = request.form.get('en_name')
                    wbackend.post_class(part_name, zh_name, en_name)
                    logger.info('穿戴: 成功新增穿戴類別')
                    return "成功新增穿戴類別"

                elif request.method == 'DELETE':
                    part_name = request.form.get('part_name')
                    zh_name = request.form.get('cls_zh_name')
                    wbackend.delete_class(part_name, zh_name)
                    logger.info('穿戴: 成功刪除穿戴類別')
                    return "成功刪除穿戴類別"
            except Exception as e:
                logger.error(f'穿戴:api_class failed, {e}')
                return 'api_class failed', 400


        #=================== 模型優化頁面 (wear_model_OPT.html) ===================#
        @app_wearing.route("/model_OPT",methods=['GET'])
        #模型優化頁
        def model_OPT():
            class_dict = wbackend.get_part_clslist_dict()
            logger.info('穿戴: 進入模型優化頁')
            return render_template('wear_model_OPT.html', class_data_list=class_dict)


        @app_wearing.route("/model_OPT/data_load",methods=['POST'])
        #資料載入 request 載入 有裝備 跟 無裝備 資料
        def model_opt_data_load():
            #取得 物品選取值 依物品得到該原始資料筆數 & 資料(依檔名串資料)    
            part = request.form.get('part')
            cls = request.form.get('class')

            o_data, x_data, check_o_data, check_x_data = wbackend.get_prediction_filenames(part, cls)
            #將check_data檔案移至原取樣資料
            for o_item in check_o_data:
                src = o_item[1]
                des_dir = os.path.split(os.path.split(o_item[1])[0])[0]
                filename = os.path.split(o_item[1])[1]
                des = os.path.join(des_dir, filename)
                os.rename(src, des)
            for x_item in check_x_data:
                src = x_item[1]
                des_dir = os.path.split(os.path.split(x_item[1])[0])[0]
                filename = os.path.split(x_item[1])[1]
                des = os.path.join(des_dir, filename)
                os.rename(src, des)
            o_data, x_data, check_o_data, check_x_data = wbackend.get_prediction_filenames(part, cls)

            dataNum_list = wbackend.get_ori_data_num(part, cls) # [有的數量, 無的數量]
            logger.info('穿戴: 前端要取有裝備、無裝備資料')
            return jsonify({
                "sync": "success",
                "og_dataNum": dataNum_list,     # 原始資料 有 跟 無 的照片張數
                "pred_data_o":o_data,           # 模型按照取樣頻率擷取的 預測「有穿戴」
                "pred_data_x":x_data,           # 模型按照取樣頻率擷取的 預測「無穿戴」
                "check_data_x":check_x_data,    # 使用者 稽核後 從 有穿戴->無穿戴
                "check_data_o":check_o_data     # 使用者 稽核後 從 無穿戴->有穿戴
            })

        @app_wearing.route("/model_OPT/remove_all_data",methods=['POST'])
        #刪除整批 有裝備 or 無裝備 資料
        def remove_all_data():
            data_json = request.get_json()
            file_list = data_json['files']
            file_path_str = data_json['file_path'] #array: selected_part, selected_class, has_no_flag, check_or_not
            file_dir = os.path.join(current_dir,'img','prediction',file_path_str[0],file_path_str[1]+'_'+file_path_str[2],file_path_str[3])
            for f in file_list:
                file = os.path.join(file_dir, f+'.png')
                wbackend.remove_file(file)
            logger.info('穿戴: 刪除整批有裝備／無裝備資料')
            return '已刪除所有檔案！'
        

        #=== 模型優化 及 取得模型優化的進度 ===#
        @app_wearing.route("/model_optimize", methods=['POST'])
        def model_optimize():
            global progress_queue
            data = request.form
            part = data['part']
            cls = data['class']
            epoch = wearingvar.train_args_setting['opt_epoch']
            batch_size = wearingvar.train_args_setting['opt_batch_size']
            min_lr = wearingvar.train_args_setting['min_lr']
            resize_shape = tuple(wearingvar.train_args_setting['resize_shape'])

            if globalvar.current_process is not None:
                globalvar.current_process.terminate() # 先關閉辨識的 process才訓練 不然記憶體會爆
                
            self.training_progress = None # 訓練進度歸零
            self.waiting_for_training = True # 等待模型開始訓練
            progress_queue = Queue()
            process = Process(target=wbackend.opt_model_sub_process, args=(part, cls, progress_queue, epoch, batch_size, min_lr, resize_shape,), daemon=True)
            # process = Thread(target=wapi.opt_model_sub_process, args=(part, cls, progress_queue,))
            process.start()
            process.join()
            globalvar.current_process = None
            globalvar.current_process = self.app_wearing_start() # 訓練完成後 重新開啟辨識的 process
            logger.info('穿戴: 優化模型執行緒 啟動')
            return '成功優化模型'

        @app_wearing.route("/model_optimize_progress", methods=['GET'])
        def model_optimize_progress():
            global progress_queue
            p = None
            try:        
                p = progress_queue.get(timeout=0.7) # 額外用一個 progress queue 來跟 訓練的Process 溝通，才能將訓練進度顯示到網頁上
            except Exception as e:
                # 若已經開始訓練 就要印出錯誤
                if not self.waiting_for_training:
                    logger.error(f'穿戴:優化模型取得進度錯誤, {e}')
                    # print('get queue failed')
            
            if p is not None: # 有取得新的進度，才更新進度的變數
                self.training_progress = p
                self.waiting_for_training = False # 取的到進度 代表模型已經開始訓練了

            if self.training_progress is None:
                return 'data loading'
            else:
                return str(self.training_progress)      


        #=================== 模型訓練頁面 (wear_model_train.html) ===================#
        @app_wearing.route("/model_train", methods=['GET','POST'])
        #模型訓練頁
        def model_train():
            #辨識類別 資料架構如下 類別對應物品
            class_dict = wbackend.get_part_clslist_dict()
            # 載入時 清空 tmp_training_check_data 的資料夾
            wbackend.delete_tmp_train_check_dir()
            # 載入時 清空 tmp_train_dir 的資料夾
            wbackend.delete_tmp_train_dir()
            logger.info('進入模型訓練頁')
            return render_template('wear_model_train.html', class_data_list=class_dict)

        #=== 模型訓練 及 取得模型訓練的進度 ===#
        @app_wearing.route("/model_train/train_model", methods=['POST'])
        def train_model():
            global progress_queue
            data = request.form
            part = data['part']
            cls = data['class']
            equd = data['equd']  # 是否有"有裝備"的dataset
            unequd = data['unequd'] # 是否有"無裝備"的dataset
            epoch = wearingvar.train_args_setting['train_epoch']
            batch_size = wearingvar.train_args_setting['train_batch_size']
            min_lr = wearingvar.train_args_setting['min_lr']
            resize_shape = tuple(wearingvar.train_args_setting['resize_shape'])
            if globalvar.current_process is not None:
                globalvar.current_process.terminate() # 先關閉辨識的 process才訓練 不然記憶體會爆
                
            self.training_progress = None # 訓練進度歸零
            self.waiting_for_training = True # 等待模型開始訓練
            progress_queue = Queue()
            process = Process(target=wbackend.train_model_sub_process, args=(part, cls, equd, unequd, progress_queue, epoch, batch_size, min_lr, resize_shape,), daemon=True)
            process.start()
            process.join()
            logger.info('模型訓練執行緒 啟動')
            # globalvar.current_process = self.app_wearing_start() # 訓練完成後 重新開啟辨識的 process
            return '成功訓練模型'
    
        @app_wearing.route("/model_train/reload_detect_process", methods=['POST'])
        def reload_detect_process():
            # 模型訓練完成後，前端呼叫此api，重新載入辨識 subprocess
            globalvar.current_process = None
            globalvar.current_process = self.app_wearing_start() # 訓練完成後 重新開啟辨識的 process
            logger.info('穿戴: 重新載入辨識 subprocess')
            return '載入成功'

        @app_wearing.route("/model_train_progress", methods=['GET'])
        def model_train_progress():
            global progress_queue
            p = None
            try:                
                while not progress_queue.empty():
                    p = progress_queue.get(timeout=0.7) # 額外用一個 progress queue 來跟 訓練的Process 溝通，才能將訓練進度顯示到網頁上
            except Exception as e:
                # 若已經開始訓練 就要印出錯誤
                if not self.waiting_for_training:
                    logger.error(f'穿戴:訓練模型取得進度錯誤, {e}')
                    # print('get queue failed')
            
            if p is not None: # 有取得新的進度，才更新進度的變數
                self.training_progress = p
                self.waiting_for_training = False # 取的到進度 代表模型已經開始訓練了

            if self.training_progress is None:
                return 'data loading'
            else:
                return str(self.training_progress)

        @app_wearing.route("/get_train_data_dir", methods=['POST'])
        def get_train_data_dir():
            return jsonify({
                "check_dir": TMP_TRAIN_CHECK_DIR,
                "train_dir": TMP_TRAIN_DIR,
            })
            
        @app_wearing.route("/model_train/og_data_num", methods=['POST'])
        #資料載入 request 載入 有裝備 跟 無裝備 資料
        def model_train_data_num():
            #取得 物品選取值 依物品得到該原始資料筆數 & 資料(依檔名串資料)    
            part = request.form.get('part')
            cls = request.form.get('class')
            # o_data, x_data, check_o_data, check_x_data = wapi.get_prediction_filenames(part, cls)
                
            dataNum_list = wbackend.get_ori_data_num(part, cls) # [有的數量, 無的數量]
            logger.info('穿戴: 前端要取有裝備、無裝備的照片張數')
            return jsonify({
                "og_dataNum": dataNum_list,     # 原始資料 有 跟 無 的照片張數
            })
        
        @app_wearing.route("/model_train/og_data_load", methods=['POST'])
        #資料載入 request 載入 有裝備 跟 無裝備 資料
        def model_train_data_load():
            #取得 物品選取值 依物品得到該原始資料筆數 & 資料(依檔名串資料)    
            part = request.form.get('part')
            cls = request.form.get('class')
            
            wbackend.move_og_data_to_tmp_training_check(part, cls) # 把原始資料複製到 tmp_training_check
            o_data, x_data = wbackend.get_tmp_train_check_filenames() # 取得 tmp_training_check 中的照片路徑 
            logger.info('穿戴: 前端要取有裝備、無裝備的原始訓練資料')
            return jsonify({
                "sync": "success",
                "og_data_o":o_data,           # 原始訓練資料中「有穿戴」
                "og_data_x":x_data,           # 原始訓練資料中「無穿戴」
            })

        @app_wearing.route("/model_train/camera_data_load", methods=['POST'])
        #資料載入 request 載入 有裝備 跟 無裝備 資料
        def camera_data_load():
            part = request.form.get('part')
            frame = request.form.get('frame')
            type = request.form.get('type')
            fence_enable = request.form.get('fence_enable')
            if type == '有':
                type = 'has'
            elif type == '無':
                type = 'no'
            
            # if globalvar.current_process is not None:
            #     globalvar.current_process.terminate() # 先關閉辨識的 process，因為用攝影機抓照片，要開啟另一個 yolo 的process

            # e = self.collect_img_start(part, frame, type)
            # self.wearingvar.click_camera_collect = True
            # p.join()

            # globalvar.current_process = self.app_wearing_start() # 訓練完成後 重新開啟辨識的 process

            self.collect_img_start(part, frame, type, fence_enable)
            self.wearingvar.click_camera_collect = True
            self.wearingvar.fence_setting =(fence_enable, self.wearingvar.fence_setting[1])
            
            import time
            while self.wearingvar.click_camera_collect: # 等到照片收集完成後，程式才會繼續往下執行
                time.sleep(0.1)
            logger.info('穿戴: 前端要取有裝備、無裝備的收集資料')
            return jsonify({
                "collect_data": self.wearingvar.collect_data              
            })

        @app_wearing.route("/model_train/stop_camera_data_load", methods=['POST'])
        #資料載入 request 載入 有裝備 跟 無裝備 資料
        def stop_camera_data_load():            
            q.c_queue.put(WearingVar.SIGNAL.END_COLLECT)
            self.wearingvar.click_camera_collect = False
            logger.info('穿戴: 停止收集資料')
            return '收集完成'


        @app_wearing.route("/model_train/delete_data/<has_or_no>", methods=['POST'])
        def model_train_data_delete(has_or_no):
            if has_or_no == 'has':
                wbackend.delete_tmp_train_check_dir_has()
            elif has_or_no == 'no':
                wbackend.delete_tmp_train_check_dir_no()
            elif has_or_no == 'final_has':
                wbackend.delete_tmp_train_dir_has()
            elif has_or_no == 'final_no':
                wbackend.delete_tmp_train_dir_no()
            elif has_or_no == 'all':
                wbackend.delete_tmp_train_check_dir_has()
                wbackend.delete_tmp_train_check_dir_no()
                wbackend.delete_tmp_train_dir_has()
                wbackend.delete_tmp_train_dir_no()
            logger.info('穿戴: 成功刪除資料')
            return '成功刪除資料'

        
        @app_wearing.route("/model_train/final_data_load/<has_or_no>", methods=['POST'])
        #資料載入 request 載入 有裝備 跟 無裝備 資料
        def model_train_final_data_load(has_or_no):
            wbackend.move_tmp_training_check_to_tmp_check(has_or_no)
            data = wbackend.get_train_check_filenames(has_or_no)
            logger.info('穿戴: 前端要取 Total Train Data 的資料')
            return jsonify({
                "sync": "success",
                "final_data": data,           # 要顯示到 Total Train Data 的資料                
            })                         

        #=================== 異常訊息頁面 (wear_log_review.html) ===================#
        @app_wearing.route("/log_review", methods=['GET','POST'])
        #異常訊息頁
        def log_review():
            alarm_log_json = wbackend.get_alarm_log()
            alarm_log_json = OrderedDict(sorted(alarm_log_json.items(), key= lambda x: x[0], reverse=True))
            if request.method =='POST':
                return alarm_log_json
            logger.info('進入異常訊息頁')
            return render_template('wear_log_review.html', alarm_log=alarm_log_json)

        @app_wearing.route('/show_img/<string:target_dir>/<string:filename>', methods=['GET'])
        def show_photo(target_dir, filename):
            return wbackend.return_image_to_web(target_dir,filename)

        # remove anomal data whice user choose
        @app_wearing.route('/remove_day_all_pic/<date>', methods=['POST'])
        def remove_day_all_pic(date):
            print(date)
            jsonFileOfAlarmLog = WearingConfig.get_alarm_log()
            try:
                del jsonFileOfAlarmLog[date]
            except:
                logger.error("wearing api: remove_day_all_pic, error event: cannot find date")
            dirPathForAlarmMsg = os.path.join(os.getcwd(), 'vgpy/wearing/img/alarm_img')
            for dirPath, dirNames, fileNames in os.walk(dirPathForAlarmMsg):
                for file in fileNames:
                    if(fnmatch.fnmatch(file, f'{date}*.jpg')):
                        os.remove(os.path.join(dirPath, file))
            WearingConfig.save_alarm_log(jsonFileOfAlarmLog)
            return date

        #======================= 控制 辨識的 Process ============================#
        @app_wearing.route("/control_detect_process/<action>", methods=['GET'])
        def control_detect_process(action):
            # 穿戴辨識 頁面的「Start」按鈕
            if action == 'click_start':
                if not globalvar.click_start:
                    q.c_queue.put('tmp') # 輸入一個暫存的訊號， 讓 /stream 先不要送圖片進來 (以防有其他 control跑到一半，但c_queue已經空了)
                    globalvar.click_start = True                    
                    logger.info('開始辨識')

            # 穿戴辨識 頁面的「Cancel」按鈕
            elif action == 'click_cancel':
                globalvar.click_start = False
                logger.info('取消辨識')
                return '已取消辨識'


            return '控制 辨識process 成功!'

        return app_wearing

    def app_wearing_start(self):
        current_process = Process(
            target=wearing_detect_init,
                args=(self.streaming_queues, self.wearingvar, self.globalvar), daemon=True)
        # current_process = Process(
        #     target=wearing_detect_init,
        #         args=(self.streaming_queues, self.wearingvar), daemon=True)
        current_process.start()
        logger.info('穿戴: subprocess啟動')
        self.streaming_queues.c_queue.get() # 卡在這等 subprocess 內的模型讀取完成
        return current_process

    def collect_img_start(self, which_part, frame, type, fence_enable):
        logger.info('穿戴: 開始使用攝影機收集訓練照片')
        save_dir = os.path.join(TMP_TRAIN_CHECK_DIR, type)
        parms ={
            'yolo_conf': CAMERA_COLLECT_YOLO_CONF,
            'which_part': which_part,
            'save_dir': save_dir,
            'frame': frame,
            "fence_enable": fence_enable
        }
        self.streaming_queues.c_queue.put(self.wearingvar.SIGNAL.START_COLLECT)
        self.streaming_queues.c_queue.put(parms)


# def wearing_detect_init(q:StreamingQueues, var:WearingVar, gvar:GlobalVar):
def wearing_detect_init(q:StreamingQueues, var:WearingVar, gvar=None):
    from .wearing_detector_torch import WearingDetector
    from ..utils.img_utils import save_img_in_subprocess
    from datetime import datetime
    from enum import Enum, auto
    import cv2

    # logger = get_logger()

    # from vgpy.config.log.log_config import get_logger, LOG_NAME, LOG_PATH
    from vgpy.utils.logger import create_logger
    logger = create_logger()
    
    class MODE(Enum):
        DETECT_MODE = auto() # 辨識模式
        COLLECT_MODE = auto() # 收集照片模式
    mode = MODE.DETECT_MODE

    # q.c_queue.put('wearing_init') # 初始化中，透過這個讓攝影機的畫面在模型載入前，能先顯示原始畫面
    wearing_detector = WearingDetector(var, gvar)
    
    # 先用測試圖片觸發一次辨識，讓模型載入，改善初始延遲問題
    # test_img = np.zeros((1080,1920,3))    #本來想用全黑的圖片來跑，無效!
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # test_img = cv2.imread(os.path.join(current_dir, 'config','first_img.jpg'))
    # wearing_detector.detect_img(test_img, yolo_conf=var.yolo_conf, first_flag=True)
    # wearing_detector.yolo_detector_init(yolo_conf=var.yolo_conf)

    
    
    # q.c_queue.get() # 把前面 put的訊號釋放掉，代表模型已載入完成，攝影機會能開始將照片輸入進來做辨識了
    q.c_queue.put('init done')
    q.c_queue.task_done()
    logger.info('穿戴: yolo_detector初始化完成')

    # 收集照片用的變數初始化
    count, yolo_conf, which_part, save_dir, frame, files  = None, None, None, None, None, None        
    while True:

        if not q.c_queue.empty():
            signal = q.c_queue.get(block=False)
            
            if type(signal) is tuple:
                signal, value = signal
                if signal is WearingVar.SIGNAL.UPDATE_WEARINGVAR:
                    new_wearingvar = value
                    var.alarm_channel = new_wearingvar.alarm_channel
                    var.fence_setting = new_wearingvar.fence_setting
                    var.train_args_setting = new_wearingvar.train_args_setting

            try:
                logger.info(f'穿戴: 收到[{signal.name}]的signal')
            except:
                pass
            
            if signal is WearingVar.SIGNAL.CHANGE_SETTING: # 若是有從 設定頁面 更新設定檔，就要重讀 model                
                wearing_detector.init()
                q.r_queue.put(True) # 更改設定完成，讓api那邊卡住的地方繼續執行，前端才會收到回應
            elif signal is WearingVar.SIGNAL.UPDATE_EFF_MODEL_CONF:
                wearing_detector.update_config()
                wearing_detector.update_merge_predictor_confs()
            elif signal is WearingVar.SIGNAL.ALARM_CLEAR:
                # wearing_detector.line_disable , wearing_detector.mail_disable = False, False
                wearing_detector.alarm_disable= False
            elif signal is GlobalVar.SIGNAL.END_PROCESS:
                # 流程結束
                break
            elif signal is WearingVar.SIGNAL.START_COLLECT:
                mode = MODE.COLLECT_MODE
                count = 0
                parms = q.c_queue.get()
                yolo_conf = parms['yolo_conf']
                which_part = parms['which_part']
                save_dir = parms['save_dir']
                frame = int(parms['frame'])
                var.fence_setting = (parms['fence_enable'], var.fence_setting[1])
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                files = []

                wearing_detector.add_collect_part(which_part) # 會判斷是否需要讓YOLO模型 加入此次要收集照片的部位                

            elif signal is WearingVar.SIGNAL.END_COLLECT:
                wearing_detector.remove_collect_part() # 會判斷是有強行加入要辨識的部位到YOLO，有的話要刪除
                wearing_detector.init()  #重新載入設定的身體部位
                mode = MODE.DETECT_MODE
            
            elif signal is GlobalVar.SIGNAL.CHANGE_CAM_ID:
                new_camid, new_camwh = q.c_queue.get(block=True, timeout=5)
                gvar.cam_id = new_camid
                gvar.cam_wh = new_camwh
                
            q.c_queue.task_done()
            continue

        if not q.d_queue.empty():
            if mode == MODE.DETECT_MODE: # 這是辨識模式下 subprocess 要處理的部分
                startTime = time()
                img = q.d_queue.get()
                
                result_img, im0 = wearing_detector.detect_img(img, yolo_conf=var.yolo_conf)
                if result_img is None:
                    result_img = im0

                q.r_queue.put(result_img)
                print(f"Time: {time()-startTime}")
            elif mode == MODE.COLLECT_MODE: # 這是收集照片模式下 subprocess 要處理的部分
                img = q.d_queue.get()
                
                if count % frame == 0:
                    result_imgs, img_with_box = wearing_detector.detect_img_part(img, which_part, yolo_conf)
                    
                    for i, cut_img in enumerate(result_imgs):
                        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-4]
                        img_name = f'cam_{datetime_str}_{i}.png'
                        path = os.path.join(save_dir, img_name)                
                        save_img_in_subprocess(path, cut_img)

                        files.append(
                            (remove_imagefile_extention(img_name), path)
                        )
                
                result_dict = {
                    'files': files,
                    'img_with_box': img_with_box,
                }
                q.r_queue.put(result_dict)
                count += 1


    q.clear_and_done_control_queue()


if __name__ == '__main__':
    pass
