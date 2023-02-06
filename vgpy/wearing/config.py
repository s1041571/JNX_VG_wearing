import os
import math
import json
from ..utils.json_utils import save_json, load_json

# object 版本
from typing import  Dict, List
from vgpy.utils.logger import create_logger
logger = create_logger()

current_dir = os.path.dirname(os.path.abspath(__file__))
PART_LIST = ['head', 'hand', 'body', 'foot']
CONFIG_DIR = 'vgpy/wearing/config'
CONFIG_FILENAME = 'wearing_config.json'
IMG_DIR = os.path.join(current_dir, 'img')
PRED_SAVE_DIR_HEAD = os.path.join(current_dir, 'img', 'prediction')
ORI_TRAIN_DIR_HEAD = os.path.join(current_dir, 'img', 'original_training_data')
TMP_TRAIN_CHECK_DIR = os.path.join(current_dir, 'img', 'tmp_training_check_data')
TMP_TRAIN_DIR = os.path.join(current_dir, 'img', 'tmp_training_data')
TRAIN_RESULT_DIR = os.path.join(current_dir, 'train_result')
EFFI_MODEL_DIR = os.path.join(current_dir, 'model')
SHUFFLE_NET_MODEL_DIR = os.path.join(current_dir, 'shuffleNet_v2_model')
ALARM_AUDIO_DIR = os.path.join(current_dir, 'audio')
ALARM_IMG_DIR = os.path.join(current_dir, 'img','alarm_img')
CAMERA_COLLECT_YOLO_CONF = 0.6
MASK_COORD_TXT = os.path.join(CONFIG_DIR,'fence_mask_coord.txt')


class DetectSetting:
    def __init__(self, name, en, zh, conf_thres, detect_class):
        self.name: str = name
        self.en: List = en
        self.zh: List = zh
        self.conf_thres: List = conf_thres
        self.detect_class: List = detect_class

class SamplingSetting:
    def __init__(self, sampling_rate, max_quantity):
        self.sampling_rate = sampling_rate
        self.max_quantity = max_quantity

class AlarmChannelSetting:
    def __init__(self, channel, mail_group, log_save_day):
        self.channel = channel
        self.mail_group = mail_group
        self.log_save_day = log_save_day

class WearingConfig:
    def __init__(self, filedir=CONFIG_DIR, filename=CONFIG_FILENAME) -> None:
        self.detect_part = []
        self.detect_setting :Dict[str, DetectSetting] = dict()
        self.sampling_setting :SamplingSetting = None
        self.fence_enable = None
        self.alarm_setting :AlarmChannelSetting = None
        self.train_args_setting = None
        self.filedir, self.filename = None, None
        if filedir is not None and filename is not None:
            self.load_config(filedir, filename)
        else:
            self.load_config()

        
    def load_config(self, filedir=CONFIG_DIR, filename=CONFIG_FILENAME):
        self.filedir = filedir
        self.filename = filename

        config = load_json(filedir, filename)
        self.detect_part = config['detect_part']
        self.detect_setting:Dict[str, DetectSetting] = dict()
        for name, setting in config['detect_setting'].items():
            detect_setting = DetectSetting(
                name = name,
                en = setting['en'],
                zh = setting['zh'],
                conf_thres = setting['conf_thres'],
                detect_class = setting['detect_class'],
            )
            self.detect_setting[name] = detect_setting
            
        self.sampling_setting = SamplingSetting(
            sampling_rate = config['sampling_setting']['sampling_rate'],
            max_quantity = config['sampling_setting']['max_quantity']
        )
        self.fence_enable = config['fence_enable']
        self.alarm_setting = AlarmChannelSetting(
            channel = config['alarm_setting']['channel'],
            mail_group = config['alarm_setting']['mail_group'],
            log_save_day = config['alarm_setting']['log_save_day']
        )
        self.train_args_setting = config['train_args_setting']

    def save_config(self, filedir=None, filename=None):
        if filedir is None or filename is None:
            filedir = self.filedir
            filename = self.filename

        json_config = dict()
        json_config['detect_part'] = self.detect_part
        json_config['detect_setting'] = dict()
                
        for name, setting in self.detect_setting.items():
            json_setting = {
                'en': setting.en,
                'zh': setting.zh,
                'conf_thres': setting.conf_thres,
                'detect_class': setting.detect_class,
            }
            json_config['detect_setting'][name] = json_setting
        
        json_config['fence_enable'] = self.fence_enable
        json_config['sampling_setting'] = {
            'sampling_rate': self.sampling_setting.sampling_rate,
            'max_quantity': self.sampling_setting.max_quantity
        }

        json_config['alarm_setting'] = {
            'channel': self.alarm_setting.channel,
            'mail_group': self.alarm_setting.mail_group,
            'log_save_day': self.alarm_setting.log_save_day
        }
        json_config['train_args_setting'] = self.train_args_setting
        save_json(json_config, filedir, filename)

    def get_part_setting(self, part) -> DetectSetting:
        return self.detect_setting.get(part)

    def get_detect_part(self):
        return self.detect_part

    def get_part_wearing_en(self, part):
        en_part_list = self.get_part_setting(part).en
        return en_part_list

    def get_part_wearing_zh(self, part):
        zh_part_list = self.get_part_setting(part).zh
        return zh_part_list

    def get_fence_enable(self):
        return self.fence_enable

    def get_sampling_rate(self):
        return self.sampling_setting.sampling_rate

    def get_max_quantity(self):
        return self.sampling_setting.max_quantity

    def get_alarm_setting(self):
        return self.alarm_setting

    def get_train_arg_dict(self):
        return self.train_args_setting

    def get_part_wearing_confs(self, part):
        return self.get_part_setting(part).conf_thres

    def get_part_wearing_confs_in_detect_class(self, part):
        detect_class = self.get_detect_class(part)
        confs = self.get_part_wearing_confs(part)
        detect_confs = []
        for cls in detect_class:
            idx = self.get_part_wearing_zh(part).index(cls)
            detect_confs.append(confs[idx])
        return detect_confs

    def get_detect_class(self, part):
        return self.get_part_setting(part).detect_class

    def add_class(self, part, zh_name, en_name):
        self.get_part_setting(part).zh.append(zh_name)
        self.get_part_setting(part).en.append(en_name)        
        self.get_part_setting(part).conf_thres.append(0.5)
        self.save_config()

    def delete_class(self, part, zh_name):
        setting = self.get_part_setting(part)
        delete_idx = setting.zh.index(zh_name)
        del setting.zh[delete_idx]
        del setting.en[delete_idx]
        del setting.conf_thres[delete_idx]

        for idx, cls_name in enumerate(setting.detect_class):
            if cls_name == zh_name:
                del setting.detect_class[idx]
                break
        
        self.save_config()

    def save_class_setting(self, part, detect_class, confs):
        self.get_part_setting(part).detect_class = detect_class
        
        for i, conf in enumerate(confs):
            if conf > 1:
                conf = round(conf/100, 2)
            confs[i] = conf
            
        self.get_part_setting(part).conf_thres = confs
        self.save_config()
    
    def save_train_args_setting(self, dict):
        self.train_args_setting = dict
        self.save_config()

    def save_wearing_setting(self, detect_parts, sample_rate, max_quantity, fence_enable, alarm_channel_list):
        self.detect_part = detect_parts
        self.sampling_setting.sampling_rate = sample_rate
        self.sampling_setting.max_quantity = max_quantity
        self.fence_enable = fence_enable     
        self.alarm_setting.channel = alarm_channel_list['channel']
        self.alarm_setting.mail_group = alarm_channel_list['mail_group']
        self.alarm_setting.log_save_day = alarm_channel_list['log_save_day']
        self.save_config()

    def get_cam_areas(self):
        # 取得 攝影機 的解析度之下的圍籬的座標
        maskArr = []
        with open(MASK_COORD_TXT, 'r') as f:
            a = f.read()
            linesArr = a.split('/')
            for lines in linesArr:
                line = lines.split('\n')
                maskCoordArr=[]
                for l in line:
                    if l !='':
                        l=l.replace(' ', '')
                        maskCoordArr.append(tuple(map(int, l.split(','))))
                if maskCoordArr:
                    maskArr.append(maskCoordArr)
        return maskArr

    def from_canvas_to_camera(self, canvas_areas_coords, canvas_wh, cam_wh):
        # 把網頁中 canvas設置的點座標，轉換成相機解析度的座標
        canvas_w, canvas_h = canvas_wh
        cam_w, cam_h = cam_wh
        cam_areas = []
        for points in canvas_areas_coords:
            tmp_area_points = []
            for point in points:
                x,y = point          
                x = math.floor(int(x)/(canvas_w/cam_w))
                y = math.floor(int(y)/(canvas_h/cam_h))
                tmp_area_points.append((x, y))
            cam_areas.append(tmp_area_points)

        return cam_areas

    def get_canvas_fence_area_str(self):
        # 取得 網頁canvas 的解析度之下的圍籬的座標  (為字串，格式如下)
        # "{202,62 153,330 147,379 452,474 595,78 },{...},..."
        f_path = os.path.join(CONFIG_DIR,'fence_area_coord.json')
        with open(f_path, 'r', encoding='utf-8') as a:
            a_coord = json.load(a)
        areas_str = ''
        for points in a_coord['coord']:
            points_str = ''
            for point in points:
                points_str += ','.join(point) + ' '
            areas_str += '{%s},' % points_str    
        a_coord['coord'] = areas_str
        return a_coord


    def save_fence_areas_ponits(self, canvas_data, cam_areas):
        # 儲存 canvas 解析度的圍籬座標
        data = {
            "coord": canvas_data[0],
            "img_width": int(canvas_data[1]),
            "img_height": int(canvas_data[2])
        }
        save_json(data, CONFIG_DIR, 'fence_area_coord.json')
        
        # 儲存 攝影機 解析度的圍籬座標
        a = ''
        for points in cam_areas:
            tmp_area_points = []
            for point in points:
                x,y = point
                a += f"{x},{y}\n"
                tmp_area_points.append((x, y))
            a += "/"

        with open(MASK_COORD_TXT, 'w') as f:
            f.write(a)
    
    # 取得異常發報訊息log
    @staticmethod
    def get_alarm_log():
        try:
            log_json = load_json(CONFIG_DIR, 'alarm_msg_log.json')
        except Exception as e:            
            logger.error(f'穿戴:取得異常發報訊息log錯誤, {e}')
            log_json = {}
        finally:
            return log_json

    # 儲存異常發報訊息log
    @staticmethod
    def save_alarm_log(data):
        try:
            save_json(data, CONFIG_DIR, 'alarm_msg_log.json')
        except Exception as e:            
            logger.error(f'穿戴:儲存異常發報訊息log錯誤, {e}')

# CONFIG_FILENAME = 'sample.json'

if __name__ == '__main__':
    # save_json(config, CONFIG_DIR, CONFIG_FILENAME)
    # config = load_json(CONFIG_DIR, CONFIG_FILENAME)

    config = WearingConfig(CONFIG_DIR, CONFIG_FILENAME)
    config.save_config(CONFIG_DIR, 'test')

    print('')

