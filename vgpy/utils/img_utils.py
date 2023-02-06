from os import listdir
import cv2
from tqdm import tqdm

from PIL import ImageFont, ImageDraw, Image
import numpy as np
from ..yolov5.utils.plots import plot_one_box
import re
import os
import emoji
from vgpy.utils.logger import create_logger
logger = create_logger()

def save_img_in_subprocess(path, img, isBGR=True):
    # 可以解決 cv2.imwrite 沒辦法在 subprocess中 儲存圖片的問題
    # 傳入的 img 預設為 cv2的img，也就是 BGR的格式
    if isBGR:
        img = img[:,:,::-1] # 轉成 RGB
    img = Image.fromarray(img)
    img.save(path)
  

def cv2_save_zh_img(path, img, file_extension='.png'):
    cv2.imencode(file_extension, img)[1].tofile(path)

def cv2_read_zh_img(path):
    # cv2 imread 不能讀取中文路徑
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img 


def transform_xyxy2xywh(h,w,xmin,ymin,xmax,ymax,class_id):
    yolo_xc = ((xmin+xmax)/2)/w
    yolo_yc = ((ymin+ymax)/2)/h
    yolo_w = (xmax-xmin)/w
    yolo_h = (ymax-ymin)/h
    yolo_format = f"{class_id} {yolo_xc} {yolo_yc} {yolo_w} {yolo_h}"
    return yolo_format

def get_cut_img_from_im0(im0, xyxy, resize=None):
    xyxy_int = [int(v) for v in xyxy]
    part_img = im0[xyxy_int[1]:xyxy_int[3], xyxy_int[0]:xyxy_int[2]]
    if resize != None:
        part_img = cv2.resize(part_img, resize, interpolation=cv2.INTER_AREA)
    
    return part_img


def put_zh_text_opencv(im, text, pos, color, fsize=50, emoji_name=None):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    img_TS = Image.fromarray(im)
    font = ImageFont.truetype('font/msjhbd.ttc', fsize)
    em_font = ImageFont.truetype("font/seguisym.ttf",fsize)
    fillColor = color #(255,0,0)
    position = pos #(100,100)
    position2 = pos[0]+65,pos[1]
    # if not isinstance(text,unicode):
    # text = text.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    emojis = emoji.emojize(emoji_name) if emoji_name else ''
    draw.text(position, emojis, font=em_font, fill=fillColor)
    draw.text(position2, text, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img



def plot_yolo_box_to_yolo_rectangle(rect, img, color, line_thickness=2, zh=False, label=None):
    # 如果 zh(中文) = True 的話， 回傳值需要被接收，例如 img = plot_yolo_box_to_yolo_rectangle(...)
    xyxy=rect.xyxy
    if label is None:
        label = f'{rect.name} {rect.yolo_conf:.2f}'
    if not zh:
        plot_one_box(xyxy, img, label=label, color=color, line_thickness=line_thickness)
    else:
        img = plot_one_box_zh_text(xyxy, img, label=label, color=color, line_thickness=line_thickness)
        return img

import random
def plot_one_box_zh_text(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    text_color = (225, 255, 255)
    font_size = 40
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        zh_count = count_zh_word(label)
        nonzh_count = len(label)-zh_count
        t_size = (
            font_size*zh_count + (font_size*nonzh_count)//2,
            font_size
        )
        c2 = c1[0] + t_size[0], c1[1] + int(t_size[1]*0.95)
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        img = put_zh_text_opencv(img, label, c1,
            color=text_color, fsize=int(font_size*0.85) )
        return img

def count_zh_word(text):
    regex = re.compile("[\u4E00-\u9FFF]+")
    zh_words = regex.findall(text)
    count = 0
    for zh_word in zh_words:
        count += len(zh_word)
    
    return count

def label_id_to_label_name(id):    
    if id == 0:
        return 'Stand'
    elif id == 1:
        return 'Other'
    elif id == 2:
        return 'Fall'
    
    return None


##### 把圖片轉換為 圖檔名稱(key) 及 value  (對應的關節點, label_id)
def filename_to_label_id(name):
    label_id = None
    if 'Stand' in name:
        label_id = 0
    elif 'Other' in name:
        label_id = 1
    elif 'Fall' in name:
        label_id = 2
    return label_id


def get_filename_vector_dic_from_img(data_dir, point_th=18):
    from . import openpose_api as op_api
    # point_th = point_threshold，是指 openpose偵測到的節點，要超過多少，才使用該筆資料
    files = listdir(data_dir)
    fileName_Xy_dict = dict()

    for f in tqdm(files):
        file_name = data_dir + f
        img = cv2.imread(file_name)
        if img is not None:
            img0 = img.copy()        
            img, pkp = op_api.body_from_image(img)
            if pkp is not None:
                for person in pkp:
                    real_point = sum([1 for x,y,c in person if c > 0])

                    if real_point >= point_th:
                        v = []
                        for x,y,c in person:
                            v.append(x)
                            v.append(y)
                        
                        label_id = filename_to_label_id(file_name)
                        fileName_Xy_dict[file_name] = (v, label_id)
                        break

            # Display Image
            # cv2.imshow("img", img)
            # cv2.waitKey(100)
            # print("Body keypoints %d: \n" % i, str(pkp))
        else:
            print(file_name, 'No Image Read')

    return fileName_Xy_dict


def get_filename_vector_dic_from_img_has_c(data_dir, point_th=18):
    from . import openpose_api as op_api
    # 將偵測到的關節點的 x y 及信心度 c 都轉成向量
    files = listdir(data_dir)
    fileName_Xy_dict = dict()

    for f in tqdm(files):
        file_name = data_dir + f
        img = cv2.imread(file_name)
        if img is not None:
            img0 = img.copy()        
            img, pkp = op_api.body_from_image(img)
            if pkp is not None:
                for person in pkp:
                    real_point = sum([1 for x,y,c in person if c > 0])

                    if real_point >= point_th:
                        v = []
                        for x,y,c in person:
                            v.append(x)
                            v.append(y)
                            v.append(c)
                        
                        label_id = filename_to_label_id(file_name)
                        fileName_Xy_dict[file_name] = (v, label_id)
                        break
        else:
            print(f, 'No Image Read')

    return fileName_Xy_dict


from pathlib import Path
class ImageSaver:
    def __init__(self, max_save_num=False):
        """
            max_save_num: 最多要儲存幾張圖片，超過會從最舊的開始刪除
            ### 注意！圖片檔名不能改。要取走圖片，請全部拿走，不要只拿部分
        """
        self.num_dict = dict()
        self.min_num_dict = dict()
        self.max_save_num = max_save_num
            
        
    def get_files_max_num(self, dir, img_name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        files = listdir(dir)
        pattern = re.compile(img_name + '_([0-9]+).*\..*$')

        file_nums = []
        for f in files:
            tmp = re.findall(pattern, f)
            if len(tmp):                
                file_nums.append(int(tmp[0]))
        
        return 0 if len(file_nums)==0 else max(file_nums) + 1

    def get_files_min_num(self, dir, img_name):
        files = listdir(dir)
        pattern = re.compile(img_name + '_([0-9]+).*\..*$')

        file_nums = []
        for f in files:
            tmp = re.findall(pattern, f)
            if len(tmp):                
                file_nums.append(int(tmp[0]))
        
        return -1 if len(file_nums)==0 else min(file_nums)
    

    def save(self, img, dir='data/', img_name='img', verbose=True):
        prepath = os.path.join(dir, img_name)
        if prepath not in self.num_dict:
            num = self.get_files_max_num(dir, img_name)
            self.num_dict[prepath] = num
        else:
            num = self.num_dict[prepath]
        

        files_num = len(list(Path(dir).glob('*.png')))
        if self.max_save_num and files_num > self.max_save_num:
            # 檔案數量超過最大限制
            if  prepath not in self.min_num_dict:
                min_num = self.get_files_min_num(dir, img_name)
                self.min_num_dict[prepath] = min_num
            else:
                min_num = self.min_num_dict[prepath]
            
            delete_path_png = f'{prepath}_{min_num}.png'
            # delete_path_txt = f'{prepath}_{min_num}.txt'
            try:
                print(f'檔案數量超過最大限制，嘗試刪除 "{delete_path_png}" : ', end='')
                os.remove(delete_path_png)
                # os.remove(delete_path_txt)
                self.min_num_dict[prepath] += 1
                print('delete done...')
            except Exception as e:
                logger.error(f'ImageSaver:*** delete failed... 請檢察儲存照片的資料夾之編號是否不連續 ***, {e}')

        
        path = f'{prepath}_{num}.png'
        # cv2.imwrite(path, img)
        cv2_save_zh_img(path, img)

        if verbose:
            print('save done...', path)
                
        self.num_dict[prepath] += 1
        return path


from datetime import datetime
class ImageSaverTime:
    def __init__(self, max_save_num=False):
        """
            max_save_num: 最多要儲存幾張圖片，超過會從最舊的開始刪除
            ### 注意！圖片檔名不能改。
        """
        self.max_save_num = max_save_num
        self.sorted_file_list = dict()
        self.cursor = dict()
            
    # def get_files_min_num(self, dir, img_name):
    #     files = listdir(dir)
    #     pattern = re.compile(img_name + '_([0-9]+).*\..*$')

    #     file_nums = []
    #     for f in files:
    #         tmp = re.findall(pattern, f)
    #         if len(tmp):                
    #             file_nums.append(int(tmp[0]))
        
    #     return -1 if len(file_nums)==0 else min(file_nums)
    
    def save(self, img, dir='data/', img_name='', verbose=True):
        if not os.path.exists(dir):
            os.makedirs(dir)

        if img_name != "":
            img_name = '_' + img_name

        files_num = len(list(Path(dir).glob('*.png')))
        if self.max_save_num and files_num > self.max_save_num:
            if dir not in self.cursor:
                self.cursor[dir] = 0
                self.sorted_file_list[dir] = []
 
            if self.cursor[dir] >= len(self.sorted_file_list[dir]):
                self.cursor[dir] = 0
                self.sorted_file_list[dir] = sorted(os.listdir(dir))                

            this_sorted_list = self.sorted_file_list[dir]
            this_cursor = self.cursor[dir]
                            
            delete_img_name = this_sorted_list[this_cursor]            
            delete_img_path = os.path.join(dir, delete_img_name)

            self.cursor[dir]+=1
            try:
                print(f'檔案數量超過最大限制，嘗試刪除 "{delete_img_path}" : ', end='')
                os.remove(delete_img_path)
                # os.remove(delete_path_txt)
                print('delete done...')
            except Exception as e:
                logger.error(f'ImageSaver:*** delete failed... 請檢察是否有移動到資料夾中的照片 ***, {e}')
                
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-4]
        img_name = f'{datetime_str}{img_name}.png'
        path = os.path.join(dir, img_name)
        
        # cv2.imwrite(path, img)
        cv2_save_zh_img(path, img)

        if verbose:
            print('save done...', path)
                
        return path


# 自己連接攝影機取得圖片，送進 yolo 預測前要做的前處理 (一台攝影機(單張圖片) )
from ..yolov5.utils.datasets import letterbox
def preprocess_frame_to_yolo_one_cam(img0, img_size=640, stride=32, auto=True):
    # img0 為1張照片，要放到 list中， 如 [ img ]
    # Letterbox
    if not type(img0) == list and (type(img0) == np.ndarray and img0.ndim==3):
        img0 = [img0]

    img = [letterbox(x, img_size, auto=auto, stride=stride)[0] for x in img0]

    # Stack
    img = np.stack(img, 0)

    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)

    return img, img0

