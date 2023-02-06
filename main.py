from sys import path
import multiprocessing

from vgpy.utils.logger import create_logger
from vgpy.utils.json_utils import load_json, save_json
from vgpy.utils.img_utils import put_zh_text_opencv
import cv2
import json
from flask import Flask, render_template, request, redirect, Response, url_for, jsonify, make_response, send_from_directory

# from werkzeug.utils import secure_filename
from threading import Thread
import os
import numpy as np
import torch
from flask_login import LoginManager, UserMixin, login_user, current_user, login_required, logout_user  

from vgpy.utils.config_utils import get_config,save_config, remove_config_section
from vgpy.utils.camera_video_read import ImageCapture

from vgpy.global_object import StreamingQueues, GlobalVar
from vgpy.global_function import process_response_image, get_administrator_dict


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

current_dir = os.path.dirname(os.path.abspath(__file__))
RootPath = os.getcwd()

# EQPT_CONFIG_TXT = os.path.join(RootPath, 'config', 'EQPT_Config.txt')
# EQPT_ID = None  #設備編號 (對應樹莓派)
# with open(EQPT_CONFIG_TXT, 'r') as f:
#     EQPT_ID = f.read()

app = Flask(__name__)
app.config.from_object(__name__) 
app.config['CUSTOM_STATIC_PATH'] =RootPath + "/clinet/"
app.config["DEBUG"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# flask 終端會打印網頁get/post狀態 儲存log時想忽略此類訊息
# log = logging.getLogger('werkzeug')
# log.disabled = True
#========================= 登入管理相關 Start =====================#
#region
app.secret_key = 'mcf0a2aiteam' 
login_manager = LoginManager(app)
login_manager.login_view = 'admin_login'

users = get_administrator_dict()

login_flag = 0
# #強制登出 避免系統重置還遺留登入session
# logout_user()

class User(UserMixin):  
    pass

@login_manager.user_loader  
def user_loader(email):  
    """  
    設置二： 透過這邊的設置讓flask_login可以隨時取到目前的使用者id   
    :param email:官網此例將email當id使用，賦值給予user.id    
    """   
    if email not in users:  
        return  
  
    user = User()  
    user.id = email  
    return user  

@app.route('/id_register', methods=['POST'])  # 系統管理員身分註冊
def id_register():
    if request.method =='POST':
        data = request.get_json()
        data_arr = data.split(',')
        account = data_arr[0]
        password = data_arr[1]
        level = 'admin'
        new_user = {
            account: {
                "password": password,
                "level": level
            }
        }
        if users.get(account):
            return "該用戶已存在"
        else:
            users.update(new_user)
            save_administrator_dict(users)
            return "註冊完成"

@app.route('/id_verify', methods=['GET', 'POST'])  # 系統管理員身分驗證機制
def id_verify():
    global login_flag 
    users = get_administrator_dict()
    if request.method =='POST':
        data = request.get_json()
        data_arr = data.split(',')
        account = data_arr[0]
        password = data_arr[1]
        # level = data_arr[2] #權限等級 (admin or user)
        try:
            level = users[account]['level']
            if level =='both' or level == 'admin':
                if account in users.keys():
                    if password == users[account]['password']: 
                        #  實作User類別  
                        user = User()  
                        #  設置id就是email  
                        user.id = account
                        #  這邊，透過login_user來記錄user_id，如下了解程式碼的login_user說明。  
                        login_user(user)
                        #  記錄登入狀態
                        
                        if users[account]['level'] == 'admin':  
                            login_flag = 1
                        else:
                            login_flag = 2
                        #  登入成功，轉址  
                        return 'success' #redirect(url_for('admin'))  
        except:
            return 'fail'
        else:
            return 'Bad login'  
    return render_template('face_admin_login.html')


@app.route('/logout', methods=['POST'])  
def logout():
    global login_flag  
    """  
    logout\_user會將所有的相關session資訊給pop掉 
    """ 
    if request.method =='POST':
        logout_user()  
        login_flag = 0 #登出狀態
        return 'logout success' #redirect(url_for('admin_login'))  

@login_manager.unauthorized_handler
def unauthorized_handler():
    # return redirect(url_for('face.admin_login'))    
    return redirect(url_for('vg_login'))    

#========================= 登入管理相關 End =====================#


#======================= 人臉管制系統 =====================================#

# 強制解鎖 -- TODO
@app.route('/door_unlock', methods=['POST'])
def door_unlock():
    if request.method =='POST':
        return 'True'
        
@app.route('/show_photo', methods=['GET'])
def show_photo():
    sys = request.args.get('sys')
    file_path = request.args.get('path')
    if sys == 'wearing':
        filepath = os.path.join(current_dir, 'vgpy', sys, 'img', file_path)
    elif sys == 'face':
        filepath = os.path.join(current_dir, 'vgpy', sys, 'image', 'face', file_path)
    elif sys == 'process':
        filepath = os.path.join(current_dir, 'vgpy', sys, 'recipes', file_path)
    return process_response_image(filepath)


@app.route("/screenShot",methods=['POST'])
def screenShot():
    global current_frame    
    if globalvar.click_screenshot:
        globalvar.click_screenshot = False
        return '返回成即時影像'
    else:
        globalvar.screen_shot_picture = current_frame
        globalvar.click_screenshot = True
        return '截圖成功'

#======================= End Region ======================================#


#======================= 攝影機相關 =====================================#
#====== 控制攝影機切換 ======#
def get_cam_url_from_config(cid):
    global cam_config
    cid = str(cid)
    try:
        return (cam_config['cam'][cid]).split(',')[1]
    except Exception as e:
        logger.error(f'main:找不到指定攝影機id的 url, {e}')
    
def get_cam_config():
    try:
        cam_config = get_config(cam_config_path) 
        cam_json = dict()
        for i in range(1, len(cam_config['cam'])+1):
            cam_json[i] =  cam_config['cam'][str(i)]
        return cam_json
    except Exception as e:
        logger.error(f'main:無法取得攝影機config的json, {e}')

@app.route('/save_cam_config', methods=['POST'])
def save_cam_config():
    global cam_config, capture
    new_config = request.get_json()
    remove_config_section(cam_config_path, 'cam')
    for item in new_config:
        save_config(cam_config_path, 'cam', str(item['id']), str(item['site']+','+ item['ip']))  
    globalvar.cam_num = len(new_config)
    cam_config = get_config(cam_config_path) # 重新讀取cam_config.ini
    capture.url = get_cam_url_from_config(globalvar.cam_id)
    capture.reload()
    return "變更成功"

@app.route('/change_cam/<new_camid>')
def change_cam(new_camid):
    global capture, globalvar
    new_camid = int(new_camid)
    new_cam_url = get_cam_url_from_config(new_camid)
    new_capture = ImageCapture(new_cam_url, resolution=cam_size)
    capture.stop()
    capture = new_capture
    globalvar.cam_id = new_camid
    globalvar.cam_wh = (capture.camera_width, capture.camera_height)
    save_config(cam_config_path, 'cam_config', 'last_cam_id', str(new_camid))    
    streaming_queues.c_queue.put(GlobalVar.SIGNAL.CHANGE_CAM_ID)
    streaming_queues.c_queue.put((new_camid, globalvar.cam_wh))
    return "攝影機切換成功"

#====== 攝影機讀取 + 預測 + 串流到網頁 ======#
@app.route('/stream/<page>', methods=['GET'])
def camera_read_predict_stream(page):
    def stream_generator():
        global current_frame, facevar, globalvar, current_app, process_main
        result_img = None
        #====== (遠端監控) 影像連接失敗設定 ========
        from vgpy.config import color 
        # global remote_cam_id       
        h,w = 1080, 1920
        font = '畫面連接失敗'
        font_size = 120
        text_location = ((w-len(font)*font_size)//2, (h-font_size)//2)
        not_connected_img = np.zeros((h, w, 3), dtype='uint8')
        not_connected_img = put_zh_text_opencv(not_connected_img, font, text_location, color.淺綠, fsize=font_size)
 
        while True:
            # 讀取攝影機
            current_frame = capture.getframe()

            if current_frame is not None:

                if current_app == 'wearing' and wearingvar.click_camera_collect:
                    # result_img = current_frame # 原始沒標 YOLO框的圖片 (收集照片時沒有框出頭部)                                    
                    streaming_queues.d_queue.put(current_frame)
                    try:
                        result_dict = streaming_queues.r_queue.get() # 等待照片截取 + 儲存
                        wearingvar.collect_data = result_dict['files']
                        result_img = result_dict['img_with_box']
                    except Exception as e:
                        logger.error(f'main:cannot get result_queue, {e}')

                elif streaming_queues.c_queue.empty():
                   
                    if page == "wearSetting" or page == "wearCollection":
                        result_img = current_frame

                    elif page == "wearingIndex":
                        streaming_queues.d_queue.put(current_frame)
                        result_img = streaming_queues.r_queue.get()

                    else:
                        # 控制佇列沒有訊號時，才將照片送進去 data_queue 並等待辨識結果
                        streaming_queues.d_queue.put(current_frame)
                        try:
                            results = streaming_queues.r_queue.get()
                            # 辨識的 Process 回傳多個 變數
                            if type(results) is list: # 這種型態的回傳，目前應該只有人臉、動作有 
                                for r in results:
                                    if type(r) is ProcessVar.ProcessResult:  #NOTE:動作多回傳肢體節點影像
                                        result_img = r.result_img
                                        current_pose = r.current_pose
                                        processvar.current_pose = current_pose
                                    elif type(r) == ProcessVar:
                                        processvar.update(r) # 更新肢體節點影像frame
                            
                            # 「流程辨識」完成 要切換成 非 辨識中
                            elif results is ProcessVar.SIGNAL.PROCESS_FINISHED:
                                globalvar.click_start = False
                                processvar.current_pose = '__流程完成__'

                            # 僅回傳辨識結果
                            else:
                                result_img = results
                                                    
                        except Exception as e:
                            result_img = current_frame
                            logger.error(f'main:cannot get result_queue, {e}')

                else:
                    result_img = current_frame
                
            else:
                result_img = not_connected_img
                result_img = put_zh_text_opencv(
                    result_img, f'Cam:{globalvar.cam_id}', (10,10), color.白色, fsize=font_size
                )

            if result_img is not None:
                (flag, encodedImage) = cv2.imencode(".jpg", result_img)  #資料格式轉換為.jpg    
                if flag: # ensure the frame was successfully encoded
                    encodedImage = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'
                    yield encodedImage           

    return Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


#====== 遠端監控的攝影機讀取 + 串流到網頁 ======#
@app.route('/remote_old/stream')
def remote_camera_read_stream():
    def stream_generator():
        global remote_cam_id, remote_captures, current_frame, vd_flag, video_writer
        from vgpy.config import color        
        h,w = 1080, 1920
        font = '畫面連接失敗'
        font_size = 120
        text_location = ((w-len(font)*font_size)//2, (h-font_size)//2)
        not_connected_img = np.zeros((h, w, 3), dtype='uint8')
        not_connected_img = put_zh_text_opencv(not_connected_img, font, text_location, color.淺綠, fsize=font_size)
        while True:
            try:
                current_frame = remote_captures[remote_cam_id].getframe()                
            except Exception as e:                
                current_frame = not_connected_img
                current_frame = put_zh_text_opencv(
                    current_frame, f'Cam:{remote_cam_id}', (10,10), color.白色, fsize=font_size
                )

            if vd_flag:
                video_writer.write(current_frame)

            if current_frame is not None:               
                (flag, encodedImage) = cv2.imencode(".jpg", current_frame)  #資料格式轉換為.jpg        
                if flag: # ensure the frame was successfully encoded
                    encodedImage = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'
                yield encodedImage
    return Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

#====== 遠端監控的攝影機切換 ======#
@app.route("/remote_old/change_cam",methods=['POST'])
def switch_Cam():
    global remote_cam_id
    if request.method =='POST':        
        if remote_cam_id+1 >= len(remote_cam_url_list):
            remote_cam_id = 0
        else:
            remote_cam_id += 1

        return str(remote_cam_id)


#======================= End Region ========================================#

#======================= 從主頁進入各個子功能 =====================================#
@app.route('/start_app/<app_name>')
def start_app(app_name):
    global is_app_running, current_app
    if is_app_running:
        return 'start app error'

    globalvar.current_process = None   
    # if app_name == "vf":
    #     globalvar.current_process = vf_main.app_vf_start()
    if app_name == "dist_detect":
        globalvar.current_process = vf_main.app_vf_start()
        if globalvar.click_screenshot:
            screenShot()    #如果剛進系統還停在截圖模式 請重新呼叫跑即時影像流
    elif app_name == "object_detect":
        globalvar.current_process = vf_main.app_vf_start()
    elif app_name == "wearing":
        globalvar.current_process = wearing_main.app_wearing_start()
    # elif app_name == 'process':
    #     globalvar.current_process = process_main.app_process_start()
    elif app_name == 'face':
        globalvar.current_process = faceTool.app_face_start()
    # elif app_name == 'remote_old':
    #     globalvar.current_process = app_remote_start()
    
    current_app = app_name
    is_app_running = True
    return 'app started'

#======================= End Region ===================================#

def app_remote_start():
    global remote_cam_id
    remote_cam_id = 0
    def start_all_cam():
        global remote_captures
        remote_captures = []
        for c_url in remote_cam_url_list:
            capture = ImageCapture(c_url)
            if remote_captures is None:
                capture.stop()
                break
            remote_captures.append(capture)

    current_process = Thread(
        target=start_all_cam, daemon=True)
    current_process.start()
    return current_process

#======================= End Region ========================================#
@app.route('/', methods=['GET'])  # 五大系統主頁
def vg_index():
    global is_app_running, streaming_queues, remote_captures, current_app
    if is_app_running:
        logger.info("有正在執行的子系統，嘗試關閉中...")
        if remote_captures is None:
            streaming_queues.c_queue.put(GlobalVar.SIGNAL.END_PROCESS)
            try:
                # 當回到主頁時，會嘗試把目前正在執行的 process 終止掉，主要是為了釋放 GPU的顯存
                globalvar.current_process.terminate()
                logger.info("關閉成功")
            except Exception as e:
                logger.error(f'main:terminate current subprocess failed, {e}')

            streaming_queues.reset()
            
        else: # 遠端監控有啟動，回首頁時要關掉全部攝影機
            for capture in remote_captures:
                capture.stop()
            remote_captures = None

        is_app_running = False
        current_app = None

    logger.info("進入VG主頁")
    return render_template('VG_index_new.html')

@app.route('/login', methods=['GET'])  # 五大系統登入頁
def vg_login():
    if current_user.is_active: 
        return redirect(url_for('vg_index'))  
    else:
        return render_template('VG_login.html')

@app.route('/global_setting', methods=['GET'])  # 五大系統設定頁
@login_required
def vg_setting():
    f_path= os.path.join(current_dir, 'vgpy', 'config')
    cmn_setting = load_json(f_path, 'common_setting.json')
    mail_json = cmn_setting['email_setting'] #load_json(f_path, 'email_config.json')
    cam_json = get_cam_config()
    admin_json = cmn_setting['admin_setting'] #get_administrator_dict()
    line_json = cmn_setting['line_setting'] #get_administrator_dict()
    link_json = cmn_setting['link_setting']
    return render_template('VG_setting.html', mail_dict=mail_json, cam_dict=cam_json, admin_dict=admin_json, line_dict=line_json, link_json=link_json)

@app.route('/crud_gb_setting', methods=['POST'])
def crud_gb_setting():
    if request.method =='POST':
        result =  request.get_json()
        try:
            c_path= os.path.join(current_dir, 'vgpy', 'config')
            setting_json = load_json(c_path, 'common_setting.json')
            
            #更新郵件資訊
            if result['item'] == 'upd_mail_group':
                print(result['member'])
                wr_json = { 
                    result['group']:result['member']
                }
                if setting_json['email_setting'].get(result['group']):
                    setting_json['email_setting'][result['group']] = result['member']
                else:
                    setting_json['email_setting'].update(wr_json)
                save_json(setting_json, c_path, 'common_setting.json')
                globalvar.mail_group =  setting_json['email_setting']    # json格式
            #刪除郵件群組
            if result['item'] == 'del_mail_group':
                if setting_json['email_setting'].get(result['group']):
                    setting_json['email_setting'].pop(result['group'])
                    save_json(setting_json, c_path, 'common_setting.json')
                else:
                    return '你尚未建立此群組的資訊，請先儲存！'
                globalvar.mail_group =  setting_json['email_setting']    # json格式

            #重新命名郵件群組
            if result['item'] == 'rename_mail_group':
                if setting_json['email_setting'].get(result['group_name']):
                    setting_json['email_setting'][result['group_new_name']] = setting_json['email_setting'].pop(result['group_name'])
                    save_json(setting_json, c_path, 'common_setting.json')
                else:
                    return '修改失敗'
                globalvar.mail_group =  setting_json['email_setting']    # 做上述任何動作，最後都要更新 mail_group (global參數)

            #更新權限管理者資訊
            if result['item'] == 'upd_admin_dict':
                current_json = {}
                for val in result['data']:
                    id = val['id']
                    current_json[id] = {
                        'password': val['password'],
                        'level': 'admin'
                    }
                setting_json['admin_setting'] = current_json
                save_json(setting_json, c_path, 'common_setting.json')

            #更新LINE設定資訊
            if result['item'] == 'upd_line_set':
                setting_json['line_setting']['token'] = result['token']
                save_json(setting_json, c_path, 'common_setting.json')

            #更新LINE設定資訊
            if result['item'] == 'upd_link_set':
                setting_json['link_setting']['systemCode'] = result['systemCode']
                setting_json['link_setting']['subCateNo'] = result['subCateNo']
                save_json(setting_json, c_path, 'common_setting.json')
                
            return 'success'

        except Exception as e:
            logger.error(f'main:crud_gb_setting failed, {e}')
            return 'fail'


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()

    is_app_running = False
    current_frame = None
    current_app = None

    cam_size = (1920, 1080)

    globalvar = GlobalVar() # 全域變數存放位置
    streaming_queues = StreamingQueues()

    
    #======================= 攝影機相關 =====================================#
    cam_config_path = os.path.join(current_dir, 'vgpy', 'config', 'cam_config.ini')
    cam_config = get_config(cam_config_path) # 攝影機的 config 檔案
    
    globalvar.cam_id = int(cam_config['cam_config']['last_cam_id'])
    globalvar.cam_num = len(cam_config['cam'])
    # TODO USB test
    capture = ImageCapture(
        get_cam_url_from_config(globalvar.cam_id), CAP_DSHOW=True, resolution=cam_size
    )

    globalvar.cam_wh = (capture.camera_width, capture.camera_height)

    CONFIG_DIR = os.path.join(current_dir, 'vgpy', 'config')
    globalvar.mail_group = load_json(CONFIG_DIR, 'common_setting.json')['email_setting']

    #======================= logging相關 =====================================#
    import logging

    logger = create_logger()

    # 穿戴偵測 初始化
    from vgpy.wearing.wearing_api import WearingMain, WearingVar
    wearingvar = WearingVar()
    wearingvar.yolo_conf = 0.35
    wearing_main = WearingMain(streaming_queues, wearingvar, globalvar)
    app_wearing = wearing_main.wearing_app_init()
    app.register_blueprint(app_wearing, url_prefix='/wearing')

    
    remote_captures = None

    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)