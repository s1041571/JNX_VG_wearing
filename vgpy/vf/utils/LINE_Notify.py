import requests, os
import cv2
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from ...utils.json_utils import load_json

# 設置最大重新訪問次數 解決requests.post()超時中斷問題
s = requests.Session()
s.mount('https://', HTTPAdapter(max_retries=Retry(total=5, method_whitelist=frozenset(['GET', 'POST'])))) # 设置 post()方法进行重访问
CONFIG_DIR = os.path.join(os.getcwd(), 'vgpy', 'config')

def lineNotifyMessage(msg):
    token = (load_json(CONFIG_DIR, 'common_setting'))['line_setting']['token']
    img_f = ''
    img_tb=''
    headers = {
        "Authorization": "Bearer " + token, 
        "Content-Type" : "application/x-www-form-urlencoded"
    }
	
    payload = {'message': msg} #,'imageThumbnail':img_tb,'imageFullsize':img_f
    # r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
    r = s.post(url="https://notify-api.line.me/api/notify", headers = headers, params = payload)

    return r.status_code



def lineNotifyMessageImg(msg, imgpath):
    token = (load_json(CONFIG_DIR, 'common_setting'))['line_setting']['token']
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": "Bearer " + token
    }
   
    payload = {'message': msg}
    files = {'imageFile': open(imgpath, 'rb')}
    # r = requests.post(url, headers = headers, params = payload, files = files)
    r = s.post(url, headers = headers, params = payload, files = files)
    return r.status_code
 
 
def line_notify_message_ndarray_img(msg, ndarray_img):
    token = (load_json(CONFIG_DIR, 'common_setting'))['line_setting']['token']
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": "Bearer " + token
    }
   
    payload = {'message': msg}

    img_str = cv2.imencode('.jpg', ndarray_img)[1].tostring()
    files = {'imageFile': img_str}
    # r = requests.post(url, headers = headers, params = payload, files = files)
    r = s.post(url, headers = headers, params = payload, files = files)
    return r.status_code

