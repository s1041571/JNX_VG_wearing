import os
import glob
from vgpy.utils.json_utils import load_json, save_json
from flask import make_response

def process_response_image(img_path):
    if not os.path.isfile(img_path):  #若找不到 人名.jpg 的照片檔，就找資料夾中第一張 人名_*.jpg 的照片檔
        temp = img_path.split(".jpg")
        img_path = glob.glob(f'{temp[0]}_*')[0]

    image_data = open(img_path, "rb").read()
    response = make_response(image_data)
    response.headers['Content-Type'] = 'image/png'
    return response   

#import pickle
import pickle
def save_pickle_object(obj, filename):
    with open(filename, 'ab') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def unpickle_database(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def get_administrator_dict():
    config_dir = os.path.join(os.getcwd(),'vgpy','config')
    setting_json = load_json(config_dir, 'common_setting')
    return setting_json['admin_setting']

def os_makedirs(dirs, verbose=True):
    try:
        os.makedirs(dirs)
        if verbose:
            print(f'已建立 dirs: {dirs}')
    except:
        print(f'已經存在的 dirs: {dirs}')


def remove_imagefile_extention(file):
    file = file.replace('.jpg', '')
    file = file.replace('.png', '')
    return file
