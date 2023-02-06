
# coding: utf-8
import os
os.environ['PATH']  = os.environ['PATH'] + ';' + 'D:\\ffmpeg-4.4-full_build\\bin;' # 要到 ffmpeg 官網下載 ffmpeg
from playsound import playsound
from vgpy.utils.logger import create_logger
logger = create_logger()

def play_mp3(mp3_path):
    try:
        playsound(mp3_path)    # playsound 播放語音版本
    except Exception as e:
        logger.error(f'測距/物件:播放音檔失敗, {e}')




