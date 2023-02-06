# -*- coding: utf-8 -*-
import cv2
import time
import threading

# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ImageCapture:
    def __init__(self, URL, CAP_DSHOW=None, resolution=None):
        self.Frame = None
        self.status = False
        self.isstop = False
        self.count = 0
        self.t0 = None
        self.url = URL
        
        if self.url.isdigit():
            self.url = int(self.url)
        
        if CAP_DSHOW is None:
            if type(self.url) is int:
                CAP_DSHOW = True # USB攝影機 加上這個連接會比較快 (僅限win10, 在 AGX上加了好像會連不到)
            else:
                # ipcam
                CAP_DSHOW = False

        if CAP_DSHOW:
            self.capture = cv2.VideoCapture(self.url, cv2.CAP_DSHOW)              
            if self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 0:
                self.capture.release()
                self.capture = cv2.VideoCapture(self.url)
        else:
            self.capture = cv2.VideoCapture(self.url)

        if resolution:
            w,h = resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self.camera_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.camera_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.none_frame_count = 0 #計算多少個 frame 取出的 都是 None，太多 None 就要重開攝影機
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        print(self.capture.get(cv2.CAP_PROP_FPS)," fps")
        print(self.camera_width, self.camera_height)
        
    # 啟動子執行緒
        self.start()
    # 暫停1秒，確保影像已經填充
        time.sleep(1)

    def reload(self):
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_DSHOW)
    
    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        self.t0 = time.time()
        # threading.Thread(target=self.queryframe, daemon=True, args=()).start()
        threading.Thread(target=self.queryframe, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()   
            if self.Frame is None:
                self.none_frame_count += 1
                if self.none_frame_count >= 30:
                    self.capture.release()
                    del self.capture
                    self.capture = cv2.VideoCapture(self.url)
                    # print('cam reopen due to fail to get frame')
                    self.none_frame_count = 0
            else:
                self.none_frame_count = 0

            self.count += 1
            time.sleep(0.1)
        
        self.capture.release()


    def stream_image_to_web(self): # 用來產生 可以放到 flask的Response() 的 generator，可以將影像 stream 到網頁上
        while True:            
            if self.Frame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", self.Frame)  #資料格式轉換為.jpg
            # ensure the frame was successfully encoded
            if not flag:
                continue            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')



# 讀取影片，一幀一幀讀，a 前進，b 後退
def play_video_frame_by_frame(video_file):
    import cv2
    # video_file = 'video/每3張預測一次.mp4'
    # video_file = 'video/WIN_20210305_15_11_53_Pro.mp4'

    cap = cv2.VideoCapture(video_file)

    frame_history = []
    current = 0
    exit_flag = False
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if ret:
            frame_history.append(frame)
        

        cv2.imshow('video', frame_history[current])
        while True:
            key = cv2.waitKey(1)
            if key == 97: # a
                current += 1

            elif key == 98: # b
                current -= 1

            elif key == 27:
                exit_flag = True
            break
        
        if exit_flag: break


    cap.release()
    cv2.destroyAllWindows()


class VideoReader:
    def __init__(self, fps=10):
        self.video_path = None
        self.video = None
        self.fps = fps

    def read_video_file(self, video_path):
        if self.video is not None:
            self.video.release()
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)

    def reload_video(self):
        if self.video is not None:
            self.video.release()
        self.video = cv2.VideoCapture(self.video_path)

    def get_frame(self):
        if self.video.isOpened():
            success, frame = self.video.read()
            if success:
                return frame
            else:
                self.video.release()

    def get_frame_fix_fps_generator(self):
        t0 = time.time()
        while self.video.isOpened():
            while time.time()-t0 < 1/self.fps:
                pass
            t0 = time.time()
            success, frame = self.video.read()
            yield success, frame

