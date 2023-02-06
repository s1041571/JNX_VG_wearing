from multiprocessing import JoinableQueue
from enum import Enum, auto

class StreamingQueues:
    def __init__(self):
        self.d_queue = JoinableQueue() # data_queue
        self.r_queue = JoinableQueue() # result_queue
        self.c_queue = JoinableQueue() # controll_queue

    def clear_data_queue(self):        
        while not self.d_queue.empty():
            try:
                self.d_queue.get_nowait()
            except:
                break

    def clear_and_done_control_queue(self):
        # 清空 control queue，怕離開辨識 process後，還有其他殘留的控制訊號在裡面
        while True:
            try:
                self.c_queue.get_nowait()
                self.c_queue.task_done()
            except:
                break
    
    
    def reset(self):
        self.d_queue.close()
        self.r_queue.close()
        self.c_queue.close()
        self.d_queue = JoinableQueue()
        self.r_queue = JoinableQueue()
        self.c_queue = JoinableQueue()

class GlobalVar:
    class SIGNAL(Enum):
        END_PROCESS = auto() # 可用來終止 MultiProcess
        CHANGE_CAM_ID = auto() # 當變更攝影機時，可以讓subprocess內也更新攝影機ID

    def __init__(self):
        """
        @attr
            screen_shot_picture:
                前端呼叫拍照時 (/screenShot)，
                會將照片存在這，
                讓後端也可以使用到此照片
            click_start:
                按下各個 app 中的開始辨識，就會變成True，
                攝影機就會開始將畫面 輸入 d_queue，
                因此目前正在執行的 process 就會開始辨識該畫面並回傳結果到 r_queue
            cam_id:
                紀錄目前正在使用哪台攝影機
            cam_wh:
                目前使用的攝影機的 解析度 tuple(寬, 高)
            current_process:
                儲存目前正在執行的 process
            mail_group:
                通報設定中的Email群組清單
        """
        self.screen_shot_picture = None
        self.click_start = False
        self.cam_id = None
        self.cam_wh = None
        self.cam_num = None
        self.current_process = None
        self.video_writer = None
        self.video_record_start = False
        self.click_screenshot = False
        self.mail_group = None

    def set_screen_shot_picture(self, picture):
        self.screen_shot_picture = picture

    def set_click_start(self, boolean):
        self.click_start = boolean
    

class DetectionVar:
    """
    @attr
        resultImg: AI detection result of image
        resultInfo: AI detection result of information
    """
    def __init__(self):
        self.resultImg = None
        self.resultInfo = None

