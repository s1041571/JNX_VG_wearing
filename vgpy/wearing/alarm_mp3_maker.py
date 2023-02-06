import json
import os
# from gtts import gTTS
### 使用說明
# mp3maker = AlarmMP3Maker('audio/')
# mp3maker.make_mp3('安全帽')

class AlarmMP3Maker:
    def __init__(self, audio_dir) -> None:
        self.audio_dir = audio_dir
        self.MAPPING_FILE_NAME = 'alarm_mp3_mapping'

        # 若 audio 資料夾不存在，則創建
        try:
            os.makedirs(audio_dir)
        except:
            print(f'創建資料夾 {audio_dir} 失敗')
        
        if os.path.isfile( os.path.join(audio_dir, f'{self.MAPPING_FILE_NAME}.json') ):
            self.wname_mp3file_dict = self.load_json(self.MAPPING_FILE_NAME) # 讀取原本的對照檔，才能接續編碼 對應到的 mp3 index
            print(f'已載入先前的 {self.MAPPING_FILE_NAME}')
        else:
            self.wname_mp3file_dict = dict()
            print('創建空白的 Dictionary')


    def make_mp3(self, wearing_name):
        # 將每個 alarm_name 作為 key，value為這個裝備沒配戴 的警告音檔
        alarm_text = self.wearing_name_to_alarm_text(wearing_name)
        if alarm_text not in self.wname_mp3file_dict:
            mp3_idx = self.wname_mp3file_dict.values()
            mp3_idx = 0 if len(mp3_idx) == 0 else max(mp3_idx)+1 # 取目前有的 index最大值 +1，若沒有index則為0            

            self.text_to_speech(alarm_text, mp3_filename=str(mp3_idx))
            self.wname_mp3file_dict[alarm_text] = mp3_idx

            print(f"{alarm_text}->{mp3_idx}， 音檔已生成")
            self.save_json(self.wname_mp3file_dict, self.MAPPING_FILE_NAME) # 將 alarm_text中文，對應到指定的mp3檔編號
            print(f'已更新 {self.MAPPING_FILE_NAME} 檔')
        else:
            pass
            # print(f"{wearing_name} 之音檔已存在")
            

    def save_json(self, aDict, filename):
        filename = filename+'.json' if '.json' not in filename else filename
        filepath = os.path.join(self.audio_dir, filename)
        with open(filepath, 'w', encoding='utf8') as f:
            json.dump(aDict, f, indent=2, ensure_ascii=False)


    def load_json(self, filename):
        filename = filename+'.json' if '.json' not in filename else filename
        filepath = os.path.join(self.audio_dir, filename)
        with open(filepath, 'r', encoding='utf8') as f:
            return json.load(f)


    def text_to_speech(self, text, mp3_filename):
        tts=gTTS(text=text, lang='zh')
        if '.mp3' not in mp3_filename:
            mp3_filename = mp3_filename+'.mp3'
        mp3_file_path = os.path.join(self.audio_dir, mp3_filename)
        tts.save(mp3_file_path)

    def wearing_name_to_alarm_text(self, wearing_name):
        return wearing_name + "沒有穿戴"

    def get_wearing_alarm_mp3_name(self, wearing_name):
        alarm_text = self.wearing_name_to_alarm_text(wearing_name)
        return self.wname_mp3file_dict[alarm_text]


