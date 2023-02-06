from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase  #用於承載附檔
from email import encoders #用於附檔編碼
import smtplib
import codecs
import datetime
import os
from .json_utils import load_json
import time
from vgpy.utils.logger import create_logger
logger = create_logger()

CONFIG_DIR = os.path.join(os.getcwd(),'vgpy','config')
MAIL_RECEIVE_GROUP =load_json(CONFIG_DIR, 'common_setting')['email_setting']

APP_PASSWORD = 'eaaigpodnfvyaadf'  #Google 應用程式密碼

def send_email(group_name , subject, mail_content, img_files=None, attach_files=None):
    from_addr="yingyywang@gmail.com" #寄件者
    to_addr = []     #收件者清單
    if MAIL_RECEIVE_GROUP.get(group_name):
        for recipient in MAIL_RECEIVE_GROUP[group_name]:
            to_addr.append(recipient['mail'])
    else:
        return "查無收件群組，無法寄送郵件"
    
    # 載入郵件內容模板
    f = codecs.open("templates\mail_templete.html", 'r',encoding="utf-8")
    html = f.read()
    # content_str = mail_content #"<b>一級警報 請注意！！</b><br>虛擬圍籬偵測到有人員靠近，請派區域負責人到場巡查，確保人員安全，謝謝！"
    dt = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    html = html.replace("[[Mail_Content]]",mail_content)
    html = html.replace("[[Datetime]]",dt)

    #建立MIMEMultipart物件
    msgRoot  = MIMEMultipart()  

    # 郵件內容
    msgRoot["subject"] = subject  #郵件標題
    msgRoot["from"] = from_addr   #寄件者[預設]
    msgRoot["to"] =', '.join(to_addr)   #收件者
    content = MIMEText(html, "html", "utf-8")
    msgRoot.attach(content)
    # msgRoot.attach(MIMEText("Demo python send email"))  #郵件內容

    #寄送圖片
    if img_files:
        for img_file in img_files:
            with open(img_file, "rb") as file:
                img_file =file.read()
            msgImage = MIMEImage(img_file)
            msgImage["Content-Type"]="application/octet-stream"  
            msgImage["Content-Disposition"]='attachment; filename="%s.jpg"'%os.path.basename(img_file)   #寫你的檔案名讓他可以找到
            # msgImage.add_header('Content-ID', 'image1') # 这个id用于上面html获取图片
            msgRoot.attach(msgImage)

    #寄送附件檔案
    if attach_files:
        for a_file in attach_files:
            with open(a_file, "rb") as file:
                file_content = file.read()
                add_file = MIMEBase('application', "octet-stream")
                add_file.set_payload(file_content)
            encoders.encode_base64(add_file)
            add_file.add_header('Content-Disposition', 'attachment', filename=f'{os.path.basename(a_file)}.jpg')
            msgRoot.attach(add_file)

    msg = msgRoot.as_string() #將物件轉為str
    with  smtplib.SMTP("smtp.gmail.com", 587) as smtp:  # 設定SMTP伺服器
        try:
            smtp.ehlo()  # 驗證SMTP伺服器
            smtp.starttls()  # 建立加密傳輸
            start_time = time.time()
            smtp.login(from_addr, APP_PASSWORD)  # 登入寄件者gmail
            # smtp.send_message(msgRoot)  # 寄送郵件
            smtp.sendmail(from_addr, to_addr, msg)
            return("郵件傳送成功! 發送花費時間: %d 秒"%(time.time()-start_time))
        except Exception as e:
            logger.error(f'郵件傳送失敗, {e}')
            return("郵件傳送失敗! Error message: ", e)

import cv2
def send_email_ndarray_img(group_name , subject, mail_content, img_files=None, attach_files=None):
    from_addr="yingyywang@gmail.com" #寄件者
    to_addr = []     #收件者清單
    if MAIL_RECEIVE_GROUP.get(group_name):
        for recipient in MAIL_RECEIVE_GROUP[group_name]:
            to_addr.append(recipient['mail'])
    else:
        return "查無收件群組，無法寄送郵件"
    
    # 載入郵件內容模板
    f = codecs.open("templates\mail_templete.html", 'r',encoding="utf-8")
    html = f.read()
    # content_str = mail_content #"<b>一級警報 請注意！！</b><br>虛擬圍籬偵測到有人員靠近，請派區域負責人到場巡查，確保人員安全，謝謝！"
    dt = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    html = html.replace("[[Mail_Content]]",mail_content)
    html = html.replace("[[Datetime]]",dt)

    #建立MIMEMultipart物件
    msgRoot  = MIMEMultipart()  

    # 郵件內容
    msgRoot["subject"] = subject  #郵件標題
    msgRoot["from"] = from_addr   #寄件者[預設]
    msgRoot["to"] =', '.join(to_addr)   #收件者
    content = MIMEText(html, "html", "utf-8")
    msgRoot.attach(content)
    # msgRoot.attach(MIMEText("Demo python send email"))  #郵件內容

    #寄送圖片
    if img_files:
        for img_file in img_files:
            with open(img_file, "rb") as file:
                img_file =file.read()
            msgImage = MIMEImage(img_file)
            msgImage["Content-Type"]="application/octet-stream"  
            msgImage["Content-Disposition"]='attachment; filename="%s.jpg"'%os.path.basename(img_file)   #寫你的檔案名讓他可以找到
            # msgImage.add_header('Content-ID', 'image1') # 这个id用于上面html获取图片
            msgRoot.attach(msgImage)

    #寄送附件檔案
    if attach_files:
        for a_file in attach_files:
            file_content = cv2.imencode('.jpg', a_file)[1].tostring()
            add_file = MIMEBase('application', "octet-stream")
            add_file.set_payload(file_content)
            encoders.encode_base64(add_file)
            add_file.add_header('Content-Transfer-Encoding', 'base64')
            add_file.add_header('Content-Disposition', 'attachment', filename='view.jpg')
            msgRoot.attach(add_file)

    msg = msgRoot.as_string() #將物件轉為str
    with  smtplib.SMTP("smtp.gmail.com", 587) as smtp:  # 設定SMTP伺服器
        try:
            smtp.ehlo()  # 驗證SMTP伺服器
            smtp.starttls()  # 建立加密傳輸
            smtp.login(from_addr, APP_PASSWORD)  # 登入寄件者gmail
            # smtp.send_message(content)  # 寄送郵件
            smtp.sendmail(from_addr, to_addr, msg)
            return("郵件傳送成功!")
        except Exception as e:
            logger.error(f'郵件傳送失敗, {e}')
            return("郵件傳送失敗! Error message: ", e)



# html="""
# <!doctype html>
# <html>
# <head>
#   <meta charset='utf-8'>
#   <title>HTML mail</title>
# </head>
# <body>
#   <b style="color:red">此為測試發送信件YAYA ^_^</b>
# </body>
# </html>
# """
if __name__ == '__main__':
    send_email('TeamName1','1006 PYTHON測試發送1',"此為測試發送信件，今天天氣晴")