import json
from suds.client import Client
import cv2
import base64
from datetime import datetime
import os
import numpy as np
from vgpy.utils.json_utils import load_json

def CV2Base64(img): 
    base64Str = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    return str(base64Str)

def CV2Base64png(img): 
    base64Str = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
    return str(base64Str)

def LinkSOAP(url, string):
    client = Client(url)
    replycode = client.service.PublishEvent(string)
    if str(type(replycode)) != "<class 'NoneType'>":
        if len(replycode) > 0:
            return json.loads(replycode)
        else:
            return 'Link no reply'
    else:
        return 'Link no reply'

def Linkpost(camFrame, msg_title, msg_text, systemCode="396hLng7S4", subCateNo="3241"):
    systemCode = (load_json(os.path.join('.', 'vgpy', 'config'), 'common_setting'))['link_setting']['systemCode']
    subCateNo = (load_json(os.path.join('.', 'vgpy', 'config'), 'common_setting'))['link_setting']['subCateNo']
    url = 'http://au6ifs02/angelia/Prod/EventPlusWS/EventPlusAPI.asmx?WSDL'
    linkTitle = msg_title
    linkMsg = msg_text
    # imgMerge = np.vstack((camFrame, subFrame))
    base64String = CV2Base64(camFrame)
    linkPost = f'''
                    {{'FunctionName':'PublishEvent',
                    'SystemCode':'{systemCode}',
                    'SubCateNo':'{subCateNo}',
                    'EventContent':'{linkTitle}<br />{linkMsg}',
                    'FixedAppendInfo':'value',
                    'HasAttachedImage':true,
                    'AttachedImages':[{{'Base64ImageContent':'{base64String}'}}]}} '''

    linkPost = linkPost.replace('\n', '')
    # try:
    replyCode = LinkSOAP(url, linkPost)
    if type(replyCode) == dict:
        if replyCode['ReturnMsg'] == 'Success':
            print(f"reply:{replyCode['ReturnMsg']}")
        else:
            print(f"reply:{replyCode['ReturnMsg']}")
    else:
        print(replyCode)