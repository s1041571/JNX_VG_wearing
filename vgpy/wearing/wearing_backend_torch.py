import os
import shutil
import cv2
import numpy as np

from ..utils.img_utils import cv2_save_zh_img, cv2_read_zh_img
from ..utils.os_utils import movefiles, copyfiles

from .config import WearingConfig, PART_LIST, PRED_SAVE_DIR_HEAD, ORI_TRAIN_DIR_HEAD, TMP_TRAIN_CHECK_DIR, TMP_TRAIN_DIR, TRAIN_RESULT_DIR, SHUFFLE_NET_MODEL_DIR, IMG_DIR
from ..global_function import process_response_image

config = WearingConfig()
current_dir = os.path.dirname(os.path.abspath(__file__))

from vgpy.utils.logger import create_logger
logger = create_logger()

def remove_imagefile_extention(file):
        file = file.replace('.jpg', '')
        file = file.replace('.png', '')
        return file

# class CRUD
def get_class_list(part):
    return config.get_part_wearing_zh(part)

def post_class(part, zh_name, en_name):
    config.add_class(part, zh_name, en_name)
    logger.info('穿戴:新增穿戴類別成功')

def delete_class(part, zh_name):
    config.delete_class(part, zh_name)
    logger.info('穿戴:刪除穿戴類別成功')


##### route: /wearing/wear_setting #####
def save_wearing_setting(detect_parts, sample_rate, max_quantity, fence_enable, alarm_channel_list):
    config.save_wearing_setting(detect_parts, sample_rate, max_quantity, fence_enable, alarm_channel_list)

def save_class_setting(part, detect_class, confs):
    config.save_class_setting(part, detect_class, confs)

def save_train_args_setting(dict):
    config.save_train_args_setting(dict)

def get_cam_areas(): 
    return config.get_cam_areas()
    
def get_canvas_fence_area_str():
    return config.get_canvas_fence_area_str()

def save_fence_coord(area_json, wh):
    canvas_areas = area_json['area_coord'] # 前端的 canvas 上畫的圍籬座標
    canvas_height =int(area_json['img_height'])
    canvas_width = int(area_json['img_width'])

    w, h = wh
    logger.debug(f'穿戴:寬*高 {str(w)},{str(h)}')
    
    cam_areas = config.from_canvas_to_camera(canvas_areas, canvas_wh=(canvas_width, canvas_height), cam_wh=wh)
                
    config.save_fence_areas_ponits(canvas_data=(canvas_areas,canvas_width,canvas_height), cam_areas=cam_areas)  #add 儲存canvas所有前端資訊

    logger.info(f'穿戴:成功儲存新設定的圍籬區域:\n{cam_areas}') # [[('1781', '41'), ('1670', '127'), ('1858', '129')]]    
    

def get_part_model_list(part):
    # 輸入指定的部位，回傳該部位有哪些訓練好的模型
    # 例如輸入 part=頭部，回傳 [安全帽, 口罩]
    part_model_path = os.path.join(SHUFFLE_NET_MODEL_DIR, part)
    files = os.listdir(part_model_path)
    ptst_files = [file.replace('.ptst', '') for file in files if '.ptst' in file] # 確認是.h5的檔案後， 去除.h5副檔名
    return ptst_files

##### route: /wearing/log_review #####
# 取得異常發報訊息log
def get_alarm_log():
    return config.get_alarm_log()

# 回傳影像
def return_image_to_web(target_dir, filename):
    filepath = os.path.join(IMG_DIR, target_dir, f'{filename}.png')
    return process_response_image(filepath)

##### route: /wearing/model_OPT #######
def get_prediction_filenames(part, cls):
    
    def get_data_and_checkdata(path):
        check_dir_path = os.path.join(path, 'check')
        if not os.path.exists(check_dir_path):
            os.makedirs(check_dir_path)

        files = sorted(set([f for f in os.listdir(path) if '.png' in f]), reverse=True)
        check_files = sorted(set([f for f in os.listdir(check_dir_path) if '.png' in f]), reverse=True)
        # files = files - check_files # 做集合運算的差集，代表 只留下 files有 且 check_files沒有的檔案

        data = [(remove_imagefile_extention(f),
            os.path.join(path, f)) for f in files]

        check_data = [(remove_imagefile_extention(f),
            os.path.join(check_dir_path, f)) for f in check_files]

        return data, check_data

    try:
        o_path = os.path.join(PRED_SAVE_DIR_HEAD, part, f'{cls}_has')
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        o_data, check_x_data = get_data_and_checkdata(o_path)
    except Exception as e:        
        logger.error(f'穿戴:取得模型預測「有穿戴」的檔案失敗, {e}')
        o_data, check_x_data = [], []

    try:
        x_path = os.path.join(PRED_SAVE_DIR_HEAD, part, f'{cls}_no')
        if not os.path.exists(x_path):
            os.makedirs(x_path)
        x_data, check_o_data = get_data_and_checkdata(x_path)
    except Exception as e:
        logger.error(f'穿戴:取得模型預測「沒穿戴」的檔案失敗, {e}')
        x_data, check_o_data = [], []

    return o_data, x_data, check_o_data, check_x_data

def opt_model_sub_process(part, cls, queue, opt_epoch, opt_batch_size, min_lr, resize_shape):
    logger.info(f'穿戴:模型優化開始...')
    queue.put("data loading")
    from . import torch_model
    from .torch_train_utils import EarlyStopping, History, binary_acc
    from torchvision import transforms
    import torch
    from .torch_dataset import WearingDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split

    device = torch.device('cuda')
    dtype = torch.float32

    model_path = os.path.join(SHUFFLE_NET_MODEL_DIR, part, f'{cls}.ptst')
    check_x_path = os.path.join(PRED_SAVE_DIR_HEAD, part, f'{cls}_has', 'check')
    check_o_path = os.path.join(PRED_SAVE_DIR_HEAD, part, f'{cls}_no', 'check')
    check_dirs = [check_x_path, check_o_path]
    check_X, check_y = get_train_data(check_dirs, reshape=resize_shape)
    logger.debug('check_X.shape:{check_X.shape}')
    logger.debug('check_y.shape:{check_y.shape}')

    ori_o_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'has')
    ori_x_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'no')
    ori_dirs = [ori_x_path, ori_o_path]
    ori_X, ori_y = get_train_data(ori_dirs, reshape=resize_shape)
    logger.debug('ori_X.shape:{ori_X.shape}')
    logger.debug('ori_y.shape:{ori_y.shape}')


    X_train = np.append(ori_X, check_X, axis=0)
    y_train = np.append(ori_y, check_y, axis=0)


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
            test_size=0.15, random_state=1028)
    logger.debug('X_train.shape:{X_train.shape}')
    logger.debug('y_train.shape:{y_train.shape}')
    logger.debug('X_val.shape:{X_val.shape}')
    logger.debug('y_val.shape:{y_val.shape}')

    # 資料增強
    data_aug = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2),
    ]

    transform = transforms.Compose([
        transforms.RandomOrder(data_aug),
    ])

    # 訓練用的 Dataset及DataLoader    
    train_dataset = WearingDataset(X_train, y_train, transform=transform) # 有資料增強
    # train_dataset = WearingDataset(X_train, y_train) # 沒資料增強
    train_dataloader = DataLoader(
        train_dataset, batch_size=opt_batch_size, shuffle=True, num_workers=0)

    val_dataset = WearingDataset(X_val, y_val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=opt_batch_size, shuffle=True, num_workers=0)

    model = torch_model.get_train_shuffleNet_v2_model()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    BCE_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.004)
    
    early_stop = EarlyStopping(patience=8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=min_lr)

    history = History()

    # 網頁上進度條用的 callback
    class ProgressCallback():
        def __init__(self, total_steps):
            self.total_steps = total_steps
            self.progress = 0
            self.current_step = 0

        # def on_train_begin(self, logs={}):
        #     queue.put(0.0)
        
        def on_train_end(self, logs={}):
            self.progress = 100.0
            queue.put(str(self.progress))
            print('training done...')
            
        def on_batch_end(self, batch, logs={}):
            self.current_step += 1
            self.progress = round(self.current_step / self.total_steps*100, 2)
            while not queue.empty():
                queue.get(block=False)
            queue.put(str(self.progress))
    
    steps_per_epoch = len(X_train) // opt_batch_size if len(X_train) % opt_batch_size == 0 \
                        else len(X_train) // opt_batch_size + 1
    total_steps = opt_epoch * steps_per_epoch
    progress = ProgressCallback(total_steps)

    def train(tepoch):
        tepoch.set_description(f"Epoch {epoch+1}")
        for i, data in enumerate(train_dataloader):
            tepoch.update(1)            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device, dtype=dtype), data[1].to(device, dtype=dtype)
            labels = labels.reshape(-1, 1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)

            loss = BCE_loss(outputs, labels)
            loss.backward()

            optimizer.step()

            accuracy = binary_acc(outputs, labels)
            accuracy = float(accuracy)
            loss = float(loss)

            tepoch.set_postfix({
                "train_loss": loss, "train_acc": accuracy,
                "val_loss": .0, "val_acc": .0,
            })

            history.batch_train(loss, accuracy)
            progress.on_batch_end(i)

        return loss, accuracy

    def validation(model, loader):
        num_correct = 0
        num_samples = 0
        loss = 0.0
        all_y_true = []
        all_y_pred = []
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y_true = y.to(device=device, dtype=dtype).reshape(-1, 1)
                y_pred = model(x)      
                
                num = y_pred.size(0)
                loss += BCE_loss(y_pred, y_true) * num
                            
                y_pred = torch.round(y_pred)
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
                
                num_correct += (y_pred == y_true).sum()
                num_samples += num
        acc = float(num_correct) / num_samples
        loss = float(loss / num_samples)
        return acc, loss, (all_y_true, all_y_pred)


    logger.info(f'穿戴:模型優化訓練開始...')
    for epoch in range(opt_epoch):  # loop over the dataset multiple times    
        n_batches = len(train_dataloader)
                    
        with tqdm(total=n_batches, unit="batch") as tepoch:
            train_loss, train_acc = train(tepoch)
            
            val_acc, val_loss, preds = validation(model, val_dataloader)
            tepoch.set_postfix({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })
            
        # Logging
        history.epoch_train(n_batches)
        history.epoch_val(val_loss, val_acc)        

        # LR Scheduler
        if scheduler.__module__ == 'torch.optim.lr_scheduler':
            scheduler.step(val_loss)

        # Early Stopping
        if early_stop != None:            
            early_stop(val_loss)
            if early_stop.early_stop:
                break
    
    logger.info(f'穿戴:模型優化訓練完成')
    progress.on_train_end()

    
    logger.info(f'穿戴:刪除舊的模型及舊模型的預測照片...')
    logger.info(f'穿戴:儲存優化後的新模型...')
    logger.info(f'穿戴:將用來優化的照片加到original_training_data中...')
    # 模型訓練完成後，把原本的模型刪除，新的儲存回去
    os.remove(model_path)
    torch.save(model.state_dict(), model_path)

    # 把 check 資料夾的照片 加到 original_training_data 中
    # shutil.copytree(check_o_path, ori_o_path, dirs_exist_ok=True) # py >= 3.8
    # shutil.copytree(check_x_path, ori_x_path, dirs_exist_ok=True) # py >= 3.8
    movefiles(check_o_path, ori_o_path)
    movefiles(check_x_path, ori_x_path)

    # 把原先模型預測的資料夾都刪除掉    
    shutil.rmtree(os.path.join(PRED_SAVE_DIR_HEAD, part, f'{cls}_has'))
    shutil.rmtree(os.path.join(PRED_SAVE_DIR_HEAD, part, f'{cls}_no'))
    
    queue.close()
    logger.info(f'模型優化流程結束')


##### page: /wearing/model_train #####
def train_model_sub_process(part, cls, equd, unequd, queue, train_epoch, train_batch_size, min_lr, resize_shape):
    logger.info(f'穿戴:新模型訓練開始...')
    queue.put("data loading")    
    from . import torch_model
    from .torch_train_utils import EarlyStopping, History, binary_acc
    from torchvision import transforms
    from .torch_dataset import WearingDataset
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm

    device = torch.device('cuda')
    dtype = torch.float32


    model_path = os.path.join(SHUFFLE_NET_MODEL_DIR, part, f'{cls}.ptst')

    train_o_path = os.path.join(TMP_TRAIN_DIR, 'has')
    train_x_path = os.path.join(TMP_TRAIN_DIR, 'no')
    train_dirs = [train_x_path, train_o_path]
    X_train, y_train = get_train_data(train_dirs, reshape=resize_shape)
 
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
            test_size=0.15, random_state=1028)

    logger.debug('X_train.shape:{X_train.shape}')
    logger.debug('y_train.shape:{y_train.shape}')
    logger.debug('X_val.shape:{X_val.shape}')
    logger.debug('y_val.shape:{y_val.shape}')

    # 資料增強
    data_aug = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2),
    ]

    transform = transforms.Compose([
        transforms.RandomOrder(data_aug),
    ])

    # 訓練用的 Dataset及DataLoader    
    train_dataset = WearingDataset(X_train, y_train, transform=transform) # 有資料增強
    # train_dataset = WearingDataset(X_train, y_train) # 沒資料增強
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    val_dataset = WearingDataset(X_val, y_val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    
    model = torch_model.get_train_shuffleNet_v2_model()
    model = model.to(device)

    BCE_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.004)
    
    early_stop = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=min_lr)


    history = History()

    # 網頁上進度條用的 callback
    class ProgressCallback():
        def __init__(self, total_steps):
            self.total_steps = total_steps
            self.progress = 0
            self.current_step = 0

        # def on_train_begin(self, logs={}):
        #     queue.put(0.0)
        
        def on_train_end(self, logs={}):
            self.progress = 100.0
            queue.put(str(self.progress))
            print('training done...')
            
        def on_batch_end(self, batch, logs={}):
            self.current_step += 1
            self.progress = round(self.current_step / self.total_steps*100, 2)
            while not queue.empty():
                queue.get(block=False)
            queue.put(str(self.progress))

    steps_per_epoch = len(X_train) // train_batch_size if len(X_train) % train_batch_size == 0 \
                        else len(X_train) // train_batch_size + 1
    total_steps = train_epoch * steps_per_epoch
    progress = ProgressCallback(total_steps)

    def train(tepoch):
        tepoch.set_description(f"Epoch {epoch+1}")
        for i, data in enumerate(train_dataloader):
            tepoch.update(1)            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device, dtype=dtype), data[1].to(device, dtype=dtype)
            labels = labels.reshape(-1, 1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)

            loss = BCE_loss(outputs, labels)
            loss.backward()

            optimizer.step()

            accuracy = binary_acc(outputs, labels)
            accuracy = float(accuracy)
            loss = float(loss)

            tepoch.set_postfix({
                "train_loss": loss, "train_acc": accuracy,
                "val_loss": .0, "val_acc": .0,
            })

            history.batch_train(loss, accuracy)
            progress.on_batch_end(i)

        return loss, accuracy

    def validation(model, loader):
        num_correct = 0
        num_samples = 0
        loss = 0.0
        all_y_true = []
        all_y_pred = []
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y_true = y.to(device=device, dtype=dtype).reshape(-1, 1)
                y_pred = model(x)      
                
                num = y_pred.size(0)
                loss += BCE_loss(y_pred, y_true) * num
                            
                y_pred = torch.round(y_pred)
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
                
                num_correct += (y_pred == y_true).sum()
                num_samples += num
        acc = float(num_correct) / num_samples
        loss = float(loss / num_samples)
        return acc, loss, (all_y_true, all_y_pred)

    logger.info(f'穿戴:新模型訓練開始...')    
    for epoch in range(train_epoch):  # loop over the dataset multiple times    
        n_batches = len(train_dataloader)
                    
        with tqdm(total=n_batches, unit="batch") as tepoch:
            train_loss, train_acc = train(tepoch)
            
            val_acc, val_loss, preds = validation(model, val_dataloader)
            tepoch.set_postfix({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })
            
        # Logging
        history.epoch_train(n_batches)
        history.epoch_val(val_loss, val_acc)        

        # LR Scheduler
        if scheduler.__module__ == 'torch.optim.lr_scheduler':
            scheduler.step(val_loss)

        # Early Stopping
        if early_stop != None:            
            early_stop(val_loss)
            if early_stop.early_stop:
                break


    logger.info(f'穿戴:新模型訓練完成')
    progress.on_train_end()


    logger.info(f'穿戴:儲存訓練結果及模型...')
    logger.info(f'穿戴:將原本的original_training_data移除...')
    logger.info(f'穿戴:將用來訓練的照片加到original_training_data中...')
    # 儲存各種訓練結果
    result_save_dir = os.path.join(TRAIN_RESULT_DIR, part, cls)
    process_and_save_train_history(equd, unequd, history, preds, result_save_dir)

    # 模型訓練完成後，儲存下來
    torch.save(model.state_dict(), model_path)

    # 把原先的模型訓練資料移除 (original_training_data)
    ori_o_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'has')
    ori_x_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'no')

    if os.path.exists(ori_o_path): shutil.rmtree(ori_o_path)    #要先有目錄才能移除，不然會出錯
    if os.path.exists(ori_x_path): shutil.rmtree(ori_x_path)

    
    # 要重新創一個空白的資料夾，才能把訓練好的資料放進來
    os.makedirs(ori_o_path) 
    os.makedirs(ori_x_path)
    # 把 tmp_training_data 資料夾的照片 加到 original_training_data 中
    movefiles(train_o_path, ori_o_path)
    movefiles(train_x_path, ori_x_path)
    
    queue.close()
    logger.info(f'新模型訓練流程結束')


def process_and_save_train_history(equd, unequd, history, preds, result_save_dir):
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)

    ACCURACY_PNG = os.path.join(result_save_dir, 'accuracy.png')
    LOSS_PNG = os.path.join(result_save_dir, 'loss.png')
    METRICS_CSV = os.path.join(result_save_dir, 'val_metrics.csv')
    REPORT_TXT = os.path.join(result_save_dir, 'val_report.txt')
    OTHER_INFO_TXT = os.path.join(result_save_dir, 'other_info.txt')

    real_train_epoch = len(history.train_acc)
    print('real_train_epoch:', real_train_epoch)

    with open(OTHER_INFO_TXT, 'w', encoding='big5') as f:
        f.writelines(f'真正訓練Epoch數: {real_train_epoch}') # 因為可能會 early stop 所以不一定是一開始宣告的 Epoch數

    

    import matplotlib.pyplot as plt
    # 绘制训练 & 验证的准确率值
    plt.plot(history.train_acc)
    plt.plot(history.val_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.savefig(ACCURACY_PNG)
    plt.close()

    # 绘制训练 & 验证的损失值
    plt.plot(history.train_loss)
    plt.plot(history.val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'])
    plt.savefig(LOSS_PNG)

    from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
    y_true, y_pred = preds
    y_true = np.squeeze(y_true[0].tolist())
    y_pred = np.squeeze(y_pred[0].tolist())
    print('confusion_matrix')
    print(confusion_matrix(y_true, y_pred))

    print('\n report')
    report = classification_report(y_true, y_pred)
    print(report)

    with open(REPORT_TXT, 'w') as f:
        f.writelines(report)

    prfs = precision_recall_fscore_support(y_true, y_pred)

    import pandas as pd
    if(equd == "true" and unequd == "true"):
        metrics = pd.DataFrame(prfs, columns=['無裝備', '有裝備'],
                    index=['precison', 'recall', 'f1-score', 'support'])
    elif(equd == "true"):
        metrics = pd.DataFrame(prfs, columns=['有裝備'],
                    index=['precison', 'recall', 'f1-score', 'support'])
    elif(unequd == "true"):
        metrics = pd.DataFrame(prfs, columns=['無裝備'],
                    index=['precison', 'recall', 'f1-score', 'support'])

    if(equd == "true" or unequd == "true"):
        metrics.to_csv(METRICS_CSV, encoding='big5')
    else:
        logger.error(f'equd and unequd variable is None!')  


def get_tmp_train_check_filenames():
    # 取得 original data 的檔名，用來顯示在 模型訓練頁面(加載原始資料後)

    def get_data(path):
        files = set([f for f in os.listdir(path) if '.png' in f])
        data = [(remove_imagefile_extention(f),
                os.path.join(path, f)) for f in files]
        return data

    try:
        o_path = os.path.join(TMP_TRAIN_CHECK_DIR, 'has')
        o_data = get_data(o_path)
    except Exception as e:
        logger.error(f'穿戴:取得暫存的「有穿戴」檔案失敗, {e}')        
        o_data = []

    try:        
        x_path = os.path.join(TMP_TRAIN_CHECK_DIR, 'no')
        x_data = get_data(x_path)
    except Exception as e:        
        logger.error(f'穿戴:取得暫存的「沒穿戴」檔案失敗, {e}')
        x_data = []
    
    return sorted(o_data), sorted(x_data)
    # return o_data, x_data

def get_train_check_filenames(has_or_no):
    # 取得 tmp_training_data 資料夾中的檔名，用來顯示在 模型訓練頁面 的 最終待訓練資料(Total Train Data)
    def get_data(path):
        files = set([f for f in os.listdir(path) if '.png' in f])
        data = [(remove_imagefile_extention(f),
                os.path.join(path, f)) for f in files]
        return data


    if has_or_no == 'has':
        try:
            o_path = os.path.join(TMP_TRAIN_DIR, 'has')
            o_data = get_data(o_path)
        except Exception as e:            
            logger.error(f'穿戴:取得暫存的「有穿戴」檔案失敗, {e}')
            o_data = []
            
        return sorted(o_data)
    elif has_or_no == 'no':
        try:        
            x_path = os.path.join(TMP_TRAIN_DIR, 'no')
            x_data = get_data(x_path)
        except Exception as e:            
            logger.error(f'穿戴:取得暫存的「沒穿戴」檔案失敗, {e}')
            x_data = []

        return sorted(x_data)
    

def delete_tmp_train_check_dir_has():
    has_dir = os.path.join(TMP_TRAIN_CHECK_DIR, 'has')
    if os.path.isdir(has_dir):
        shutil.rmtree(has_dir)

def delete_tmp_train_check_dir_no():
    no_dir = os.path.join(TMP_TRAIN_CHECK_DIR, 'no')
    if os.path.isdir(no_dir):
        shutil.rmtree(no_dir)

def delete_tmp_train_check_dir():
    if os.path.isdir(TMP_TRAIN_CHECK_DIR):
        shutil.rmtree(TMP_TRAIN_CHECK_DIR)

def delete_tmp_train_dir():
    if os.path.isdir(TMP_TRAIN_DIR):
        shutil.rmtree(TMP_TRAIN_DIR)

def delete_tmp_train_dir_has():
    has_dir = os.path.join(TMP_TRAIN_DIR, 'has')
    if os.path.isdir(has_dir):
        shutil.rmtree(has_dir)

def delete_tmp_train_dir_no():
    no_dir = os.path.join(TMP_TRAIN_DIR, 'no')
    if os.path.isdir(no_dir):
        shutil.rmtree(no_dir)

def move_og_data_to_tmp_training_check(part, cls):
    og_has_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'has')
    og_no_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'no')

    def rename_function(filename):
        # return f'{part}_{cls}_{filename}' #資料重覆加載會有part & cls多疊的問題
        return f'{filename}'

    dest_has_dir = os.path.join(TMP_TRAIN_CHECK_DIR, 'has')
    dest_no_dir = os.path.join(TMP_TRAIN_CHECK_DIR, 'no')
    copyfiles(og_has_path, dest_has_dir, rename_function)
    copyfiles(og_no_path, dest_no_dir, rename_function)
    
def move_tmp_training_check_to_tmp_check(has_or_no):
    src_dir = os.path.join(TMP_TRAIN_CHECK_DIR, has_or_no)
    if os.path.isdir(src_dir):
        dest_dir = os.path.join(TMP_TRAIN_DIR, has_or_no)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        movefiles(src_dir, dest_dir)

# get data
def get_confs(part):
    return config.get_part_wearing_confs(part)

def get_part_clslist_dict():
    part_clssses = dict()
    for part in PART_LIST:
        part_clssses[part] = get_class_list(part)
    return part_clssses

def get_part_confs_dict():
    class_confs = dict()
    for part in PART_LIST:
        confs = [int(conf*100) for conf in get_confs(part)]
        class_confs[part] = confs
    return class_confs

def get_part_detect_class():
    part_detect_class = dict()
    for part in PART_LIST:
        part_detect_class[part] = config.get_detect_class(part)
    return part_detect_class


def get_detect_part_list():
    return config.get_detect_part()

def get_fence_enable():
    return config.get_fence_enable()

def get_sample_rate():
    return config.get_sampling_rate()

def get_max_quantity():
    return config.get_max_quantity()

def get_alarm_setting():
    result = config.get_alarm_setting()
    return [result.channel,result.mail_group,result.log_save_day]

def get_train_arg_dict():
    return config.get_train_arg_dict()

def get_train_data(dirs, reshape=None):
    # dirs: ['no_dir', 'has_dir'] 要照順序放 沒有裝備的照片的資料夾 跟 有裝備的照片的資料夾
    X,y = [],[]
    for label, dir in enumerate(dirs):
        if dir == '':
            continue
            
        files = Path(dir).rglob("*.[pP][nN][gG]")
        
        if reshape == None:
            tmp_x = np.array([cv2_read_zh_img(str(f)) for f in files])
        else:
            tmp_x = np.array([cv2.resize(cv2_read_zh_img(str(f)), reshape,
                                         interpolation=cv2.INTER_AREA) for f in files])
        if len(X):
            if len(tmp_x):
                X = np.append(X, tmp_x, axis=0)
                y = np.append(y, np.array([label]*len(tmp_x)), axis=0)
        else:
            X = tmp_x
            y = np.array([label]*len(X))
    
    return np.array(X), np.array(y)


# 取得 某個模型 的 原始訓練資料，有跟無的數量
def get_ori_data_num(part, cls):
    ori_o_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'has')
    ori_x_path = os.path.join(ORI_TRAIN_DIR_HEAD, part, cls, 'no')
    o_data_num = len(list(Path(ori_o_path).glob('*.png')))
    x_data_num = len(list(Path(ori_x_path).glob('*.png')))
    return [o_data_num, x_data_num]
    

# move or remove data
from pathlib import Path
def move2check(source):
    src_path = Path(source)
    dest_dir = os.path.join(src_path.parents[0], 'check')
    dest = os.path.join(dest_dir, src_path.name)    
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    shutil.move(source, dest)
    return dest # 回傳移動後的，新的檔案位置

def undo_move2check(source):
    src_path = Path(source)
    dest_dir = os.path.join(src_path.parents[1])
    dest = os.path.join(dest_dir, src_path.name)
    shutil.move(source, dest)
    return dest

def remove_file(path):
    os.remove(path)

def remove_from_check(source):
    src_path = Path(source)
    dest_dir = os.path.join(src_path.parents[0], 'check')
    dest = os.path.join(dest_dir, src_path.name)
    remove_file(dest)    
