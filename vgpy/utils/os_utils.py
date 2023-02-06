import os
from pathlib import Path
import shutil

def movefiles(src_path, dest_path):
    # 會把 src_path 中的所有檔案都 "移動" 到 dest_path
    for file in Path(src_path).glob('*.*'): # grabs all files
        file.rename(os.path.join(dest_path, file.name))

def copyfiles(src_path, dest_path, rename_func=None):
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    # 會把 src_path 中的所有檔案都 "複製" 到 dest_path
    for file in Path(src_path).glob('*.*'): # grabs all files
        if rename_func is not None:
            shutil.copyfile(str(file), os.path.join(dest_path, rename_func(file.name)))
        else:
            shutil.copyfile(str(file), dest_path)
          

