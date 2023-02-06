import sys
import os

# op_dir_path = 'D:\\Boris\\pyPackage\\openpose_cp38'
# build_path = os.path.join(op_dir_path, 'build')
# sys_path = os.path.join(build_path, 'python', 'openpose', 'Release')
# sys.path.append(sys_path)

# os_env_path = build_path + '/x64/Release;' +  build_path + '/bin;'
# os.environ['PATH']  = os.environ['PATH'] + ';' + os_env_path

op_dir = os.path.dirname(os.path.abspath(__file__))+ '\\openpose\\build'  #取得本py檔案的路徑+\\openpose\\build
# sys.path.append(".\\vgpy\\utils\\openpose\\build\\python\\openpose\\Release")
sys.path.append(os.path.join(op_dir, 'python', 'openpose', 'Release'))  # 新增import時搜尋的路徑
# os.environ['PATH']  = os.environ['PATH'] + ';' + '.\\vgpy\\utils\\openpose\\build\\x64\\Release;'+ ';' + '.\\vgpy\\utils\\openpose\\build\\bin;'
os.environ['PATH']  = os.environ['PATH'] + ';' + os.path.join(op_dir, 'x64', 'Release')+ ';' + os.path.join(op_dir, 'bin')

import pyopenpose as op


# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
# params["model_folder"] = os.path.join(op_dir_path, 'models')
params["model_folder"] = ".\\vgpy\\utils\\openpose\\models"
params["net_resolution"] = "320x176"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def body_from_image(img):
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.cvOutputData, datum.poseKeypoints


### test openpose code



