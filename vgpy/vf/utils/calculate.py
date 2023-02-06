from ..config import config
import numpy as np

def get_bbox_middle_point(xyxy, IMG_WIDTH, IMG_HEIGHT):    
    ############# 計算Bonding box的中心點 #############
    #  (x1,y1) 代表Bbox左上的點 (x2,y2) 代表Bbox右下的點
    x1, y1, x2, y2 = [int(num) for num in xyxy]
    x1 -= 5
    y1 -= 5
    x2 += 5
    y2 += 5

    x1 = max(0, np.round(x1).astype('int32'))
    y1 = max(0, np.round(y1).astype('int32'))
    x2 = min(IMG_WIDTH, np.round(x2).astype('int32'))
    y2 = min(IMG_HEIGHT, np.round(y2).astype('int32'))
    box_middle_x = round((x1+x2)/2)
    box_middle_y = round((y1+y2)/2)
    # box_middle_point_list.append((box_middle_x, box_middle_y))
    return (box_middle_x, box_middle_y)


def init_fence_point_to_int(ptsArr):
    #####  將圍籬的點 轉為 int
    fences_int = []
    for fence_points in ptsArr:
        points = []
        # 將所有圍籬的所有點 轉成 int
        for fence_point in fence_points:
            fence_point = tuple(int(p) for p in fence_point) # 將圍籬的座標轉成整數
            points.append(fence_point)
        fences_int.append(points)
    return fences_int


def get_distance(p1, p2):
    tmp = np.array(p1) - np.array(p2)
    return np.sqrt(np.sum(np.square(tmp)))


def get_bbox_to_fence_shortest_distance_and_point(box_middle_point, fence_points, cam_wh):
    fence_m_x, fence_m_y = get_middle_point(fence_points) # 還要與圍籬中心點比較
    all_points = fence_points.copy()
    all_points.append((fence_m_x, fence_m_y))
    bbox_to_fence_distances = []
    for point in all_points:
        dist = get_distance(point, box_middle_point) # bbox中心 與 圍籬某點 的距離
        bbox_to_fence_distances.append(tuple([dist, point]))
    
    bbox_to_fence_distances = sorted(bbox_to_fence_distances, key=lambda x:x[0])
    bili_data = config.get_bili()
    bili = (bili_data['img_width']/cam_wh[0])*bili_data['bili']
    return bbox_to_fence_distances[0][0]*float(bili), bbox_to_fence_distances[0][1]


def get_middle_point(fence): # 取得某個圍籬的中心點
    area = 0.0
    x,y = 0.0,0.0
 
    a = len(fence)
    for i in range(a):
        #取得第i個點座標
        lat = fence[i][0] #x
        lng = fence[i][1] #y
#         print(lat,lng)
  
        #拿前一個點座標
        if i == 0:
            #拿最後一個點
            lat1 = fence[-1][0]#x
            lng1 = fence[-1][1]#y
 
        else:
            #拿第i-1個點
            lat1 = fence[i-1][0]#x
            lng1 = fence[i-1][1]#y
 
        fg = (lat*lng1 - lng*lat1)/2.0
 
        area += fg
        x += fg*(lat+lat1)/3.0
        y += fg*(lng+lng1)/3.0
 
    x = x/area
    y = y/area
 
    return round(x),round(y)
