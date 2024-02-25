import os
import numpy as np
import xmltodict
import torch
import matplotlib.pyplot as plt


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, 'car.csv')
    samples = np.loadtxt(csv_path, dtype=np.str)
    return samples


def parse_xml(xml_path):
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        # print(xml_dict)
        
        bndboxs = list()
        objects = xml_dict['annotation']['object']
        # 여러 박스 오브젝트가 있는 경우 -> for문
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj['name']
                difficult = int(obj['difficult'])
                if 'car'.__eq__(obj_name) and difficult != 1:
                    bndbox = obj['bndbox']
                    bndboxs.append(
                        (int(bndbox['xmin']),
                         int(bndbox['ymin']),
                         int(bndbox['xmax']),
                         int(bndbox['ymax']),)
                    )
        # 하나의 박스 오브젝트밖에 없는 경우
        elif isinstance(objects, dict):
            obj_name = objects['name']
            difficult = int(objects['difficult'])
            if 'car'.__eq__(obj_name) and difficult != 1:
                bndbox = objects['bndbox']
                bndboxs.append(
                    (int(bndbox['xmin']),
                     int(bndbox['ymin']),
                     int(bndbox['xmax']),
                     int(bndbox['ymax']),)
                )
        else:
            pass
        
        return np.array(bndboxs)


def iou(pred_box, target_box):
    '''
    예측 박스 [4] 1개와 정답 박스 [N, 4] N개 간의 iou
    결과 : [N] 개의 iou score
    '''
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]
    
    # 왼쪽위 꼭짓점 좌표
    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    # 오른쪽 아래 꼭짓점 좌표
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])
    
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxAarea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBarea = (target_box[:, 2] - target_box[:, 0]) * (target_box[:, 3] - target_box[:, 1])
    
    scores = intersection / (boxAarea + boxBarea - intersection)
    return scores


def compute_ious(rects, bndboxs):
    iou_list = list()
    for rect in rects:
        scores = iou(rect, bndboxs)
        # 가장 많이 겹쳤을 때의 iou값만 추가
        iou_list.append(max(scores))
    return iou_list