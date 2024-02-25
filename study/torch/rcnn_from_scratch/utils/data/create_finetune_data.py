'''

'''
import time
import shutil
import numpy as np
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import selectivesearch
from util import check_dir, parse_car_csv, parse_xml, compute_ious

# train
# positive num: 66517
#negative num: 464340
# val
# positive num: 64712
#negative num: 415134


def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    '''
    양성 및 음성 샘플 가져오기
    - difficult가 True인 상자는 무시
    - 양성 : selectivesearch로 생성된 proposal과 정답 박스간의 iou가 0.5 이상
    - 음성 : 0~0.5미만, 음성 개수가 너무 많은 것을 방지하기 위해 
            proposal 크기가 정답 박스 크기의 1/5 이상인 것만 음성
    
    # TODO_1 selectivesearch로 N개의 proposal을 생성한다.
    # TODO_2 정답 박스 중 가장 큰 박스의 크기를 구한다.
    # TODO_3 N개의 proposal에 대해 iou값을 구한다.
    # TODO_4 각 proposal들이 
            iou값이 0.5 이상이면 positive로, 
            0.5미만이고 가장 큰 박스 크기의 1/5보다 클 때 negative로 분류된다.
    
    return : proposal로 구성된 pos list, neg list
    '''
    img = cv2.imread(jpeg_path)
    selectivesearch.config(gs, img, strategy='q')
    rects = selectivesearch.get_rects(gs)
    bndboxs = parse_xml(annotation_path)
    
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax-ymin) * (xmax-xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size
    
    iou_list = compute_ious(rects, bndboxs)
    
    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)
        iou_score = iou_list[i]
        
        if iou_list[i] >= 0.5:
            positive_list.append(rects[i])
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            negative_list.append(rects[i])
        else:
            pass
    
    return positive_list, negative_list
    

if __name__ == '__main__':
    car_root_dir = '../../data/voc_car'
    finetune_root_dir = '../../data/finetune_car/'
    check_dir(finetune_root_dir)
    
    gs = selectivesearch.get_selective_search()
    for name in ['train', 'val']:
        # src 경로
        src_root_dir = os.path.join(car_root_dir, name)
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        # dst 경로
        dst_root_dir = os.path.join(finetune_root_dir, name)
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        # dst 디렉터리 생성
        check_dir(dst_root_dir)
        check_dir(dst_jpeg_dir)
        check_dir(dst_annotation_dir)
        # car.csv 파일 복사
        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)
        
        total_num_positive = 0
        total_num_negative = 0     
        samples = parse_car_csv(src_root_dir)
        for sample_name in samples:
            since = time.time()
            
            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')
            print(src_jpeg_path)
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)
            
            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')
            
            shutil.copyfile(src_jpeg_path, dst_jpeg_path)
            # ../../data/finetune_car/train/Annotations/{sample_name}_1.csv 파일에 (m)개의 positive_list
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            # ../../data/finetune_car/train/Annotations/{sample_name}_0.csv 파일에 (n-m)개의 negative_list
            # 만약 m == n 이라면 negative_list가 빈 파일이 될 것이다.
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')
            
            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        
        print('%s positive num: %d' %(name, total_num_positive))
        print('%s negative num: %d' %(name, total_num_negative))
    print('done')
