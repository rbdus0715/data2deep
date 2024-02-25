'''
# car 클래스에 해당하는 데이터만 추출하여
# ../../data/voc_car 위치에 car 데이터셋 만들기
'''
import os, sys
import shutil
import random
import numpy as np
import xmltodict
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util import check_dir

suffix_xml = '.xml'
suffix_jpeg = '.jpg'

car_train_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
car_val_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'

voc_annotation_dir = '../../data/VOCdevkit/VOC2007/Annotations/'
voc_jpeg_dir = '../../data/VOCdevkit/VOC2007/JPEGImages/'

car_root_dir = '../../data/voc_car/'


def parse_train_val(data_path):
    '''
    car_train.txt (ex) 파일을 읽는다.
    000005 -1
    000009 1 
    ...
    'car' class에 맞는 이미지 번호만 추출하여 리스트형태로 반환한다.
    '''
    samples = []
    
    with open(data_path, 'r') as file:
        lines = file.readlines()
        # line : ex) 000005 -1
        for line in lines:
            res = line.strip().split(' ')
            # 0. 1인 경우 split 결과가 3이다.
            # 그 중에서도 클래스에 속한다면 sample에 추가한다.
            if len(res) == 3 and int(res[2]) == 1:
                samples.append(res[0])
    
    return np.array(samples)


def sample_train_val(samples):
    '''
    10개만 랜덤으로 샘플링
    '''
    for name in ['train', 'val']:
        dataset = samples[name]
        length = len(dataset)
        
        # 10개 랜덤으로 샘플링
        random_samples = random.sample(range(length), int(length/10)) 
        new_dataset = dataset[random_samples]
        samples[name] = new_dataset
    
    return samples


def  save_car(car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    '''
    (1) car 데이터셋 만들기
    car 이미지 번호만 샘플링한 list를 이용해
    VOC2007/JPEGImages와 VOC2007/Annotations에서 car 데이터를 
    voc_car/train/JPEGImages/ voc_car/train/Annotations으로 옮긴다.
    
    (2) 이미지 번호를 car.csv에 따로 저장
    '''
    for sample_name in car_samples:
        # src: source, dst: destination
        src_jpeg_path = os.path.join(voc_jpeg_dir, sample_name + suffix_jpeg)
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpeg)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)
        
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml)
        shutil.copyfile(src_annotation_path, dst_annotation_path)
        
    csv_path = os.path.join(data_root_dir, 'car.csv')
    # 컴마로 구분된 list는 '\n'으로 구분되게 된다.
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')


if __name__ == '__main__':
    '''
    (1) 아래와 같은 폴더를 만든다.
    voc_car/train
    |-- JPEGImages
    |-- Annotations
    
    (2) car 데이터만 샘플링해서 voc_car/ 디렉터리에 car 데이터셋 만들기
    
    '''
    samples = {
        'train': parse_train_val(car_train_path), 
        'val': parse_train_val(car_val_path)
    }
    print(samples)
    # 예시로 10개만 샘플링해보기
    # samples = sample_train_val(samples)
    # print(samples)
    
    check_dir(car_root_dir)
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name) # voc_car/train
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')
        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
    
        check_dir(data_root_dir)        
        check_dir(data_jpeg_dir)
        check_dir(data_annotation_dir)
        
        save_car(
            samples[name],
            data_root_dir,
            data_annotation_dir,
            data_jpeg_dir
        )
    
    print('done')
