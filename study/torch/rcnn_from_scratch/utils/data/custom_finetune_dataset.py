import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from utils.util import parse_car_csv


class CustomFinetuneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        '''
        # TODO_1 finetune_car/train/car.csv에 있는 경로 목록을 읽음
        # TODO_2 읽어들인 모든 샘플에 대해 _1.csv와 _0.csv를 붙인 path를 각각 저장함 
        # TODO_3 pos/neg 샘플의 크기와 상자 좌표들을 저장한다. 
        '''
        
        # TODO_1 finetune_car/train/car.csv에 있는 경로 목록을 읽음
        samples = parse_car_csv(root_dir)
        # ../../data/finetune_car/train/JPEGImages/{sample_name}.jpg
        jpeg_images = [cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg")) 
                       for sample_name in samples]
        
        # TODO_2 읽어들인 모든 샘플에 대해 _1.csv와 _0.csv를 붙인 path를 각각 저장함 
        positive_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_1.csv') 
                                for sample_name in samples]
        negative_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_0.csv') 
                                for sample_name in samples]    
    
        # TODO_3 pos/neg 샘플의 크기와 상자 좌표들을 저장한다.
        # 상자 크기
        positive_sizes = list()
        negative_sizes = list()
        # 상자 좌표
        positive_rects = list()
        negative_rects = list()
        for annotation_path in positive_annotations:
            # csv 파일 읽기
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            
            if len(rects.shape) == 1:  # 파일에 한 줄만 있거나 비어있는 경우
                if rects.shape[0] == 4:  # 한 줄
                    positive_rects.append(rects)
                    positive_sizes.append(1)
                else:  # 비어있는 경우
                    positive_sizes.append(0)
            else:  # 여러 줄 있는 경우
                positive_rects.extend(rects)
                positive_sizes.append(len(rects))
        
        for annotation_path in negative_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')

            if len(rects.shape) == 1: 
                if rects.shape[0] == 4: 
                    negative_rects.append(rects)
                    negative_sizes.append(1)
                else:
                    negative_sizes.append(0) 
            else:
                negative_rects.extend(rects)
                negative_sizes.append(len(rects))
        
        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_sizes = positive_sizes
        self.negative_sizes = negative_sizes
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects
        self.total_positive_num = int(np.sum(positive_sizes))
        self.total_negative_num = int(np.sum(negative_sizes))
    
    def __getitem__(self, index: int):
        '''
        데이터셋 구조는 아래와 같이 생각할 수 있다.
        [positive_num | negative_num]
        
        idx < total_positive_num 이면 즉, 경계선보다 왼쪽이면 positive sample을
        반대쪽이라면 negative sample을 뽑아서
        그 샘플에 해당하는 경계 박스로 image를 크롭한 image와
        positive인지 negative인지를 표시하는 0, 1를 target으로 하여 반환한다.
        
        return: image, target
        '''
        image_id = len(self.jpeg_images) - 1
        if index < self.total_positive_num:
            target = 1
            xmin, ymin, xmax, ymax = self.positive_rects[index]
            for i in range(len(self.positive_sizes)-1):
                # np.sum(positive_sizes[:i])는 (i-1)index까지의 positive sample 개수를 뜻한다.
                if np.sum(self.positive_sizes[:i]) <= index < np.sum(self.positive_sizes[:(i+1)]):
                    image_id = i
                    break
            # 부분 이미지 crop
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        else:
            target = 0
            idx = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[idx]
            for i in range(len(self.negative_sizes) - 1):
                if np.sum(self.negative_sizes[:i]) <= idx < np.sum(self.negative_sizes[:(i+1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def __len__(self) -> int:
        return self.total_positive_num + self.total_negative_num
    
    def get_positive_num(self) -> int:
        return self.total_positive_num
    
    def get_negative_num(self) -> int:
        return self.total_negative_num


def test(idx):
    root_dir = '../../data/finetune_car/train'
    train_data_set = CustomFinetuneDataset(root_dir)
    
    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())
    
    image, target = train_data_set.__getitem__(idx)
    print('target: %d' % target)
    
    image = Image.fromarray(image)
    print(image)
    print(type(image))
    
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    
if __name__ == '__main__':
    # test(159622)
    # test(4051)
    test(24768)