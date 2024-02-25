'''
VOC2007 dataset 다운로드
'''
import cv2
import numpy as np
from torchvision.datasets import VOCDetection


if __name__ == '__main__':
    dataset = VOCDetection('../../data', year='2007', image_set='trainval', download=True)
    
    img, target = dataset.__getitem__(1000)
    img = np.array(img)
    
    print(img.shape)
    print(target)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)