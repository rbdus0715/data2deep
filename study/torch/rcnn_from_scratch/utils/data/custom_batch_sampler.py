'''
positive/negative 샘플들의 총 개수가 주어질 때 
pos 몇 개, neg 몇 개를 골라서 하나의 미니배치로 만들 수 있도록 하는 sampler
'''
import numpy as np
import random
from torch.utils.data import Sampler, DataLoader
import torchvision.transforms as transforms

# test 하는데 사용
from utils.data.custom_finetune_dataset import CustomFinetuneDataset


class CustomBatchSampler(Sampler):
    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative
        
        # 데이터 전체 길이
        length = num_positive + num_negative
        self.idx_list = list(range(length))
        # 배치사이즈
        self.batch = batch_negative + batch_positive
        # iteration 몇 번 하는지
        self.num_iter = length // self.batch
    
    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate(
                # np.concatenate((a, b)) 꼴
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)
    
    def __len__(self) -> int:
        return self.num_iter * self.batch
    
    def get_num_batch(self) -> int:
        return self.num_iter

    
def test():
    root_dir = '../../data/finetune_car/train'
    train_data_set = CustomFinetuneDataset(root_dir)
    train_sampler = CustomBatchSampler(
        train_data_set.get_positive_num(),
        train_data_set.get_negative_num(),
        32, 96
    )
    
    print('sampler len: %d' % train_sampler.__len__())
    print('sampler batch num: %d' % train_sampler.get_num_batch())
    first_idx_list = list(train_sampler.__iter__())[:128]
    print(first_idx_list)
    print('positive batch: %d' % np.sum(np.array(first_idx_list) < 66517))
    
    
if __name__ == '__main__':
    test()