'''
torch의 pre-trained된 alexnet 모델을 fine tuning 시킨다.
데이터셋은 CustomFinetuneDataset을 통해 구축하였고, 
CustomBatchSampler을 통해 pos/neg 샘플 비율을 맞추었다.
'''
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from utils.data.custom_finetune_dataset import CustomFinetuneDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir


def load_data(data_root_dir):
    '''
    train, val 데이터로더를 각각 만든다.
    이때 CustomBatchSampler를 통해 ositive 샘플을 32개, negative 샘플을 96개로 하여
    총 128개가 하나의 미니배치가 되도록 구성한다.
    '''
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'var']:
        data_dir = os.path.join(data_root_dir, name)
        data_set = CustomFinetuneDataset(data_dir, transform=transform)
        data_sampler = CustomBatchSampler(
            data_set.get_positive_num(),
            data_set.get_negative_num(),
            32, 96
        )
        data_loader = DataLoader(
            data_set,
            batch_size=128,
            sampler=data_sampler,
            num_workers=8,
            drop_last=True
        )
        
        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()
    
    return data_loaders, data_sizes


def train_model(data_loaders,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                num_epochs=25,
                device=None):
    '''
    각 epoch마다 validation을 함으로써 best_model을 계속해서 갱신해나간다.
    가장 성능이 좋았던 epoch의 가중치를 탑재한 모델을 return한다.
    '''
    since = time.time()
    
    # pre-trained된 alexnet 가중치
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels, in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # 값, 정답인덱스
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                lr_scheduler.step()
            
            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_weights)
    return model
    
    
if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_loaders, data_sizes = load_data('./data/finetune_car')
    model = models.alexnet(pretrained=True)
    # print(model)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    # print(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_model = train_model(
        data_loaders,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        num_epochs=25,
        device=device
    )
    
    check_dir('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.path')