"""
code by zzg 2020-06-12
"""

import sys
import numpy
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from . import settings
from .auto_augment import AutoAugment, Cutout


def get_training_dataloader(batch_size=16, num_workers=2, shuffle=True, auto_augment=False, cutout=False):
    """ return training dataloader
    Args:
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """
    transform_train = [
            transforms.Resize([224, 320]),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5)
        ]
    if auto_augment:
            transform_train.append(AutoAugment())
    if cutout:
            transform_train.append(Cutout())

    transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    transform_train = transforms.Compose(transform_train)
    
    '''
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]), # transforms.Resize([h, w]) pinie 224*320
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        # transforms.Normalize((0.6786, 0.6511, 0.6284), (0.1992, 0.2055, 0.2068)) #-own
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #-cifar10
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #- imagenet
    ])
    '''
    train_data = torchvision.datasets.ImageFolder(root=settings.TRAIN_DATA_PATH, transform=transform_train)
    #对应文件夹的label
    print(train_data.class_to_idx)
    print(train_data[0][0].shape)
    #所有图片的路径和对应的label
    # print(train_data.imgs)
    trainloader = data.DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
  
    return trainloader 

def get_test_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """
    # default = 224
    transform_test = transforms.Compose([
     transforms.Resize([224, 320]), #注意testset不要randomresize h*w=224*320
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #-cifar10
     #transforms.Normalize((0.6786, 0.6511, 0.6284), (0.1992, 0.2055, 0.2068))#-own
     #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #- imagenet
     ])

    test_data = torchvision.datasets.ImageFolder(root=settings.TEST_DATA_PATH, transform=transform_test)
    print(test_data.class_to_idx)
    testloader = data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return testloader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
        print("====", channels_sum)

    print(num_batches)
    print(channels_sum)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) ** 0.5

    return mean, std

    
if __name__ == "__main__":

    ## 计算自己数据集的均值和方差 
    train_loader = get_training_dataloader(batch_size=64, num_workers=2, shuffle=True)
    mean, std = get_mean_std(train_loader)
    print(mean, std)



