""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
PRE_18_CHECKPOINT_PATH = "/home/zigangzhao/DMS/pytorch-cifar100/pretrain-model/resnet18-5c106cde.pth"
PRE_50_CHECKPOINT_PATH = "/home/zigangzhao/DMS/pytorch-cifar100/pretrain-model/resnet50-19c8e357.pth"
#total training epoches
EPOCH = 75
# MILESTONES = [60, 120, 160]
MILESTONES = [30, 40, 60] # [30, 50, 70]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

## tain data
TRAIN_DATA_PATH = "/code/zzg/project/seal_project/license_classification/dataset/train_data_1213_stitch/train"
TEST_DATA_PATH = "/code/zzg/project/seal_project/license_classification/dataset/train_data_1213_stitch/test"

##test_images_path
TSET_IMAGE = "/code/zzg/project/seal_project/license_classification/dataset/test_img/1114_test"

BASE = "/code/zzg/project/seal_project/license_classification/output/"







