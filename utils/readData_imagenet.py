import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.cutout import Cutout
import sys
import os

# # 获取当前文件的绝对路径
# current_file_path = os.path.abspath(__file__)
# # 获取当前文件所在的目录
# current_dir = os.path.dirname(current_file_path)
# # 获取项目的根目录（即 utils 目录的上一级目录）
# project_root = os.path.dirname(current_dir)
# # print(current_dir)
# # 将项目根目录添加到 sys.path
# if project_root not in sys.path:
#     sys.path.append(project_root)
# print(sys.path)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 0
# 每批加载图数量
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2

def read_dataset(batch_size=16,valid_size=0.2,num_workers=0,pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
        # 需要更多数据预处理，自己查
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
        # 需要更多数据预处理，自己查
    ])

    train_data = datasets.ImageFolder(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\val', transform_test)
    # 读取数据
    valid_data = datasets.ImageFolder(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\train', transform_test)
    test_data = datasets.ImageFolder(r'D:\BaiduNetdiskDownload\mini-imagenet\mini-imagenet\train', transform_train)




    # # 将数据转换为torch.FloatTensor，并标准化。
    # train_data = datasets.CIFAR10(pic_path, train=True,
    #                             download=True, transform=transform_train)
    # valid_data = datasets.CIFAR10(pic_path, train=True,
    #                             download=True, transform=transform_test)
    # test_data = datasets.CIFAR10(pic_path, train=False,
    #                             download=True, transform=transform_test)
        

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader

