import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import os
import bisect
import random


class TrainDataset(Dataset):
    def __init__(self, train_idx):
        train_file = os.listdir("./BEVData_01")
        mask_file = os.listdir("./BEVMaskData_01")
        label_file = os.listdir("./BEVLabel_01")
        train_file = sorted(train_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1 : x.find(".")])))
        mask_file = sorted(mask_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1 : x.find(".")])))
        label_file = sorted(label_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1 : x.find(".")])))
        self.train_file_list = [train_file[j] for j in train_idx]
        self.mask_file_list = [mask_file[j] for j in train_idx]
        self.label_file_list = [label_file[j] for j in train_idx]
        self.len = len(self.train_file_list)
        print(self.train_file_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 读取三个文件夹里的同名文件
        train_npy = np.load("./BEVData_01/" + self.train_file_list[idx])
        mask_npy = np.load("./BEVMaskData_01/" + self.mask_file_list[idx])
        label_npy = np.load("./BEVLabel_01/" + self.label_file_list[idx])
        
        # 转换成模型认识的格式 (Float32)
        train_arr = torch.from_numpy(np.array(train_npy, dtype=np.float32))
        mask_arr = torch.from_numpy(np.array(mask_npy, dtype=np.uint8)).unsqueeze(-1) # 增加一个维度
        label_arr = torch.from_numpy(np.array(label_npy, dtype=np.float32))
        
        return train_arr, mask_arr, label_arr


class TestDataset(Dataset):
    def __init__(self, test_idx):
        test_file = os.listdir("./BEVData_01")
        mask_file = os.listdir("./BEVMaskData_01")
        label_file = os.listdir("./BEVLabel_01")
        test_file = sorted(test_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1 : x.find(".")])))
        mask_file = sorted(mask_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1 : x.find(".")])))
        label_file = sorted(label_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1 : x.find(".")])))
        self.test_file_list = [test_file[j] for j in test_idx]
        self.mask_file_list = [mask_file[j] for j in test_idx]
        self.label_file_list = [label_file[j] for j in test_idx]
        self.len = len(self.test_file_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        test_npy = np.load("./BEVData_01/" + self.test_file_list[idx])
        mask_npy = np.load("./BEVMaskData_01/" + self.mask_file_list[idx])
        label_npy = np.load("./BEVLabel_01/" + self.label_file_list[idx])
        test_arr = np.array(test_npy, dtype=np.float32)
        mask_arr = np.array(mask_npy, dtype=np.uint8)
        label_arr = np.array(label_npy, dtype=np.float32)
        test_arr = torch.from_numpy(test_arr)
        mask_arr = torch.from_numpy(mask_arr)
        label_arr = torch.from_numpy(label_arr)

        return test_arr, mask_arr.unsqueeze(-1), label_arr


if __name__ == '__main__':
    test_file = os.listdir("./BEVData_01")
    mask_file = os.listdir("./BEVMaskData_01")
    label_file = os.listdir("./BEVLabel_01")
    test_file = sorted(test_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1: x.find(".")])))
    mask_file = sorted(mask_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1: x.find(".")])))
    label_file = sorted(label_file, key=lambda x: (int(x[:x.find("_")]), int(x[x.find("_") + 1: x.find(".")])))
    print(test_file.index("5_23000.npy"))