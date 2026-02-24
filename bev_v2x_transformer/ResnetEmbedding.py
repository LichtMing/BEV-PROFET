import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
from torch.utils.data import Dataset,TensorDataset,DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import random

class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResnetModel(nn.Module):
    def __init__(self, BasicBlock, layers, d_model, in_feature):
        super(ResnetModel, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feature, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(BasicBlock, 64, layers[0], stride=2)
        self.layer2 = self.make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, layers[2], stride=2)
        self.fc = nn.Linear(256, d_model)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MapEmbeddingModel(nn.Module):
    def __init__(self, BasicBlock, layers, in_feature):
        super(MapEmbeddingModel, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feature, 64, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(BasicBlock, 64, layers[0], stride=2)
        self.layer2 = self.make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, 1024)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.reshape(out.shape[0], 1, 16, 64)
        return out