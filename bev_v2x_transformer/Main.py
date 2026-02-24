from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import linear, pad, softmax, dropout
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
from TemporalSpatialAttention import Encoder, Decoder
from ResnetEmbedding import BasicBlock, ResnetModel, MapEmbeddingModel
from UpSample import UP
import random
import numpy as np
import cv2
from PosEmbedding import TimeEmbeddingSine, TimeEmbeddingLearnable
import os
import time
from data_loader_v4 import TrainDataset

device = torch.device("cuda:0")
class MainModel(nn.Module):
    def __init__(self, d_head=4, d_model=64, d_hid=128, in_feature=6, img_size=72, d_out = 16 * 64, whole_scale = 680 * 4, crop_h = 20 * 2, crop_w = 34 * 2):
        super(MainModel, self).__init__()
        self.resnet_embedding = ResnetModel(BasicBlock, [2, 2, 2, 2], d_model=d_model, in_feature=in_feature)
        self.map_embedding = MapEmbeddingModel(BasicBlock, [2, 2, 2, 2], in_feature=2)
        self.encoder = Encoder(d_head, d_model, d_hid)
        self.decoder = Decoder(d_head, d_model, d_hid)
        self.vehicle_linear = nn.Linear(d_out, whole_scale)
        self.road_linear = nn.Linear(d_out, whole_scale)
        self.temporal_embed = TimeEmbeddingLearnable()
        self.up1 = UP(5, 5, True)
        self.up2 = UP(5, 5, True)
        self.in_feature = in_feature
        self.img_size = img_size
        self.d_out = d_out
        self.whole_scale = whole_scale
        self.crop_h = crop_h
        self.crop_w = crop_w

    def forward(self, x, key_padding_mask, map):
        re_x = x.reshape(-1, self.in_feature, self.img_size, self.img_size)
        embed = self.resnet_embedding(re_x)
        map_embed = self.map_embedding(map)
        embed = embed.reshape(x.shape[0], x.shape[1], x.shape[2], embed.shape[-1])
        key_padding_mask = key_padding_mask.bool()
        idx = torch.arange(7).reshape(1, 7, 1, 1).repeat(x.shape[0], 1, x.shape[2], 1).to(device)
        temporal_pos_embed = self.temporal_embed(idx)
        encoder_out = self.encoder(embed, key_padding_mask, temporal_pos_embed[:, :7, :, :], map_embed)[0]
        #newly added
        encoder_out = encoder_out[:, -4:, :, :]

        outputs = torch.zeros(encoder_out.shape[0], 3, encoder_out.shape[2], encoder_out.shape[3]).to(device)
        attn_weights = torch.zeros(encoder_out.shape[0], 3, encoder_out.shape[2], encoder_out.shape[2]).to(device)
        hidden_out = encoder_out
        key_padding_mask = key_padding_mask[:, -1:, :, :].repeat(1, 4, 1, 1)
        for t in range(3):
            decoder_out, dec_spa_attn_list, dec_tem_attn_list = self.decoder(encoder_out, hidden_out, key_padding_mask, temporal_pos_embed[:, t:t+4, :, :],
                                                                             temporal_pos_embed[:, t:t+4, :, :], map_embed)
            decoder_out += encoder_out
            outputs[:, t:t + 1, :, :] = decoder_out[:, -1:, :, :]
            attn_weights[:, t:t + 1, :, :] = dec_spa_attn_list[-1][:, -1:, :, :]
            encoder_out = decoder_out
            #changed
            hidden_out = self.encoder(encoder_out, key_padding_mask[:, :4], temporal_pos_embed[:, t+1:t+5, :, :], map_embed)[0]

        vehicle_out = self.vehicle_linear(outputs.reshape(outputs.shape[0], outputs.shape[1], self.d_out))
        road_out = self.road_linear(outputs[:, -2:, :, :].reshape(outputs.shape[0], 2, self.d_out))
        all_out = torch.cat([road_out, vehicle_out], 1)
        all_out = all_out.reshape(all_out.shape[0], all_out.shape[1], self.crop_h, self.crop_w)
        up1_out = self.up1(all_out)
        up2_out = self.up2(up1_out)
        out = torch.sigmoid(up2_out)
        return out, attn_weights

