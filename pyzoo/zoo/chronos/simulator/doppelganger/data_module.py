import os
import pickle
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from .util import add_gen_flag, normalize_per_sample, renormalize_per_sample


class DoppelGANgerDataModule(LightningDataModule):
    '''
    Note that for now, we will still follow the Dataset format stated in
    https://github.com/fjxmlzn/DoppelGANger#dataset-format. In other words,
    the data_dir should be set to a dir with following three files.
    1. data_feature_output.pkl (change this by setting DATA_FEATURE_OUTPUT_FILENAME)
    2. data_attribute_output.pkl (change this by setting DATA_ATTRIBUTE_OUTPUT_FILENAME)
    3. data_train.npz (change this by setting DATA_TRAIN_FILENAME)

    Please notice that this module can not work alone without doppelganger_torch.
    '''
    def __init__(self,
                 sample_len,
                 real_data,
                 feature_outputs,
                 attribute_outputs,
                 batch_size=32):
        super().__init__()
        self.sample_len = sample_len
        self.batch_size = batch_size

        # load data from data_dir
        # ===================================================================================
        data_all = real_data['data_feature']
        data_attribute = real_data['data_attribute']
        data_gen_flag = real_data['data_gen_flag']
        data_feature_outputs = feature_outputs
        data_attribute_outputs = attribute_outputs
        self.num_real_attribute = len(data_attribute_outputs)
        self.num_feature_dim = len(data_feature_outputs)

        # print loaded data basic status
        print("------------loaded data-------------")
        print("====================================")
        print("data_all type:", type(data_all), "shape:", data_all.shape)
        print("data_attribute type:", type(data_attribute), "shape:", data_attribute.shape)
        print("data_gen_flag type:", type(data_gen_flag), "shape:", data_gen_flag.shape)
        print("data_feature_outputs:", data_feature_outputs)
        print("data_attribute_outputs:", data_attribute_outputs)
        print("num_feature_dim:", self.num_feature_dim)
        print("num_real_attribute:", self.num_real_attribute)

        # normalize data (use this only if you want to use additional attribute(max, min))
        # actually, no additional attribute has not been fully tested now
        # ===================================================================================
        (data_feature, data_attribute, data_attribute_outputs,
            real_attribute_mask) = normalize_per_sample(
            data_all, data_attribute, data_feature_outputs,
            data_attribute_outputs)
        
        # add generation flag to features
        # ===================================================================================
        data_feature, data_feature_outputs = add_gen_flag(
            data_feature, data_gen_flag, data_feature_outputs, self.sample_len)
        print("")
        print("------------processed data------------")
        print("======================================")
        print("data_feature shape(after processing):", data_feature.shape)
        print("data_attribute shape(after processing):", data_attribute.shape)
        print("data_gen_flag shape(after processing):", data_gen_flag.shape)

        # will be used in model init
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.real_attribute_mask = real_attribute_mask

        # prepare input meta data
        # ===================================================================================
        total_generate_num_sample = data_feature.shape[0]
        if data_feature.shape[1] % self.sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        self.length = int(data_feature.shape[1] / self.sample_len)

        # will be used in dataset init
        self.data_feature = data_feature
        self.data_attribute = data_attribute

        self.gen_flag_dims = []

        dim = 0
        for output in self.data_feature_outputs:
            if output.is_gen_flag:
                if output.dim != 2:
                    raise Exception("gen flag output's dim should be 2")
                self.gen_flag_dims = [dim, dim + 1]
                break
            dim += output.dim
        if len(self.gen_flag_dims) == 0:
            raise Exception("gen flag not found")

    def train_dataloader(self):
        self.data_feature = torch.from_numpy(data_feature).float()
        self.data_attribute = torch.from_numpy(data_attribute).float()
        dataset = CustomizedDataset(self.data_feature,
                                    self.data_attribute)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

class CustomizedDataset(Dataset):
    def __init__(self,
                 data_feature,
                 data_attribute):
        self.data_feature = data_feature
        self.data_attribute = data_attribute
    
    def __len__(self):
        return self.data_feature.shape[0]
    
    def __getitem__(self, index):
        return self.data_feature[index],\
               self.data_attribute[index]

if __name__ == "__main__":
    dm = DoppelGANgerDataModule(sample_len=10,
                                data_dir="/home/cpx/junweid/doppelganger-pytorch/data/WWT")
    print("Done")
