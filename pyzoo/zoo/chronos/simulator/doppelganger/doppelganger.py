import numpy as np
from tqdm import tqdm
import datetime
import os
import math
import sys
import torch
import torch.nn as nn
from .network import Discriminator, AttrDiscriminator, DoppelGANgerGenerator, RNNInitialStateType


class DoppelGANger(nn.Module):
    def __init__(self,
                 data_feature_outputs,
                 data_attribute_outputs,
                 real_attribute_mask,
                 sample_len,
                 L_max,
                 num_packing=1,
                 # discriminator parameters
                 discriminator_num_layers=5, discriminator_num_units=200,
                 # attr_discriminator parameters
                 attr_discriminator_num_layers=5, attr_discriminator_num_units=200,
                 # generator parameters
                 attribute_num_units=100, attribute_num_layers=3,
                 feature_num_units=100, feature_num_layers=2,
                 attribute_input_noise_dim=5, addi_attribute_input_noise_dim=5,
                 initial_state=RNNInitialStateType.RANDOM):
        '''
        :param data_feature_outputs: A list of Output objects, indicating the 
            dimension, type, normalization of each feature
        :param data_attribute_outputs A list of Output objects, indicating the 
            dimension, type, normalization of each attribute
        :param real_attribute_mask: List of True/False, the length equals the 
            number of attributes. False if the attribute is (max-min)/2 or
            (max+min)/2, True otherwise
        :param num_packing: Packing degree in PacGAN (a method for solving mode
            collapse in NeurIPS 2018, see https://arxiv.org/abs/1712.04086), the
            value defaults to 1.
        '''
        super().__init__()
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.num_packing = num_packing
        self.sample_len = sample_len
        self.real_attribute_mask = real_attribute_mask
        self.feature_out_dim = (np.sum([t.dim for t in data_feature_outputs]) *
                                self.sample_len)
        self.attribute_out_dim = np.sum([t.dim for t in data_attribute_outputs])

        self.generator\
            = DoppelGANgerGenerator(feed_back=False, # feed back mode has not been supported
                                    noise=True,
                                    feature_outputs=data_feature_outputs,
                                    attribute_outputs=data_attribute_outputs,
                                    real_attribute_mask=real_attribute_mask,
                                    sample_len=sample_len,
                                    attribute_num_units=attribute_num_units,
                                    attribute_num_layers=attribute_num_layers,
                                    feature_num_units=feature_num_units,
                                    feature_num_layers=feature_num_layers,
                                    attribute_input_noise_dim=attribute_input_noise_dim,
                                    addi_attribute_input_noise_dim=addi_attribute_input_noise_dim,
                                    attribute_dim=None, # known attribute feed-in has not been supported
                                    initial_state=initial_state, # only ZERO and RANDOM are supported
                                    initial_stddev=0.02 # placehold without any usage
            )
        self.discriminator\
            = Discriminator(input_size=(int(self.feature_out_dim*L_max/self.sample_len))*self.num_packing + self.attribute_out_dim,
                            num_layers=discriminator_num_layers,
                            num_units=discriminator_num_units)
        self.attr_discriminator\
            = AttrDiscriminator(input_size=self.attribute_out_dim,
                                num_layers=attr_discriminator_num_layers,
                                num_units=attr_discriminator_num_units)

    def _check_data(self):
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

    def forward(self,
                data_feature,
                real_attribute_input_noise,
                addi_attribute_input_noise,
                feature_input_noise,
                data_attribute):
        # since we still not support self.num_packing
        '''
        :param data_feature: Training features, in numpy float32 array format.
            The size is [(number of training samples) x (maximum length) x
            (total dimension of features)]. The last two dimensions of 
            features are for indicating whether the time series has already 
            ended. [1, 0] means the time series does not end at this time
            step (i.e., the time series is still activated at the next time
            step). [0, 1] means the time series ends exactly at this time 
            step or has ended before. The features are padded by zeros 
            after the last activated batch.
            For example, 
            (1) assume maximum length is 6, and sample_len (the time series
            batch size) is 3:
            (1.1) If the length of a sample is 1, the last two dimensions
            of features should be: 
            [[0, 1],[0, 1],[0, 1],[0, 0],[0, 0],[0, 0]]
            (1.2) If the length of a sample is 3, the last two dimensions
            of features should be: 
            [[1, 0],[1, 0],[0, 1],[0, 0],[0, 0],[0, 0]]
            (1.3) If the length of a sample is 4, the last two dimensions
            of features should be:
            [[1, 0],[1, 0],[1, 0],[0, 1],[0, 1],[0, 1]]
            (2) assume maximum length is 6, and sample_len (the time series
            batch size) is 1:
            (1.1) If the length of a sample is 1, the last two dimensions
            of features should be: 
            [[0, 1],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]
            (1.2) If the length of a sample is 3, the last two dimensions
            of features should be: 
            [[1, 0],[1, 0],[0, 1],[0, 0],[0, 0],[0, 0]]
            (1.3) If the length of a sample is 4, the last two dimensions
            of features should be:
            [[1, 0],[1, 0],[1, 0],[0, 1],[0, 0],[0, 0]]
            Actually, you do not need to deal with generating those two
            dimensions. Function util.add_gen_flag does the job of adding
            those two dimensions to the original data.
            Those two dimensions are for enabling DoppelGANger to generate
            samples with different length
        :param data_attribute: Training attributes, in numpy float32 array format.
            The size is [(number of training samples) x (total dimension 
            of attributes)]
        :param data_gen_flag: Flags indicating the activation of features, in 
            numpy float32 array format. The size is [(number of training 
            samples) x (maximum length)]. 1 means the time series is 
            activated at this time step, 0 means the time series is 
            inactivated at this timestep. 
            For example, 
            (1) assume maximum length is 6:
            (1.1) If the length of a sample is 1, the flags should be: 
            [1, 0, 0, 0, 0, 0]
            (1.2) If the length of a sample is 3, the flags should be:
            [1, 1, 1, 0, 0, 0]
            Different from the last two dimensions of data_feature, the
            values of data_gen_flag does not influenced by sample_len
        :param sample_len: The time series batch size
        '''
        self.data_feature = data_feature
        self.data_attribute = data_attribute
        # self.data_gen_flag = data_gen_flag

        self._check_data() # temp removed

        if self.data_feature[0].shape[1] % self.sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        self.sample_time = int(self.data_feature[0].shape[1] / self.sample_len)
        self.sample_feature_dim = self.data_feature[0].shape[2]
        self.sample_attribute_dim = self.data_attribute[0].shape[1]
        self.sample_real_attribute_dim = 0
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.sample_real_attribute_dim += \
                    self.data_attribute_outputs[i].dim
        
        self.batch_size = self.data_feature[0].shape[0]

        self.real_attribute_mask_tensor = []
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                sub_mask_tensor = torch.ones(
                    self.batch_size, self.data_attribute_outputs[i].dim)
            else:
                sub_mask_tensor = torch.zeros(
                    self.batch_size, self.data_attribute_outputs[i].dim)
            self.real_attribute_mask_tensor.append(sub_mask_tensor)
        self.real_attribute_mask_tensor = torch.cat(
            self.real_attribute_mask_tensor,
            dim=1)

        # if len(self.data_gen_flag.shape) != 2:
        #     raise Exception("data_gen_flag should be 2 dimension")

        # self.data_gen_flag = np.expand_dims(self.data_gen_flag, 2)
        # (batch_size, max_length, 1)

        # generate training route (fake)
        self.g_output_feature_train_tf_l = []
        self.g_output_attribute_train_tf_l = []
        # self.g_output_gen_flag_train_tf_l = []
        # self.g_output_length_train_tf_l = []
        # self.g_output_argmax_train_tf_l = []

        for i in range(self.num_packing):
            (g_output_feature_train_tf, g_output_attribute_train_tf,
             _, _, _) = \
                self.generator(real_attribute_input_noise[i],
                               addi_attribute_input_noise[i],
                               feature_input_noise[i],
                               data_feature[i])
            self.g_output_feature_train_tf_l.append(
                g_output_feature_train_tf)
            self.g_output_attribute_train_tf_l.append(
                g_output_attribute_train_tf)
            # self.g_output_gen_flag_train_tf_l.append(
            #     g_output_gen_flag_train_tf)
            # self.g_output_length_train_tf_l.append(
            #     g_output_length_train_tf)
            # self.g_output_argmax_train_tf_l.append(
            #     g_output_argmax_train_tf)

        self.g_output_feature_train_tf = torch.cat(
            self.g_output_feature_train_tf_l,
            dim=1)
        self.g_output_attribute_train_tf = torch.cat(
            self.g_output_attribute_train_tf_l,
            dim=1)

        self.d_fake_train_tf = self.discriminator(
            self.g_output_feature_train_tf,
            self.g_output_attribute_train_tf)
        self.attr_d_fake_train_tf = self.attr_discriminator(
            self.g_output_attribute_train_tf)

        # generate training route (real)
        self.real_feature_pl = torch.cat(
            self.data_feature,
            dim=1)
        self.real_attribute_pl = torch.cat(
            self.data_attribute,
            dim=1)
        self.d_real_train_tf = self.discriminator(
            self.real_feature_pl,
            self.real_attribute_pl)
        self.attr_d_real_train_tf = self.attr_discriminator(
            self.real_attribute_pl)

        return self.d_fake_train_tf, self.attr_d_fake_train_tf,\
               self.d_real_train_tf, self.attr_d_real_train_tf
    
    def sample_from(self,
                    real_attribute_input_noise,
                    addi_attribute_input_noise,
                    feature_input_noise,
                    feature_input_data,
                    batch_size=32):
        self._check_data()
        features = []
        attributes = []
        gen_flags = []
        lengths = []
        round_ = int(math.ceil(float(feature_input_noise.shape[0]) / batch_size))
        assert self.training is False, "please call .eval() on the model"
        self.generator.eval()
        for i in range(round_):
            (feature, attribute, gen_flag, length, _) = \
                    self.generator(real_attribute_input_noise[i * batch_size:
                                                              (i + 1) * batch_size],
                                   addi_attribute_input_noise[i * batch_size:
                                                           (i + 1) * batch_size],
                                   feature_input_noise[i * batch_size:
                                                       (i + 1) * batch_size],
                                   feature_input_data[i * batch_size:
                                                      (i + 1) * batch_size])
            features.append(feature)
            attributes.append(attribute)
            gen_flags.append(gen_flag)
            lengths.append(length)
        features = torch.cat(features, dim=0)
        attributes = torch.cat(attributes, dim=0)
        gen_flags = torch.cat(gen_flags, dim=0)
        lengths = torch.cat(lengths, dim=0)
        gen_flags = gen_flags[:, :, 0]

        features = features.detach().numpy()
        attributes = attributes.detach().numpy()
        gen_flags = gen_flags.detach().numpy()
        lengths = lengths.detach().numpy()

        features = np.delete(features, self.gen_flag_dims, axis=2)

        return features, attributes, gen_flags, lengths

