#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import pickle
import os

from zoo.chronos.simulator.doppelganger.data_module import DoppelGANgerDataModule
from zoo.chronos.simulator.doppelganger.util import gen_attribute_input_noise,\
    gen_feature_input_noise, gen_feature_input_data_free, renormalize_per_sample
from zoo.chronos.simulator.doppelganger.doppelganger_pl import DoppelGANger_pl

import torch
from pytorch_lightning import Trainer, seed_everything

MODEL_PATH = "doppelganger.ckpt"
FEATURE_OUTPUT = "feature.output.ckpt"
ATTRIBUTE_OUTPUT = "attribute.output.ckpt"


class DoppelGANgerSimulator:
    '''
    Doppelganger Simulator for time series generation.
    '''
    def __init__(self,
                 L_max,
                 sample_len,
                 feature_dim,
                 num_real_attribute,
                 discriminator_num_layers=5,
                 discriminator_num_units=200,
                 attr_discriminator_num_layers=5,
                 attr_discriminator_num_units=200,
                 attribute_num_units=100,
                 attribute_num_layers=3,
                 feature_num_units=100,
                 feature_num_layers=1,
                 attribute_input_noise_dim=5,
                 addi_attribute_input_noise_dim=5,
                 d_gp_coe=10,
                 attr_d_gp_coe=10,
                 g_attr_d_coe=1,
                 d_lr=0.001,
                 attr_d_lr=0.001,
                 g_lr=0.001,
                 g_rounds=1,
                 d_rounds=1,
                 seed=0,
                 num_threads=None,
                 ckpt_dir=None):
        '''
        Initialize a doppelganger simulator.

        :param L_max: the maximum length of your feature.
        :param sample_len: the sample length to control LSTM length, should be a divider to L_max
        :param feature_dim: dimention of the feature
        :param num_real_attribute: the length of you attribute, which should be equal to the
               len(data_attribute).
        :param discriminator_num_layers: MLP layer num for discriminator.
        :param discriminator_num_units: MLP hidden unit for discriminator.
        :param attr_discriminator_num_layers: MLP layer num for attr discriminator.
        :param attr_discriminator_num_units: MLP hidden unit for attr discriminator.
        :param attribute_num_units: MLP layer num for attr generator/addi attr generator.
        :param attribute_num_layers:  MLP hidden unit for attr generator/addi attr generator.
        :param feature_num_units: LSTM hidden unit for feature generator.
        :param feature_num_layers: LSTM layer num for feature generator.
        :param attribute_input_noise_dim: noise data dim for attr generator.
        :param addi_attribute_input_noise_dim: noise data dim for addi attr generator.
        :param d_gp_coe: gradient penalty ratio for d loss.
        :param attr_d_gp_coe: gradient penalty ratio for attr d loss.
        :param g_attr_d_coe: ratio between feature loss and attr loss for g loss.
        :param d_lr: learning rate for discriminator.
        :param attr_d_lr: learning rate for attr discriminator.
        :param g_lr: learning rate for genereators.
        :param g_rounds: g rounds.
        :param d_rounds: d rounds.
        :param seed: random seed.
        :param num_threads: num of threads to be used for training.
        '''
        # additional settings
        seed_everything(seed=seed)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.ckpt_dir = ckpt_dir
        self.sample_len = sample_len
        self.L_max = L_max
        self.feature_dim = feature_dim
        self.num_real_attribute = num_real_attribute

        # hparam saving
        self.params = {"discriminator_num_layers": discriminator_num_layers,
                       "discriminator_num_units": discriminator_num_units,
                       "attr_discriminator_num_layers": attr_discriminator_num_layers,
                       "attr_discriminator_num_units": attr_discriminator_num_units,
                       "attribute_num_units": attribute_num_units,
                       "attribute_num_layers": attribute_num_layers,
                       "feature_num_units": feature_num_units,
                       "feature_num_layers": feature_num_layers,
                       "attribute_input_noise_dim": attribute_input_noise_dim,
                       "addi_attribute_input_noise_dim": addi_attribute_input_noise_dim,
                       "d_gp_coe": d_gp_coe,
                       "attr_d_gp_coe": attr_d_gp_coe,
                       "g_attr_d_coe": g_attr_d_coe,
                       "d_lr": d_lr,
                       "attr_d_lr": attr_d_lr,
                       "g_lr": g_lr,
                       "g_rounds": g_rounds,
                       "d_rounds": d_rounds}

        # model init
        self.model = None  # model will be lazy built since the dim will depend on the data

    def fit(self,
            data_feature,
            data_attribute,
            data_gen_flag,
            feature_outputs,
            attribute_outputs,
            epoch=1,
            batch_size=32):
        '''
        Fit on the training data(typically the private data).

        :param data_feature: Training features, in numpy float32 array format.
               The size is [(number of training samples) x (maximum length)
               x (total dimension of features)]. Categorical features are stored
               by one-hot encoding; for example, if a categorical feature has 3
               possibilities, then it can take values between [1., 0., 0.],
               [0., 1., 0.], and [0., 0., 1.]. Each continuous feature should be
               normalized to [0, 1] or [-1, 1]. The array is padded by zeros after
               the time series ends.
        :param data_attribute: Training attributes, in numpy float32 array format. The size is
               [(number of training samples) x (total dimension of attributes)]. Categorical
               attributes are stored by one-hot encoding; for example, if a categorical
               attribute has 3 possibilities, then it can take values between [1., 0., 0.],
               [0., 1., 0.], and [0., 0., 1.]. Each continuous attribute should be normalized
               to [0, 1] or [-1, 1].
        :param data_gen_flag: Flags indicating the activation of features, in numpy float32
               array format. The size is [(number of training samples) x (maximum length)].
               1 means the time series is activated at this time step, 0 means the time series
               is inactivated at this timestep.
        :param feature_outputs: A list of Output indicates the meta data of data_feature.
        :param attribute_outputs: A list of Output indicates the meta data of data_attribute.
        :param epoch: training epoch.
        :param batch_size: training batchsize.
        '''
        # data preparation
        real_data = {}
        real_data["data_feature"] = data_feature
        real_data["data_attribute"] = data_attribute
        real_data["data_gen_flag"] = data_gen_flag
        self.data_module = DoppelGANgerDataModule(real_data=real_data,
                                                  feature_outputs=feature_outputs,
                                                  attribute_outputs=attribute_outputs,
                                                  sample_len=self.sample_len,
                                                  batch_size=batch_size)

        # build the model
        self.model = DoppelGANger_pl(data_feature_outputs=self.data_module.data_feature_outputs,
                                     data_attribute_outputs=self.data_module.data_attribute_outputs,
                                     L_max=self.L_max,
                                     sample_len=self.sample_len,
                                     num_real_attribute=self.num_real_attribute,
                                     **self.params)
        self.trainer = Trainer(logger=False,
                               checkpoint_callback=False,
                               max_epochs=epoch,
                               default_root_dir=self.ckpt_dir)

        # fit!
        self.trainer.fit(self.model, self.data_module)

    def generate(self, sample_num=1, batch_size=32):
        '''
        Generate synthetic data with similar distribution as training data.

        :param sample_num: How many samples to be generated.
        :param batch_size: batch size to generate.
        '''
        # set to inference mode
        self.model.eval()
        total_generate_num_sample = sample_num

        # generate noise and inputs
        real_attribute_input_noise = gen_attribute_input_noise(total_generate_num_sample)
        addi_attribute_input_noise = gen_attribute_input_noise(total_generate_num_sample)
        feature_input_noise = gen_feature_input_noise(total_generate_num_sample, self.model.length)
        feature_input_data = gen_feature_input_data_free(total_generate_num_sample,
                                                         self.model.sample_len,
                                                         self.feature_dim)
        real_attribute_input_noise = torch.from_numpy(real_attribute_input_noise).float()
        addi_attribute_input_noise = torch.from_numpy(addi_attribute_input_noise).float()
        feature_input_noise = torch.from_numpy(feature_input_noise).float()
        feature_input_data = torch.from_numpy(feature_input_data).float()

        # generate
        features, attributes, gen_flags, lengths\
            = self.model.sample_from(real_attribute_input_noise,
                                     addi_attribute_input_noise,
                                     feature_input_noise,
                                     feature_input_data,
                                     batch_size=batch_size)

        # renormalize (max, min)
        features, attributes = renormalize_per_sample(
            features, attributes, self.model.data_feature_outputs,
            self.model.data_attribute_outputs, gen_flags,
            num_real_attribute=self.num_real_attribute)  # -2 for addi attr

        return features, attributes, gen_flags, lengths

    def save(self, path_dir):
        '''
        Save the simulator.

        :param path_dir: saving path
        '''
        self.trainer.save_checkpoint(os.path.join(path_dir, MODEL_PATH))
        with open(os.path.join(path_dir, FEATURE_OUTPUT), "wb") as f:
            pickle.dump(self.data_module.data_feature_outputs, f)
        with open(os.path.join(path_dir, ATTRIBUTE_OUTPUT), "wb") as f:
            pickle.dump(self.data_module.data_attribute_outputs, f)

    def load(self,
             path_dir):
        '''
        Load the simulator.

        :param path_dir: saving path
        '''
        with open(os.path.join(path_dir, FEATURE_OUTPUT), "rb") as f:
            data_feature_outputs = pickle.load(f)
        with open(os.path.join(path_dir, ATTRIBUTE_OUTPUT), "rb") as f:
            data_attribute_outputs = pickle.load(f)
        self.model =\
            DoppelGANger_pl.load_from_checkpoint(os.path.join(path_dir, MODEL_PATH),
                                                 data_feature_outputs=data_feature_outputs,
                                                 data_attribute_outputs=data_attribute_outputs)
