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

# The Clear BSD License

# Copyright (c) 2019 Carnegie Mellon University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#     * Neither the name of Carnegie Mellon University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

from zoo.chronos.simulator.doppelganger.data_module import DoppelGANgerDataModule
from zoo.chronos.simulator.doppelganger.util import gen_attribute_input_noise, gen_feature_input_noise,\
    gen_feature_input_data_free, renormalize_per_sample
from zoo.chronos.simulator.doppelganger.doppelganger_pl import DoppelGANger_pl

import torch
from pytorch_lightning import Trainer, seed_everything

class DoppelGANgerSimulator:
    '''
    Doppelganger Simulator for time series generation.
    '''
    def __init__(self,
                 sample_len,
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
        self.model = None  # model will be lazy built during fit since the dim will depend on the data

    def fit(self,
            real_data,
            feature_outputs,
            attribute_outputs,
            epoch=1,
            batch_size=32):
        '''
        :param real_data: The real data should be a collection with indexs "data_feature",
               "data_attribute" and "data_gen_flag".
        :param feature_outputs: A list of Output indicates the meta data of data_feature.
        :param attribute_outputs: A list of Output indicates the meta data of data_attribute.
        :param epoch: training epoch.
        :param batch_size: training batchsize.
        '''
        # data preparation
        self.data_module = DoppelGANgerDataModule(real_data=real_data,
                                                  feature_outputs=feature_outputs,
                                                  attribute_outputs=attribute_outputs,
                                                  sample_len=self.sample_len,
                                                  batch_size=batch_size)
        
        # profiler
        from pytorch_lightning.profiler.pytorch import PyTorchProfiler
        self.profiler = PyTorchProfiler(dirpath=".",
                                        filename="py.log",
                                        profile_memory=True)

        # build the model
        self.model = DoppelGANger_pl(datamodule=self.data_module, **self.params)
        self.trainer = Trainer(logger=False,
                               max_epochs=epoch,
                               default_root_dir=self.ckpt_dir,
                               profiler=self.profiler)

        # fit!
        self.trainer.fit(self.model, self.data_module)

    def generate(self, sample_num=1, batch_size=32):
        '''
        :param sample_num: How many samples to be generated.
        :param batch_size: batch size to generate.
        '''
        # set to inference mode
        self.model.eval()
        total_generate_num_sample = sample_num

        # generate noise and inputs
        length = self.data_module.length
        real_attribute_input_noise = gen_attribute_input_noise(total_generate_num_sample)
        addi_attribute_input_noise = gen_attribute_input_noise(total_generate_num_sample)
        feature_input_noise = gen_feature_input_noise(total_generate_num_sample, length)
        feature_input_data = gen_feature_input_data_free(total_generate_num_sample,
                                                         self.data_module.sample_len,
                                                         self.data_module.data_feature.shape[-1])
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
                    features, attributes, self.data_module.data_feature_outputs,
                    self.data_module.data_attribute_outputs, gen_flags,
                    num_real_attribute=self.data_module.num_real_attribute)

        return features, attributes, gen_flags, lengths

    def save(self, path):
        '''
        :param path: saving path
        '''
        self.trainer.save_checkpoint(path)

    def load(self,
             path,
             real_data,
             feature_outputs,
             attribute_outputs,
             batch_size=32):
        '''
        :param path: saving path
        :param real_data: same as in fit.
        :param feature_outputs: same as in fit.
        :param attribute_outputs: same as in fit.
        :param batch_size: same as in fit.
        '''
        self.data_module = DoppelGANgerDataModule(real_data=real_data,
                                                  feature_outputs=feature_outputs,
                                                  attribute_outputs=attribute_outputs,
                                                  sample_len=self.sample_len,
                                                  batch_size=batch_size)
        self.model = DoppelGANger_pl.load_from_checkpoint(path, datamodule=self.data_module)
