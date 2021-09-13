import os
import numpy as np
import sys
from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
from torch import nn

from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything

from .doppelganger import DoppelGANger
from .network import RNNInitialStateType
from .loss import doppelganger_loss
from .util import gen_attribute_input_noise, gen_feature_input_noise,\
    gen_feature_input_data_free, renormalize_per_sample

class DoppelGANger_pl(LightningModule):
    def __init__(self,
                 datamodule,
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
                 **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters("discriminator_num_layers",
                                  "discriminator_num_units",
                                  "attr_discriminator_num_layers",
                                  "attr_discriminator_num_units",
                                  "attribute_num_units",
                                  "attribute_num_layers",
                                  "feature_num_units",
                                  "feature_num_layers",
                                  "attribute_input_noise_dim",
                                  "addi_attribute_input_noise_dim",
                                  "d_gp_coe",
                                  "attr_d_gp_coe",
                                  "g_attr_d_coe",
                                  "d_lr",
                                  "attr_d_lr",
                                  "g_lr",
                                  "g_rounds",
                                  "d_rounds")
        self.datamodule = datamodule
        self.length = self.datamodule.length
        self.g_rounds = g_rounds
        self.d_rounds = d_rounds
        # model init
        self.model = DoppelGANger(data_feature_outputs=self.datamodule.data_feature_outputs,
                                  data_attribute_outputs=self.datamodule.data_attribute_outputs,
                                  real_attribute_mask=self.datamodule.real_attribute_mask,
                                  sample_len=self.datamodule.sample_len,
                                  L_max=self.datamodule.data_feature.shape[1],
                                  num_packing=1,  # any num other than 1 will be supported later
                                  discriminator_num_layers=self.hparams.discriminator_num_layers,
                                  discriminator_num_units=self.hparams.discriminator_num_units,
                                  attr_discriminator_num_layers=self.hparams.attr_discriminator_num_layers,
                                  attr_discriminator_num_units=self.hparams.attr_discriminator_num_units,
                                  attribute_num_units=self.hparams.attribute_num_units,
                                  attribute_num_layers=self.hparams.attribute_num_layers,
                                  feature_num_units=self.hparams.feature_num_units,
                                  feature_num_layers=self.hparams.feature_num_layers,
                                  attribute_input_noise_dim=self.hparams.attribute_input_noise_dim,
                                  addi_attribute_input_noise_dim=self.hparams.addi_attribute_input_noise_dim,
                                  initial_state=RNNInitialStateType.RANDOM)  # currently we fix this value
    
    def forward(self,
                data_feature,
                real_attribute_input_noise,
                addi_attribute_input_noise,
                feature_input_noise,
                data_attribute):
        return self.model([data_feature],
                          [real_attribute_input_noise],
                          [addi_attribute_input_noise],
                          [feature_input_noise],
                          [data_attribute])
    
    def training_step(self, batch, batch_idx):
        # data preparation
        data_feature, data_attribute = batch
        optimizer_d, optimizer_attr_d, optimizer_g = self.optimizers()
        # generate noise input
        real_attribute_input_noise = gen_attribute_input_noise(data_feature.shape[0])
        addi_attribute_input_noise = gen_attribute_input_noise(data_feature.shape[0])
        feature_input_noise = gen_feature_input_noise(data_feature.shape[0], self.length)
        real_attribute_input_noise = torch.from_numpy(real_attribute_input_noise).float()
        addi_attribute_input_noise = torch.from_numpy(addi_attribute_input_noise).float()
        feature_input_noise = torch.from_numpy(feature_input_noise).float()

        # g backward
        # open the generator grad since we need to update the weights in g
        for p in self.model.generator.parameters():
            p.requires_grad = True
        for i in range(self.g_rounds):
            d_fake, attr_d_fake,\
                    d_real, attr_d_real = self(data_feature,
                                               real_attribute_input_noise,
                                               addi_attribute_input_noise,
                                               feature_input_noise,
                                               data_attribute)
            g_loss, _, _ =\
                doppelganger_loss(d_fake, attr_d_fake, d_real, attr_d_real)
            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()

        # d backward
        # close the generator grad since we only need to update the weights in d
        for p in self.model.generator.parameters():
            p.requires_grad = False
        for i in range(self.d_rounds):
            d_fake, attr_d_fake,\
            d_real, attr_d_real = self(data_feature,
                                        real_attribute_input_noise,
                                        addi_attribute_input_noise,
                                        feature_input_noise,
                                        data_attribute)
            _, d_loss, attr_d_loss =\
                doppelganger_loss(d_fake, attr_d_fake, d_real, attr_d_real,
                                  g_attr_d_coe=self.hparams.g_attr_d_coe,
                                  gradient_penalty=True,
                                  discriminator=self.model.discriminator,
                                  attr_discriminator=self.model.attr_discriminator,
                                  g_output_feature_train_tf=self.model.g_output_feature_train_tf,
                                  g_output_attribute_train_tf=self.model.g_output_attribute_train_tf,
                                  real_feature_pl=self.model.real_feature_pl,
                                  real_attribute_pl=self.model.real_attribute_pl,
                                  d_gp_coe=self.hparams.d_gp_coe,
                                  attr_d_gp_coe=self.hparams.attr_d_gp_coe)
            optimizer_d.zero_grad()
            optimizer_attr_d.zero_grad()
            self.manual_backward(d_loss)
            self.manual_backward(attr_d_loss)
            optimizer_d.step()
            optimizer_attr_d.step()
            self.log("g_loss", g_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("d_loss", d_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("attr_d_loss", attr_d_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.hparams.d_lr, betas=(0.5, 0.999))
        optimizer_attr_d = torch.optim.Adam(self.model.attr_discriminator.parameters(), lr=self.hparams.attr_d_lr, betas=(0.5, 0.999))
        optimizer_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.hparams.g_lr, betas=(0.5, 0.999))
        return optimizer_d, optimizer_attr_d, optimizer_g

    def sample_from(self,
                    real_attribute_input_noise,
                    addi_attribute_input_noise,
                    feature_input_noise,
                    feature_input_data,
                    batch_size=32):
        features, attributes, gen_flags, lengths\
            = self.model.sample_from(real_attribute_input_noise,
                                     addi_attribute_input_noise,
                                     feature_input_noise,
                                     feature_input_data,
                                     batch_size=batch_size)
        return features, attributes, gen_flags, lengths
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DoppelGANger_pl")
        parser.add_argument("--discriminator_num_layers", type=int, default=5)
        parser.add_argument("--discriminator_num_units", type=int, default=200)
        parser.add_argument("--attr_discriminator_num_layers", type=int, default=5)
        parser.add_argument("--attr_discriminator_num_units", type=int, default=200)
        parser.add_argument("--attribute_num_units", type=int, default=100)
        parser.add_argument("--attribute_num_layers", type=int, default=3)
        parser.add_argument("--feature_num_units", type=int, default=100)
        parser.add_argument("--feature_num_layers", type=int, default=1)
        parser.add_argument("--attribute_input_noise_dim", type=int, default=5)
        parser.add_argument("--addi_attribute_input_noise_dim", type=int, default=5)
        parser.add_argument("--d_gp_coe", type=int, default=10)
        parser.add_argument("--attr_d_gp_coe", type=int, default=10)
        parser.add_argument("--g_attr_d_coe", type=int, default=1)
        parser.add_argument("--d_lr", type=float, default=0.001)
        parser.add_argument("--attr_d_lr", type=float, default=0.001)
        parser.add_argument("--g_lr", type=float, default=0.001)
        return parent_parser
