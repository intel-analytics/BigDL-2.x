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

import torch
import torch.nn as nn
from zoo.automl.model.base_pytorch_model import PytorchBaseModel
import numpy as np


class LSTMSeq2Seq(nn.Module):
    def __init__(self,
                input_feature_num,
                future_seq_len,
                output_feature_num,
                lstm_hidden_dim=128,
                fc_hidden_dim=128,
                fc_layer_num=2,
                lstm_layer_num=1, 
                dropout=0.25):
        super(LSTMSeq2Seq, self).__init__()
        self.lstm_encoder = nn.LSTM(input_feature_num,
                                    lstm_hidden_dim,
                                    lstm_layer_num, 
                                    dropout=dropout, 
                                    batch_first=True)
        self.lstm_decoder = nn.LSTMCell(lstm_hidden_dim, lstm_hidden_dim)
        fc_layers = [lstm_hidden_dim] + [fc_hidden_dim]*(fc_layer_num-1) + [output_feature_num]
        layers = []
        for i in range(fc_layer_num):
            if i != 0:
                layers += [nn.Dropout(p=dropout)]
            layers += [nn.Linear(fc_layers[i], fc_layers[i+1])]
        self.fc = nn.Sequential(*layers)
        self.output_step = future_seq_len

    def forward(self, input_seq):
        x, (hid, cta) = self.lstm_encoder(input_seq)
        h, c = hid[-1], cta[-1]

        decode_list = []
        for i in range(self.output_step):
            h, c = self.lstm_decoder(h, (h, c))
            decode_list.append(h)

        decode = torch.stack(decode_list, dim=1)
        out = self.fc(decode)

        return out

def model_creator(config):
    return LSTMSeq2Seq(input_feature_num=config["input_feature_num"],
                       lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
                       lstm_layer_num=config.get("lstm_layer_num", 2),
                       dropout=config.get("dropout", 0.25),
                       output_feature_num=config["output_feature_num"],
                       future_seq_len=config["future_seq_len"],
                       fc_layer_num=config.get("fc_layer_num", 2)
                       fc_hidden_dim=config.get("fc_hidden_dim", 128))

def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 4e-3))

def loss_creator(config):
    return nn.MSELoss()

class Seq2SeqPytorch(PytorchBaseModel):
    def __init__(self, check_optional_config=False):
        super().__init__(model_creator=model_creator,
                         optimizer_creator=optimizer_creator,
                         loss_creator=loss_creator,
                         check_optional_config=check_optional_config)

    def _get_required_parameters(self):
        return {
            "input_feature_num",
            "future_seq_len",
            "output_feature_num"
        }

    def _get_optional_parameters(self):
        return {
            "lstm_hidden_dim",
            "lstm_layer_num",
            "fc_layer_num",
            "fc_hidden_dim"
        } | super()._get_optional_parameters()