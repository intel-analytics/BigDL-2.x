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


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_num, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_seq):
        lstm_out, hidden = self.lstm(input_seq)
        # reshaping the outputs to feed in fully connected layer
        # out = lstm_out[-1].contiguous().view(-1, self.hidden_dim)
        # out = self.linear(out, len(input_seq), -1)
        out = self.fc(lstm_out[:, -1, :])
        out = out.view(out.shape[0], 1, out.shape[1])
        return out


def model_creator(config):
    return LSTMModel(input_dim=config["input_feature_num"],
                     hidden_dim=config.get("hidden_dim", 32),
                     layer_num=config.get("layer_num", 2),
                     dropout=config.get("dropout", 0.2),
                     output_dim=config["output_feature_num"],)


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 0.001))


def loss_creator(config):
    return nn.MSELoss()


class VanillaLSTMPytorch(PytorchBaseModel):

    def __init__(self, check_optional_config=True):
        """
        Constructor of Vanilla LSTM model
        """
        super().__init__(model_creator=model_creator,
                         optimizer_creator=optimizer_creator,
                         loss_creator=loss_creator,
                         check_optional_config=check_optional_config)

    def _get_required_parameters(self):
        return {
            "input_feature_num",
            "output_feature_num"
        }

    def _get_optional_parameters(self):
        return {
            'hidden_dim',
            'layer_num',
            'dropout',
        } | super()._get_optional_parameters()
