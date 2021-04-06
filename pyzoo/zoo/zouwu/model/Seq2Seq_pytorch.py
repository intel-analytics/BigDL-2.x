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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_num, dropout=dropout, batch_first=True)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_seq):
        # input_seq = [batch_size, seq_len, feature_num]
        outputs, (hidden, cell) = self.rnn(input_seq)
        # outputs = [batch size, seq len, hidden dim]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]

        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, layer_num, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        self.rnn = nn.LSTM(output_dim, hidden_dim, layer_num, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, decoder_input, hidden, cell):
        # input = [batch size]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]

        # input = decoder_input.view(-1, 1)
        # decoder_input = [batch size, 1], since output_dim is 1
        decoder_input = decoder_input.unsqueeze(1)
        # decoder_input = [batch_size, 1, 1]

        output, (hidden, cell) = self.rnn(decoder_input, (hidden, cell))

        # output = [batch size, seq len, hidden dim]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]

        # seq len will always be 1 in the decoder, therefore:
        # output = [batch size, 1, hidden dim]
        # hidden = [batch size, layer num, hidden dim]
        # cell = [batch size, layer num, hidden dim]
        prediction = self.fc_out(output.squeeze())
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout, output_dim, future_seq_len=1):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, layer_num, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, layer_num, dropout)
        self.future_seq_len = future_seq_len

    def forward(self, source, target=None):
        # past_seq_len
        batch_size = source.shape[0]

        output_dim = self.decoder.output_dim

        # tensor to store the predicted outputs
        target_seq = torch.zeros(batch_size, self.future_seq_len, output_dim)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(source)

        # Populate the first target sequence with end of encoding series value
        # decoder_input : [batch_size, output_dim]
        decoder_input = source[:, -1, :output_dim]

        for i in range(self.future_seq_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            target_seq[:, i] = decoder_output
            if target is None:
                # in test mode
                decoder_input = decoder_output
            else:
                decoder_input = target[:, i]
        return target_seq


def model_creator(config):
    return Seq2Seq(input_dim=config["input_dim"],
                   hidden_dim=config.get("hidden_dim", 32),
                   layer_num=config.get("layer_num", 2),
                   dropout=config.get("dropout", 0.2),
                   output_dim=config["output_dim"],
                   future_seq_len=config["future_seq_len"])


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 0.001))


def loss_creator(config):
    return nn.MSELoss()


class Seq2SeqPytorch(PytorchBaseModel):

    def __init__(self, config, check_optional_config=True, future_seq_len=1):
        """
        Constructor of Vanilla LSTM model
        """
        super().__init__(model_creator=model_creator,
                         optimizer_creator=optimizer_creator,
                         loss_creator=loss_creator,
                         config=config.update({"future_seq_len": future_seq_len}),
                         check_optional_config=check_optional_config)

    def _forward(self, x, y):
        yhat = self.model(x, y)
        return yhat

    def _pre_processing(self, x, y, validation_data):
        """
        pre_process input data
        1. expand dims for y and vay_y
        2. get input lengths
        :param x:
        :param y:
        :param validation_data:
        :return:
        """
        def expand_y(y):
            while len(y.shape) < 3:
                y = np.expand_dims(y, axis=2)
            return y
        y = expand_y(y)
        self.feature_num = x.shape[2]
        self.output_dim = y.shape[2]
        if validation_data is not None:
            val_x, val_y = validation_data
            val_y = expand_y(val_y)
            validation_data = (val_x, val_y)
        return x, y, validation_data

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, **config):
        x, y, validation_data = self._pre_processing(x, y, validation_data)
        return super().fit_eval(x, y, validation_data, mc, verbose, **config)

    def _get_required_parameters(self):
        return {
            "input_dim"
            "ouput_dim"
        }

    def _get_optional_parameters(self):
        return {
            'hidden_dim',
            'layer_num',
            'hidden_dim',
        } | super()._get_optional_parameters()
