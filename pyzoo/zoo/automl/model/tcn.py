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
# MIT License
#
# Copyright (c) 2018 CMU Locus Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file is adapted from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# https://github.com/locuslab/TCN/blob/master/TCN/adding_problem/add_test.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


def model_creator(config):
    return TemporalConvNet(input_size=config["input_size"],
                           output_size=config["output_size"],
                           num_channels=config.get("nhid", 30) * config.get("levels", 8),
                           kernel_size=config.get("kernel_size", 7),
                           dropout=config.get("dropout", 0.2))


def optimizer_creator(model, config):
    return getattr(torch.optim, config.get("optim", "Adam"))(model.parameters(),
                                                             lr=config.get("lr", 4e-3))


def loss_creator(config):
    return nn.MSELoss()


class TCNPytorch(BaseModel):
    def __init__(self, config, check_optional_config=False):
        self.check_optional_config = check_optional_config
        self._check_config(**config)
        self.model = model_creator(config)
        self.optimizer = optimizer_creator(self.model, config)
        self.criterion = loss_creator(config)
        self.config = config

    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, epochs=1, metric=["mse"],
                 **config):
        epoch_losses = []
        for i in range(epochs):
            train_loss = self.train_epoch(x, y)
            epoch_losses.append(train_loss)
        train_stats = {"loss": np.mean(epoch_losses), "last_loss": epoch_losses[-1]}
        # todo: support input validation data None
        assert validation_data is not None, "You must input validation data!"
        val_stats = self._validate(validation_data[0], validation_data[1], metric=metric)
        return train_stats.update(val_stats)

    def train_epoch(self, x, y):
        # todo: support torch data loader
        batch_size = self.config["batch_size"]
        self.model.train()
        batch_idx = 0
        total_loss = 0
        for i in range(0, x.size(0), batch_size):
            if i + batch_size > x.size(0):
                x, y = x[i:], y[i:]
            else:
                x, y = x[i:(i + batch_size)], y[i:(i + batch_size)]
            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.criterion(yhat, y)
            loss.backward()
            self.optimizer.step()
            batch_idx += 1
            total_loss += loss.item()
        train_loss = total_loss / batch_idx
        return train_loss

    def _validate(self, x, y, metric):
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(x)
            val_loss = self.criterion(yhat, y)
            eval_result = Evaluator.evaluate(metric=metric,
                                             y_true=y, y_pred=yhat.numpy(),
                                             multioutput='uniform_average')
        return {"val_loss": val_loss.item(),
                metric: eval_result}

    def evaluate(self, x, y, metric=['mse']):
        yhat = self.predict(x)
        eval_result = [Evaluator.evaluate(m, y_true=y, y_pred=yhat, multioutput="raw_values")
                       for m in metric]
        return eval_result

    def predict(self, x, mc=False):
        if mc:
            self.model.train()
        else:
            self.model.eval()
        yhat = self.model(x)
        return yhat

    def predict_with_uncertainty(self, x, n_iter=100):
        result = np.zeros((n_iter,) + (x.shape[0], self.config["output_size"]))
        for i in range(n_iter):
            result[i, :, :] = self.predict(x, mc=True)

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty

    def state_dict(self):
        state = {
            "config": self.config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.config = state["config"]
        self.optimizer.load_state_dict(state["optimizer"])

    def save(self, model_path, config_path, **config):
        state_dict = self.state_dict()
        torch.save(state_dict, model_path)

    def restore(self, model_path, **config):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def _get_required_parameters(self):
        return {
            "input_size",
            "output_size"
        }

    def _get_optional_parameters(self):
        return {
            "nhid",
            "levels",
            "kernel_size",
            'lr',
            "dropout",
            'batch_size'
        }
