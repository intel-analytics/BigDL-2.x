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
from unittest import TestCase
from zoo.automl.model import ModelBuilder
import numpy as np
import torch
import torch.nn as nn


def get_data():
    def get_linear_data(a, b, size):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        y = a*x + b
        return x, y
    train_x, train_y = get_linear_data(2, 5, 1000)
    val_x, val_y = get_linear_data(2, 5, 400)
    data = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
    return data


def model_creator_pytorch(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def loss_creator(config):
    return nn.MSELoss()


class TestBasePytorchModel(TestCase):
    data = get_data()

    def test_fit_evaluate(self):
        modelBuilder = ModelBuilder.from_pytorch(model_creator=model_creator_pytorch,
                                                 optimizer_creator=optimizer_creator,
                                                 loss_creator=loss_creator)
        model = modelBuilder.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        val_result = model.fit_eval(x=self.data["x"],
                                    y=self.data["y"],
                                    validation_data=(self.data["val_x"], self.data["val_y"]),
                                    epochs=20)
        assert val_result is not None
