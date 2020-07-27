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
import pytest
from unittest import TestCase
from zoo.orca.learn.pytorch.estimator import Estimator
import torch
import torch.nn as nn
import numpy as np


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, a, b, size=1000):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


def model_creator(config):
    return nn.Linear(1, config.get("hidden_size", 1))


def optimizer_creator(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def scheduler_creator(optimizer, config):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def train_data_creator(config):
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )
    return train_loader


def validation_data_creator(config):
    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))
    return validation_loader


class TestEstimatorForRay(TestCase):
    def test_estimator_horovod(self):
        estimator = Estimator.from_model_creator(model_creator=model_creator,
                                                 optimizer_creator=optimizer_creator,
                                                 loss_creator=nn.MSELoss,
                                                 scheduler_creator=scheduler_creator,
                                                 config={
                                                     "lr": 1e-2,  # used in optimizer_creator
                                                     "hidden_size": 1,  # used in model_creator
                                                     "batch_size": 4,  # used in data_creator
                                                 })
        stats_list = estimator.fit(data=train_data_creator, epochs=5)
        assert len(stats_list) == 5, "length of stats_list should be equal to epochs"
        print(stats_list)
        val_stats = estimator.evaluate(validation_data_creator)
        assert "val_loss" in val_stats
        print(val_stats)


if __name__ == "__main__":
    pytest.main([__file__])
