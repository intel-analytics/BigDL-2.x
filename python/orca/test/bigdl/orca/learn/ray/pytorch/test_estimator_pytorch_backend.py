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

import numpy as np
import pytest

import torch
import torch.nn as nn
from zoo.orca.learn.pytorch import Estimator

np.random.seed(1337)  # for reproducibility


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, size=1000):
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.x = torch.cat([X1, X2], dim=0)
        Y1 = torch.zeros(size // 2, 1)
        Y2 = torch.ones(size // 2, 1)
        self.y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


def train_data_loader(config):
    train_dataset = LinearDataset(size=config.get("data_size", 1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )
    return train_loader


def val_data_loader(config):
    val_dataset = LinearDataset(size=config.get("val_size", 400))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))
    return validation_loader


def get_model(config):
    return Net()


def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


class TestPyTorchEstimator(TestCase):
    def test_data_creator(self):
        estimator = Estimator.from_torch(model=get_model,
                                         optimizer=get_optimizer,
                                         loss=nn.BCELoss(),
                                         config={"lr": 1e-2},
                                         workers_per_node=2,
                                         backend="torch_distributed")
        train_stats = estimator.fit(train_data_loader, epochs=2, batch_size=128)
        print(train_stats)
        val_stats = estimator.evaluate(val_data_loader, batch_size=64)
        print(val_stats)
        assert 0 < val_stats["val_accuracy"] < 1
        assert estimator.get_model()

        # Verify syncing weights, i.e. the two workers have the same weights after training
        import ray
        remote_workers = estimator.estimator.remote_workers
        state_dicts = ray.get([worker.state_dict.remote() for worker in remote_workers])
        weights = [state["models"] for state in state_dicts]
        worker1_weights = weights[0][0]
        worker2_weights = weights[1][0]
        for layer in list(worker1_weights.keys()):
            assert np.allclose(worker1_weights[layer].numpy(),
                               worker2_weights[layer].numpy())
        estimator.shutdown()

    def test_spark_xshards(self):
        from zoo import init_nncontext
        from zoo.orca.data import SparkXShards
        estimator = Estimator.from_torch(model=get_model,
                                         optimizer=get_optimizer,
                                         loss=nn.BCELoss(),
                                         config={"lr": 1e-1},
                                         backend="torch_distributed")
        sc = init_nncontext()
        x_rdd = sc.parallelize(np.random.rand(4000, 1, 50).astype(np.float32))
        y_rdd = sc.parallelize(np.random.randint(0, 2, size=(4000, 1)).astype(np.float32))
        rdd = x_rdd.zip(y_rdd).map(lambda x_y: {'x': x_y[0], 'y': x_y[1]})
        train_rdd, val_rdd = rdd.randomSplit([0.9, 0.1])
        train_xshards = SparkXShards(train_rdd)
        val_xshards = SparkXShards(val_rdd)
        train_stats = estimator.fit(train_xshards, batch_size=256, epochs=2)
        print(train_stats)
        val_stats = estimator.evaluate(val_xshards, batch_size=128)
        print(val_stats)
        estimator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
