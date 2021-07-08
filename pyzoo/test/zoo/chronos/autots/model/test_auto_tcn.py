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
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from unittest import TestCase
import pytest
from zoo.chronos.autots.model.auto_tcn import AutoTCN
from zoo.orca.automl import hp

input_feature_dim = 10
output_feature_dim = 2
past_seq_len = 5
future_seq_len = 1


def get_x_y(size):
    x = np.random.randn(size, past_seq_len, input_feature_dim)
    y = np.random.randn(size, future_seq_len, output_feature_dim)
    return x, y


class RandomDataset(Dataset):
    def __init__(self, size=1000):
        x, y = get_x_y(size)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_dataloader_creator(config):
    return DataLoader(RandomDataset(size=1000),
                      batch_size=config["batch_size"],
                      shuffle=True)


def valid_dataloader_creator(config):
    return DataLoader(RandomDataset(size=400),
                      batch_size=config["batch_size"],
                      shuffle=True)


def get_auto_estimator():
    auto_tcn = AutoTCN(input_feature_num=input_feature_dim,
                       output_target_num=output_feature_dim,
                       past_seq_len=past_seq_len,
                       future_seq_len=future_seq_len,
                       optimizer='Adam',
                       loss=torch.nn.MSELoss(),
                       metric="mse",
                       hidden_units=8,
                       levels=hp.randint(1, 3),
                       kernel_size=hp.choice([2, 3]),
                       lr=hp.choice([0.001, 0.003, 0.01]),
                       dropout=hp.uniform(0.1, 0.2),
                       logs_dir="/tmp/auto_tcn",
                       cpus_per_trial=2,
                       name="auto_tcn")
    return auto_tcn


class TestAutoTCN(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit_np(self):
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=get_x_y(size=1000),
                     epochs=1,
                     batch_size=hp.choice([32, 64]),
                     validation_data=get_x_y(size=400),
                     n_sampling=1,
                     )
        assert auto_tcn.get_best_model()
        best_config = auto_tcn.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert best_config['batch_size'] in (32, 64)
        assert 1 <= best_config['levels'] < 3

    def test_fit_data_creator(self):
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=train_dataloader_creator,
                     epochs=1,
                     batch_size=hp.choice([32, 64]),
                     validation_data=valid_dataloader_creator,
                     n_sampling=1,
                     )
        assert auto_tcn.get_best_model()
        best_config = auto_tcn.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert best_config['batch_size'] in (32, 64)
        assert 1 <= best_config['levels'] < 3

    def test_num_channels(self):
        auto_tcn = AutoTCN(input_feature_num=input_feature_dim,
                           output_target_num=output_feature_dim,
                           past_seq_len=past_seq_len,
                           future_seq_len=future_seq_len,
                           optimizer='Adam',
                           loss=torch.nn.MSELoss(),
                           metric="mse",
                           hidden_units=4,
                           levels=hp.randint(1, 3),
                           num_channels=[8] * 2,
                           kernel_size=hp.choice([2, 3]),
                           lr=hp.choice([0.001, 0.003, 0.01]),
                           dropout=hp.uniform(0.1, 0.2),
                           logs_dir="/tmp/auto_tcn",
                           cpus_per_trial=2,
                           name="auto_tcn")
        auto_tcn.fit(data=train_dataloader_creator,
                     epochs=1,
                     batch_size=hp.choice([32, 64]),
                     validation_data=valid_dataloader_creator,
                     n_sampling=1,
                     )
        assert auto_tcn.get_best_model()
        best_config = auto_tcn.get_best_config()
        assert best_config['num_channels'] == [8]*2


if __name__ == "__main__":
    pytest.main([__file__])
