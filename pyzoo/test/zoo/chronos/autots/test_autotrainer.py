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
import pytest

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from zoo.chronos.autots.trainer import AutoTSTrainer
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


class TestAutoTrainer(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit_builtin_lstm(self):
        search_space = {
            'input_feature_num': input_feature_dim,
            'output_target_num': output_feature_dim,
            'optimizer': 'Adam',
            'hidden_dim': hp.grid_search([32, 64]),
            'layer_num': hp.randint(1, 3),
            'lr': hp.choice([0.001, 0.003, 0.01]),
            'dropout': hp.uniform(0.1, 0.2),
        }
        auto_trainer = AutoTSTrainer(model='lstm',
                                     search_space=search_space,
                                     metric="mse",
                                     loss=torch.nn.MSELoss(),
                                     logs_dir="/tmp/auto_trainer",
                                     cpus_per_trial=2,
                                     name="auto_trainer"
                                     )
        auto_trainer.fit(data=get_x_y(size=1000),
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=get_x_y(size=400),
                         n_sampling=1
                         )
        best_model = auto_trainer.get_best_model()
        assert best_model.config['lr'] in (0.001, 0.003, 0.01)
        assert best_model.config['batch_size'] in (32, 64)

    def test_fit_builtin_tcn(self):
        search_space = {
            'input_feature_num': input_feature_dim,
            'output_target_num': output_feature_dim,
            'past_seq_len': 5,
            'hidden_units': 8,
            'levels': hp.randint(1, 3),
            'kernel_size': hp.choice([2, 3]),
            'lr': hp.choice([0.001, 0.003, 0.01]),
            'dropout': hp.uniform(0.1, 0.2),
        }
        auto_trainer = AutoTSTrainer(model='tcn',
                                     search_space=search_space,
                                     logs_dir="/tmp/auto_trainer",
                                     cpus_per_trial=2,
                                     name="auto_trainer"
                                     )
        auto_trainer.fit(data=get_x_y(size=1000),
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=get_x_y(size=400),
                         n_sampling=1
                         )

        best_model = auto_trainer.get_best_model()
        assert best_model.config['lr'] in (0.001, 0.003, 0.01)
        assert best_model.config['batch_size'] in (32, 64)


if __name__ == "__main__":
    pytest.main([__file__])
