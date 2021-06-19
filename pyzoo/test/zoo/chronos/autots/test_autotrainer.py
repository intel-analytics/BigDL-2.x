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

import torch
import numpy as np
from zoo.chronos.autots.experimental_trainer import AutoTSTrainer
from zoo.chronos.data.tsdataset import TSDataset
from zoo.orca.automl import hp
import pandas as pd


def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value 1": np.random.randn(sample_num),
                             "value 2": np.random.randn(sample_num),
                             "id": np.array(['00'] * sample_num),
                             "extra feature 1": np.random.randn(sample_num),
                             "extra feature 2": np.random.randn(sample_num)})
    return train_df


def get_tsdataset():
    df = get_ts_df()
    return TSDataset.from_pandas(df,
                                 dt_col="datetime",
                                 target_col=["value 1", "value 2"],
                                 extra_feature_col=["extra feature 1", "extra feature 2"],
                                 id_col="id")


class TestAutoTrainer(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit_lstm_feature(self):
        input_feature_dim = 11  # This param will not be used
        output_feature_dim = 2  # 2 targets (i.e. "value 1", "value 2") is generated in get_tsdataset

        search_space = {
            'optimizer': 'Adam',
            'hidden_dim': hp.grid_search([32, 64]),
            'layer_num': hp.randint(1, 3),
            'lr': hp.choice([0.001, 0.003, 0.01]),
            'dropout': hp.uniform(0.1, 0.2)
        }
        auto_trainer = AutoTSTrainer(model='lstm',
                                     search_space=search_space,
                                     past_seq_len=hp.randint(4, 6),
                                     future_seq_len=1,
                                     input_feature_num=input_feature_dim,
                                     output_target_num=output_feature_dim,
                                     selected_features="auto",
                                     metric="mse",
                                     loss=torch.nn.MSELoss(),
                                     logs_dir="/tmp/auto_trainer",
                                     cpus_per_trial=2,
                                     name="auto_trainer"
                                     )
        auto_trainer.fit(data=get_tsdataset().gen_dt_feature(),
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=get_tsdataset().gen_dt_feature(),
                         n_sampling=1
                         )
        config = auto_trainer.get_best_config()


if __name__ == "__main__":
    pytest.main([__file__])