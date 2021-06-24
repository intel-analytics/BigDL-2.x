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
import os
import requests
import zipfile
import io

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler


from zoo.chronos.data.tsdataset import TSDataset
from zoo.chronos.autots.model.auto_lstm import AutoLSTM
from zoo.orca.automl import hp
from zoo.orca import init_orca_context, stop_orca_context


class AutoLSTMDataset(Dataset):

    def __init__(self, data_selection, past_seq_len, future_seq_len, **kwargs):
        self.x, self.y, _ = AutoLSTMDataset.preprocess_data(
            data_selection, past_seq_len, future_seq_len)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @staticmethod
    def preprocess_data(data_selection, past_seq_len, future_seq_len, **kwargs):
        url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        url_file_path = "00235/household_power_consumption.zip"
        file_name = url_file_path.split('/')[-1].rpartition('.')[0] + '.txt'
        file_path = os.path.abspath(
            os.path.dirname(__file__)) + '/' + file_name
        if not os.path.exists(file_path):
            download_file = requests.get(url_base + url_file_path)
            file = zipfile.ZipFile(io.BytesIO(download_file.content))
            file.extractall(os.path.abspath(os.path.dirname(__file__)))
        df = pd.read_csv(file_path, sep=';', header=0, low_memory=False,
                         infer_datetime_format=True, parse_dates={'datetime': [0, 1]}, nrows=2000)
        train_data = TSDataset.from_pandas(df[:int(len(df)*0.8)], dt_col="datetime",
                                           target_col=["Global_active_power"],
                                           extra_feature_col=["Global_reactive_power"])
        x_train, y_train = train_data.impute(mode="linear")\
                                     .scale(mm)\
                                     .roll(lookback=past_seq_len,
                                           horizon=future_seq_len)\
                                     .to_numpy()
        test_data = TSDataset.from_pandas(df[int(len(df)*0.8):], dt_col="datetime",
                                          target_col=["Global_active_power"],
                                          extra_feature_col=["Global_reactive_power"])
        x_test, y_test = test_data.impute(mode="linear")\
                                  .scale(mm, fit=False)\
                                  .roll(lookback=past_seq_len,
                                        horizon=future_seq_len)\
                                  .to_numpy()

        if data_selection == 'train':
            return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), test_data
        elif data_selection == 'valid':
            return torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float(), test_data


def train_dataloader_creator(config):
    return DataLoader(AutoLSTMDataset('train', **config),
                      batch_size=config["batch_size"],
                      shuffle=True)


def valid_dataloader_creator(config):
    return DataLoader(AutoLSTMDataset('valid', **config),
                      batch_size=config["batch_size"],
                      shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2,
                        help="The number of nodes to be used in the cluster. "
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--cluster_mode', type=str, default='local',
                        help="The mode for the Spark cluster.")
    parser.add_argument('--cores', type=int, default=4,
                        help="The number of cpu cores you want to use on each node."
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--memory', type=str, default="10g",
                        help="The memory you want to use on each node."
                        "You can change it depending on your own cluster setting.")

    parser.add_argument('--epoch', type=int, default=1,
                        help="Max number of epochs to train in each trial.")
    parser.add_argument('--cpus_per_trial', type=int, default=2,
                        help="Int. Number of cpus for each trial")
    parser.add_argument('--n_sampling', type=int, default=1,
                        help="Number of times to sample from the search_space.")

    args = parser.parse_args()

    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      memory=args.memory, num_nodes=num_nodes, init_ray_on_spark=True)

    mm = MinMaxScaler()

    auto_lstm = AutoLSTM(input_feature_num=2,
                         output_target_num=1,
                         past_seq_len=14,
                         optimizer='Adam',
                         loss=torch.nn.MSELoss(),
                         metric="mse",
                         hidden_dim=hp.grid_search([32, 64]),
                         layer_num=hp.randint(1, 3),
                         lr=hp.choice([0.001, 0.003, 0.01]),
                         dropout=hp.uniform(0.1, 0.2),
                         logs_dir="/tmp/auto_lstm",
                         cpus_per_trial=args.cpus_per_trial,
                         name="auto_lstm")
    auto_lstm.fit(data=train_dataloader_creator,
                  epochs=args.epoch,
                  batch_size=hp.choice([32, 64]),
                  validation_data=valid_dataloader_creator,
                  n_sampling=args.n_sampling,
                  )
    best_model = auto_lstm.get_best_model()

    _, _, test_data = AutoLSTMDataset.preprocess_data(
        'valid', **best_model.config)
    test_data_x, test_data_y = next(
        iter(valid_dataloader_creator(best_model.config)))

    testdata_x_unscale, testdata_y_unscale = test_data._unscale_numpy(
        test_data_x.numpy()), test_data._unscale_numpy(test_data_y.numpy())

    mse, smape = best_model.evaluate(
        x=testdata_x_unscale, y=testdata_y_unscale, metrics=['mse', 'smape'])
    print(f'mse is {np.mean(mse)}, smape is {np.mean(smape)}')
    print(
        f"The hyperparameters of the model include lr:{best_model.config['lr']},"
        f"dropout:{best_model.config['dropout']}, batch_size:{best_model.config['batch_size']}")
    stop_orca_context()
