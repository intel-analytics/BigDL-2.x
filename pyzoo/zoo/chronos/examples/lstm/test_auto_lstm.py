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
import time
from matplotlib.pyplot import sca

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import argparse
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from zoo.chronos.autots.model.auto_lstm import AutoLSTM
from zoo.orca.automl import hp
from zoo.orca import init_orca_context, stop_orca_context


input_feature_dim = 6
output_feature_dim = 2
past_seq_len = 5
future_seq_len = 1


def get_random_data(size):
    X = np.random.randn(size, past_seq_len, input_feature_dim)
    y = np.random.randn(size, future_seq_len, output_feature_dim)
    return np.array(X),np.array(y)


def get_data(data_selection='train'):
    load_data = sns.load_dataset('mpg')
    dataset = load_data[['displacement', 'horsepower',
                        'weight', 'acceleration', 'model_year', 'mpg']]
    _dataset = dataset.loc[~np.isnan(dataset).any(axis=1), :]
    train_dataset, valid_dataset = train_test_split(
        _dataset, train_size=0.8, test_size=0.2, shuffle=True)
    dataset = (train_dataset if data_selection == 'train' else valid_dataset)
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    return dataset


class AutoLSTMDataset(Dataset):

    def __init__(self, data_selection):
        self.data = get_data(data_selection)

    def __len__(self):
        return self.data.shape[0] - past_seq_len - future_seq_len - 1

    def __getitem__(self, idx):
        X = self.data[idx:idx+past_seq_len, :input_feature_dim]
        y = self.data[idx+past_seq_len:idx+past_seq_len +
                      future_seq_len, -output_feature_dim:]
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def train_dataloader_creator(config):
    return DataLoader(AutoLSTMDataset('train'),
                      batch_size=config["batch_size"],
                      shuffle=True)


def valid_dataloader_creator(config):
    return DataLoader(AutoLSTMDataset('valid'),
                      batch_size=config["batch_size"],
                      shuffle=True)


def test_fit_np_data(args):
    auto_lstm = AutoLSTM(input_feature_num=input_feature_dim,
                         output_target_num=output_feature_dim,
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
    auto_lstm.fit(data=get_random_data(1000),
                  epochs=args.epoch,
                  batch_size=hp.choice([32, 64]),
                  validation_data=get_random_data(400),
                  n_sampling=args.n_sampling
                  )
    best_model = auto_lstm.get_best_model()
    best_model_mse = best_model.evaluate(
        x=get_random_data(1000)[0], y=get_random_data(1000)[1], metrics=['mse'])
    print(f"Evaluation result is {np.mean(best_model_mse[0][0]):.2f}")
    print(
        f'The hyperparameters of the model include lr:{best_model.config["lr"]}, dropout:{best_model.config["dropout"]}, batch_size:{best_model.config["batch_size"]}')
    best_model_path = '/tmp/Checkpoint/auto_lstm_best_model_' + \
        str(time.time()) + '.ckpt'
    if not os.path.exists('/tmp/Checkpoint'):
        os.makedirs('/tmp/Checkpoint')
    best_model.save(best_model_path)


def test_fit_mpg_data(args):
    auto_lstm = AutoLSTM(input_feature_num=input_feature_dim,
                         output_target_num=output_feature_dim,
                         optimizer='Adam',
                         loss=torch.nn.MSELoss(),
                         metric="mse",
                         hidden_dim=hp.grid_search([32, 64]),
                         layer_num=hp.randint(1, 3),
                         lr=hp.choice([0.001, 0.003, 0.01]),
                         dropout=hp.uniform(0.1, 0.2),
                         backend=args.backend,
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
    best_model_mse = best_model._validate(
        valid_dataloader_creator(best_model.config), metric='mse')
    print(f'Evaluation result is {best_model_mse["mse"]:.2f}')
    print(
        f'The hyperparameters of the model include lr:{best_model.config["lr"]}, dropout:{best_model.config["dropout"]}, batch_size:{best_model.config["batch_size"]}')


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

    parser.add_argument('--backend', type=str, default="torch",
                        help="The backend of the lstm model. We only support backend as 'torch' for now.")
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
    test_fit_np_data(args)
    test_fit_mpg_data(args)
    stop_orca_context()
