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

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import argparse
import seaborn as sns

from zoo.chronos.autots.model.auto_lstm import AutoLSTM
from zoo.orca.automl import hp
from zoo.orca import init_orca_context, stop_orca_context


input_feature_dim = 6
output_feature_dim = 2
past_seq_len = 5
future_seq_len = 1

# Prepare dataset
load_data = sns.load_dataset('mpg')
dataset = load_data[['displacement', 'horsepower',
                     'weight', 'acceleration', 'model_year', 'mpg']]
dataset = dataset.loc[~np.isnan(dataset).any(axis=1), :]
_train_data, _val_data = train_test_split(
    dataset, train_size=0.8, test_size=0.2, shuffle=True)
# standardization
scaler = StandardScaler()
train_data, val_data = scaler.fit_transform(
    _train_data), scaler.fit_transform(_val_data)


def get_x_y(data):
    X, y = [], []
    for i in range(len(data) - past_seq_len - future_seq_len - 1):
        X.append(data[i: i + past_seq_len, 0: input_feature_dim])
        y.append(data[i + past_seq_len: i +
                 past_seq_len + future_seq_len, -output_feature_dim:])
    return np.array(X), np.array(y)


class RandomDataset(Dataset):

    def __init__(self, data):
        x, y = get_x_y(data)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_dataloader_creator(config):
    return DataLoader(RandomDataset(data=train_data),
                      batch_size=config["batch_size"],
                      shuffle=True)


def valid_dataloader_creator(config):
    return DataLoader(RandomDataset(data=val_data),
                      batch_size=config["batch_size"],
                      shuffle=True)


def test_fit_np(args):
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
    auto_lstm.fit(data=get_x_y(train_data),
                  epochs=args.epoch,
                  batch_size=hp.choice([32, 64]),
                  validation_data=get_x_y(val_data),
                  n_sampling=args.n_sampling
                  )
    best_model = auto_lstm.get_best_model()
    best_model_rmse = best_model.evaluate(
        x=get_x_y(train_data)[0], y=get_x_y(train_data)[1], metrics=['rmse'])
    print(f"Evaluation result is {np.mean(best_model_rmse[0][0]):.2f}")
    print(f'Some hyperparameters of the model are {best_model.config}')


def test_fit_data_creator(args):
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
        valid_dataloader_creator(best_model.config), metric='rmse')
    print(f'Evaluation result is {best_model_mse["rmse"]:.2f}')
    print(f'Some hyperparameters of the model are {best_model.config}')


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
    test_fit_np(args)
    test_fit_data_creator(args)
    stop_orca_context()
