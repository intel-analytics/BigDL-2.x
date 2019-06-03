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


import numpy as np
import pandas as pd
import os
import json


def load_nytaxi_data_df(csv_path=None, val_split_ratio=0, test_split_ratio=0.1):
    if csv_path is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(curr_dir, "../../../../data/nyc_taxi.csv")
    full_df = pd.read_csv(csv_path)
    full_df['datetime'] = pd.to_datetime(full_df['timestamp'])

    val_size = int(len(full_df) * val_split_ratio)
    test_size = int(len(full_df) * test_split_ratio)

    train_df = full_df.iloc[:-(test_size+val_size)]
    val_df = full_df.iloc[-(test_size+val_size):-test_size]
    test_df = full_df.iloc[-test_size:]

    output_train_df = train_df[["datetime", "value"]].copy()
    output_val_df = val_df[["datetime", "value"]].copy()
    output_test_df = test_df[["datetime", "value"]].copy()
    output_val_df = output_val_df.reset_index(drop=True)
    output_test_df = output_test_df.reset_index(drop=True)
    return output_train_df, output_val_df, output_test_df


def load_nytaxi_data(npz_path):
    data = np.load(npz_path)
    return data['x_train'], data['y_train'], data['x_test'], data['y_test']


class NumpyEncoder(json.JSONEncoder):
    """
    convert numpy array to list for JSON serialize
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_config(file_path, config, replace=False):
    if os.path.isfile(file_path) and not replace:
        with open(file_path, "r") as input_file:
            data = json.load(input_file)
        config.update(data)

    with open(file_path, "w") as output_file:
        json.dump(config, output_file, cls=NumpyEncoder)


def load_config(file_path):
    with open(file_path, "r") as input_file:
        data = json.load(input_file)
    return data
