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
import shutil
import tempfile
import zipfile

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
            old_config = json.load(input_file)
        old_config.update(config)
        config = old_config.copy()

    with open(file_path, "w") as output_file:
        json.dump(config, output_file, cls=NumpyEncoder)


def load_config(file_path):
    with open(file_path, "r") as input_file:
        data = json.load(input_file)
    return data


def save(file, feature_transformers=None, model=None, config=None):
    if not os.path.isdir(file):
        os.mkdir(file)
    config_path = os.path.join(file, "config.json")
    model_path = os.path.join(file, "weights_tune.h5")
    if feature_transformers is not None:
        feature_transformers.save(config_path, replace=True)
    if model is not None:
        model.save(model_path, config_path)
    if config is not None:
        save_config(config_path, config)


def save_zip(file, feature_transformers=None, model=None, config=None):
    dirname = tempfile.mkdtemp(prefix="automl_save_")
    try:
        save(dirname,
             feature_transformers=feature_transformers,
             model=model,
             config=config)
        with zipfile.ZipFile(file, 'w') as f:
            for dirpath, dirnames, filenames in os.walk(dirname):
                for filename in filenames:
                    f.write(os.path.join(dirpath, filename), filename)
    finally:
        shutil.rmtree(dirname)


def restore(file, feature_transformers=None, model=None, config=None):
    model_path = os.path.join(file, "weights_tune.h5")
    config_path = os.path.join(file, "config.json")
    local_config = load_config(config_path)
    if config is not None:
        all_config = config.copy()
        all_config.update(local_config)
    else:
        all_config = local_config
    if model:
        model.restore(model_path, **all_config)
    if feature_transformers:
        feature_transformers.restore(**all_config)
    return all_config


def restore_zip(file, feature_transformers=None, model=None, config=None):
    dirname = tempfile.mkdtemp(prefix="automl_save_")
    try:
        with zipfile.ZipFile(file) as zf:
            zf.extractall(dirname)

        all_config = restore(dirname, feature_transformers, model, config)
    finally:
        shutil.rmtree(dirname)
    return all_config

