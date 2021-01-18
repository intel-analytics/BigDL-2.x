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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.search import SearchEngineFactory
from zoo.automl.model import ModelBuilder
from zoo.automl.model.VanillaLSTM_pytorch import model_creator as LSTM_model_creator
import torch
import torch.nn as nn
from zoo.automl.config.recipe import Recipe
from ray import tune
import pandas as pd
import numpy as np
from zoo.orca import init_orca_context, stop_orca_context
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
import json


class SimpleRecipe(Recipe):
    def __init__(self):
        super().__init__()
        self.num_samples = 2
        self.training_iteration = 20

    def search_space(self, all_available_features):
        return {
            "lr": tune.uniform(0.001, 0.01),
            "batch_size": tune.choice([32, 64]),
            "selected_features": json.dumps(all_available_features),
            "input_dim": len(all_available_features)+1 if all_available_features else 1,
            "output_dim": 1
        }


def linear_model_creator(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(config.get("input_dim", 1), config.get("output_dim", 1))


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def loss_creator(config):
    return nn.MSELoss()


def prepare_searcher(data,
                     model_creator=linear_model_creator,
                     optimizer_creator=optimizer_creator,
                     loss_creator=loss_creator,
                     feature_transformer=None,
                     recipe=SimpleRecipe(),
                     name="demo"):
    modelBuilder = ModelBuilder.from_pytorch(model_creator=model_creator,
                                             optimizer_creator=optimizer_creator,
                                             loss_creator=loss_creator)
    searcher = SearchEngineFactory.create_engine(backend="ray",
                                                 logs_dir="~/zoo_automl_logs",
                                                 resources_per_trial={"cpu": 2},
                                                 name=name)
    search_space = recipe.search_space(feature_transformer.get_feature_list())\
        if feature_transformer else None
    searcher.compile(data=data,
                     model_create_func=modelBuilder,
                     recipe=recipe,
                     feature_transformers=feature_transformer,
                     search_space=search_space)
    return searcher


def get_np_input():
    def get_linear_data(a, b, size):
        x = np.arange(0, 10, 10 / size)
        y = a*x + b
        return x, y
    train_x, train_y = get_linear_data(2, 5, 1000)
    val_x, val_y = get_linear_data(2, 5, 400)
    return train_x, train_y, val_x, val_y


def get_ts_input():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range(
        '1/1/2019', periods=sample_num), "value": np.random.randn(sample_num)})
    val_sample_num = np.random.randint(20, 30)
    validation_df = pd.DataFrame({"datetime": pd.date_range(
        '1/1/2019', periods=val_sample_num), "value": np.random.randn(val_sample_num)})
    future_seq_len = 1
    return train_df, validation_df, future_seq_len


class TestRayTuneSearchEngine(ZooTestCase):

    def setup_method(self, method):
        init_orca_context(init_ray_on_spark=True)

    def teardown_method(self, method):
        stop_orca_context()

    def test_numpy_input(self):
        train_x, train_y, val_x, val_y = get_np_input()
        data_with_val = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
        searcher = prepare_searcher(data=data_with_val, name='test_ray_numpy_with_val')
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        assert best_trials is not None

    def test_dataframe_input(self):
        train_x, train_y, val_x, val_y = get_np_input()
        dataframe_with_val = {'df': pd.DataFrame({'x': train_x, 'y': train_y}),
                              'val_df': pd.DataFrame({'x': val_x, 'y': val_y}),
                              'feature_cols': ['x'],
                              'target_col': 'y'}
        searcher = prepare_searcher(data=dataframe_with_val, name='test_ray_dataframe_with_val')
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        assert best_trials is not None

    def test_dataframe_input_with_datetime(self):
        train_df, validation_df, future_seq_len = get_ts_input()
        dataframe_with_datetime = {'df': train_df, 'val_df': validation_df}
        ft = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len,
                                            dt_col="datetime",
                                            target_col="value")
        searcher = prepare_searcher(data=dataframe_with_datetime,
                                    model_creator=LSTM_model_creator,
                                    name='test_ray_dateframe_with_datetime_with_val',
                                    feature_transformer=ft)
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        assert best_trials is not None
