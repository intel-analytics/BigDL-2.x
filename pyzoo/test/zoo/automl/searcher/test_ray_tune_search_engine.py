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
from zoo.automl.model.VanillaLSTM_pytorch import LSTMModel
import torch
import torch.nn as nn
from zoo.automl.config.recipe import Recipe, BayesRecipe
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
            "input_dim": len(all_available_features)+1 if all_available_features else None,
            "output_dim": 1
        }

def model_creator(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))

def LSTM_model_creator(config):
    return LSTMModel(input_dim=config["input_dim"],
                    hidden_dim=config.get("hidden_dim", 32),
                    layer_num=config.get("layer_num", 2),
                    dropout=config.get("dropout", 0.2),
                    output_dim=config["output_dim"])

def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def loss_creator(config):
    return nn.MSELoss()

class TestRayTuneSearchEngine(ZooTestCase):

    def setup_method(self, method):
        init_orca_context(init_ray_on_spark=True)

    def teardown_method(self, method):
        stop_orca_context()

    def get_data(self):
        def get_linear_data(a, b, size):
            x = np.arange(0, 10, 10 / size, dtype=np.float32)
            y = a*x + b
            return x, y
        train_x, train_y = get_linear_data(2, 5, 1000)
        val_x, val_y = get_linear_data(2, 5, 400)
        return train_x, train_y, val_x, val_y
    
    def create_date_dataset(self):
        sample_num = np.random.randint(100, 200)
        train_df = pd.DataFrame({"datetime": pd.date_range(
            '1/1/2019', periods=sample_num), "value": np.random.randn(sample_num).astype(np.float32)})
        val_sample_num = np.random.randint(20, 30)
        validation_df = pd.DataFrame({"datetime": pd.date_range(
            '1/1/2019', periods=val_sample_num), "value": np.random.randn(val_sample_num).astype(np.float32)})
        future_seq_len = 1
        return train_df, validation_df, future_seq_len

    def test_numpy_input(self):
        modelBuilder = ModelBuilder.from_pytorch(model_creator=model_creator,
                                                optimizer_creator=optimizer_creator,
                                                loss_creator=loss_creator)

        train_x, train_y, val_x, val_y = self.get_data()

        searcher = SearchEngineFactory.create_engine(backend="ray",
                                                    logs_dir="~/zoo_automl_logs",
                                                    resources_per_trial={"cpu": 2},
                                                    name="test_ray_numpy_with_val")
        data_with_val = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
        searcher.compile(data=data_with_val,
                        model_create_func=modelBuilder,
                        recipe=SimpleRecipe())
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        print("test_ray_numpy_with_val:", best_trials[0].config)
        

    def test_dataframe_input(self):
        modelBuilder = ModelBuilder.from_pytorch(model_creator=model_creator,
                                                optimizer_creator=optimizer_creator,
                                                loss_creator=loss_creator)

        train_x, train_y, val_x, val_y = self.get_data()

        searcher = SearchEngineFactory.create_engine(backend="ray",
                                                    logs_dir="~/zoo_automl_logs",
                                                    resources_per_trial={"cpu": 2},
                                                    name="test_ray_dateframe_with_val")
        dataframe_with_val = {'df': pd.DataFrame({'x': train_x, 'y': train_y}), 
                              'val_df': pd.DataFrame({'x': val_x, 'y': val_y}),
                              'feature_cols': ['x'],
                              'target_col': 'y'}
        searcher.compile(data=dataframe_with_val,
                        model_create_func=modelBuilder,
                        recipe=SimpleRecipe())
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        print("test_ray_dataframe_with_val:", best_trials[0].config)

    def test_dataframe_input_with_datetime(self):
        modelBuilder = ModelBuilder.from_pytorch(model_creator=LSTM_model_creator,
                                                optimizer_creator=optimizer_creator,
                                                loss_creator=loss_creator)

        train_df, validation_df, future_seq_len = self.create_date_dataset()

        ft = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len,
                                       dt_col="datetime",
                                       target_col="value")
        dataframe_with_val = {'df': train_df, 'val_df': validation_df}
        searcher = SearchEngineFactory.create_engine(backend="ray",
                                                    logs_dir="~/zoo_automl_logs",
                                                    resources_per_trial={"cpu": 2},
                                                    name="test_ray_dateframe_with_datetime_with_val")
        recipe = SimpleRecipe()
        searcher.compile(data=dataframe_with_val,
                        model_create_func=modelBuilder,
                        recipe=recipe,
                        feature_transformers=ft,
                        search_space=recipe.search_space(ft.get_feature_list()))
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        print("test_ray_dataframe_with_val:", best_trials[0].config)