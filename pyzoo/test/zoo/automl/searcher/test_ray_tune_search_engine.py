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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.search import SearchEngineFactory
from zoo.automl.search.ray_tune_search_engine import RayTuneSearchEngine
from zoo.automl.model import PytorchModelBuilder
from zoo.zouwu.model.VanillaLSTM_pytorch import model_creator as LSTM_model_creator
import torch
import torch.nn as nn
from zoo.automl.recipe.base import Recipe
from zoo.orca.automl import hp
import pandas as pd
import numpy as np
from zoo.orca import init_orca_context, stop_orca_context
from zoo.zouwu.feature.time_sequence import TimeSequenceFeatureTransformer


class SimpleRecipe(Recipe):
    def __init__(self, stop_metric=0):
        super().__init__()
        self.num_samples = 2
        self.training_iteration = 20
        self.reward_metric = stop_metric

    def search_space(self):
        return {
            "lr": hp.uniform(0.001, 0.01),
            "batch_size": hp.choice([32, 64]),
        }


def create_lstm_recipe(input_dim):
    class LSTMRecipe(Recipe):
        def __init__(self):
            super().__init__()
            self.num_samples = 2
            self.training_iteration = 20

        def search_space(self):
            return {
                "lr": hp.uniform(0.001, 0.01),
                "batch_size": hp.choice([32, 64]),
                "input_dim": input_dim,
                "output_dim": 1
            }
    return LSTMRecipe()


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
                     metric="mse",
                     name="demo"):
    modelBuilder = PytorchModelBuilder(model_creator=model_creator,
                                       optimizer_creator=optimizer_creator,
                                       loss_creator=loss_creator)
    searcher = SearchEngineFactory.create_engine(backend="ray",
                                                 logs_dir="~/zoo_automl_logs",
                                                 resources_per_trial={"cpu": 2},
                                                 name=name)
    searcher.compile(data=data,
                     model_create_func=modelBuilder,
                     recipe=recipe,
                     feature_transformers=feature_transformer,
                     metric=metric)
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
        searcher = prepare_searcher(data=data_with_val,
                                    name='test_ray_numpy_with_val',
                                    recipe=SimpleRecipe())
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        assert best_trials is not None

    def test_dataframe_input(self):
        train_x, train_y, val_x, val_y = get_np_input()
        dataframe_with_val = {'df': pd.DataFrame({'x': train_x, 'y': train_y}),
                              'val_df': pd.DataFrame({'x': val_x, 'y': val_y}),
                              'feature_cols': ['x'],
                              'target_col': 'y'}
        searcher = prepare_searcher(data=dataframe_with_val,
                                    name='test_ray_dataframe_with_val',
                                    recipe=SimpleRecipe())
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        assert best_trials is not None

    def test_dataframe_input_with_datetime(self):
        train_df, validation_df, future_seq_len = get_ts_input()
        dataframe_with_datetime = {'df': train_df, 'val_df': validation_df}
        ft = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len,
                                            dt_col="datetime",
                                            target_col="value")
        input_dim = len(ft.get_feature_list()) + 1
        searcher = prepare_searcher(data=dataframe_with_datetime,
                                    model_creator=LSTM_model_creator,
                                    name='test_ray_dateframe_with_datetime_with_val',
                                    recipe=create_lstm_recipe(input_dim),
                                    feature_transformer=ft)
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        assert best_trials is not None

    def test_searcher_metric(self):
        train_x, train_y, val_x, val_y = get_np_input()
        data_with_val = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}

        # test metric name is returned and max mode can be stopped
        searcher = prepare_searcher(data=data_with_val,
                                    name='test_searcher_metric_name',
                                    metric='mse',
                                    recipe=SimpleRecipe(stop_metric=float('-inf')))  # stop at once
        analysis = searcher.run()
        sorted_results = list(map(lambda x: x.last_result['mse'],
                                  RayTuneSearchEngine._get_sorted_trials(analysis.trials,
                                                                         metric='mse',
                                                                         mode="min")))

        # assert metric name is reported
        assert 'mse' in analysis.trials[0].last_result.keys()
        # assert _get_sorted_trials get increasing result
        assert all(sorted_results[i] <= sorted_results[i+1] for i in range(len(sorted_results)-1))
        # assert _get_best_result get minimum result
        assert RayTuneSearchEngine._get_best_result(analysis.trials,
                                                    metric='mse',
                                                    mode="min")['mse'] == sorted_results[0]
        assert all(analysis.trials[i].last_result['mse'] >=
                   analysis.trials[i].last_result['best_mse'] for i in range(len(sorted_results)))
        # assert the trail stop at once since mse has mode of 'min'
        assert analysis.trials[0].last_result['iterations_since_restore'] == 1

        # max mode metric with stop
        searcher = prepare_searcher(data=data_with_val,
                                    name='test_searcher_metric_name',
                                    metric='r2',
                                    recipe=SimpleRecipe(stop_metric=0))  # stop at once
        analysis = searcher.run()
        sorted_results = list(map(lambda x: x.last_result['r2'],
                                  RayTuneSearchEngine._get_sorted_trials(analysis.trials,
                                                                         metric='r2',
                                                                         mode="max")))

        # assert metric name is reported
        assert 'r2' in analysis.trials[0].last_result.keys()
        # assert _get_sorted_trials get decreasing result
        assert all(sorted_results[i] >= sorted_results[i+1] for i in range(len(sorted_results)-1))
        # assert _get_best_result get maximum result
        assert RayTuneSearchEngine._get_best_result(analysis.trials,
                                                    metric='r2',
                                                    mode="max")['r2'] == sorted_results[0]
        assert all(analysis.trials[i].last_result['r2'] <=
                   analysis.trials[i].last_result['best_r2'] for i in range(len(sorted_results)))
        # assert the trail stop at once since mse has mode of 'max'
        assert analysis.trials[0].last_result['iterations_since_restore'] == 1

        # test min mode metric without stop
        searcher = prepare_searcher(data=data_with_val,
                                    name='test_searcher_metric_name',
                                    metric='mae',
                                    recipe=SimpleRecipe(stop_metric=0))  # never stop by metric
        analysis = searcher.run()
        sorted_results = list(map(lambda x: x.last_result['mae'],
                                  RayTuneSearchEngine._get_sorted_trials(analysis.trials,
                                                                         metric='mae',
                                                                         mode="min")))

        # assert metric name is reported
        assert 'mae' in analysis.trials[0].last_result.keys()
        # assert _get_sorted_trials get increasing result
        assert all(sorted_results[i] <= sorted_results[i+1] for i in range(len(sorted_results)-1))
        # assert _get_best_result get minimum result
        assert RayTuneSearchEngine._get_best_result(analysis.trials,
                                                    metric='mae',
                                                    mode="min")['mae'] == sorted_results[0]
        assert all(analysis.trials[i].last_result['mae'] >=
                   analysis.trials[i].last_result['best_mae'] for i in range(len(sorted_results)))
        # assert the trail stop at once since mse has mode of 'min'
        assert analysis.trials[0].last_result['iterations_since_restore'] == 20
