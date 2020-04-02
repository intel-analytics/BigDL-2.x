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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import numpy as np
import tempfile
import zipfile
import os
import shutil
import ray

from zoo.automl.search.abstract import *
from zoo.automl.search.RayTuneSearchEngine import RayTuneSearchEngine
from zoo.automl.common.metrics import Evaluator
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer

from zoo.automl.model import TimeSequenceModel
from zoo.automl.pipeline.time_sequence import TimeSequencePipeline, load_ts_pipeline
from zoo.automl.common.util import *
from abc import ABC, abstractmethod


class Recipe(ABC):
    @abstractmethod
    def search_space(self, all_available_features):
        pass

    @abstractmethod
    def runtime_params(self):
        pass

    def fixed_params(self):
        return None

    def search_algorithm_params(self):
        return None

    def search_algorithm(self):
        return None

    def scheduler_params(self):
        pass


class SmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """
    def __init__(self):
        pass

    def search_space(self, all_available_features):
        return {
            "selected_features": all_available_features,
            "model": "LSTM",
            "lstm_1_units": RandomSample(lambda spec: np.random.choice([32, 64], size=1)[0]),
            "dropout_1": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "lstm_2_units": RandomSample(lambda spec: np.random.choice([32, 64], size=1)[0]),
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "lr": 0.001,
            "batch_size": 1024,
            "epochs": 1,
            "past_seq_len": 2,
        }

    def runtime_params(self):
        return {
            "training_iteration": 1,
            "num_samples": 1,
        }


class MTNetSmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """
    def __init__(self):
        pass

    def search_space(self, all_available_features):
        return {
            "selected_features": all_available_features,
            "model": "MTNet",
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 1,
            "dropout": 0.2,
            "time_step": RandomSample(lambda spec: np.random.choice([3, 4], size=1)[0]),
            "filter_size": 2,
            "long_num": RandomSample(lambda spec: np.random.choice([3, 4], size=1)[0]),
            "ar_size": RandomSample(lambda spec: np.random.choice([2, 3], size=1)[0]),
            "past_seq_len": RandomSample(lambda spec: (spec.config.long_num + 1)
                                         * spec.config.time_step),
        }

    def runtime_params(self):
        return {
            "training_iteration": 1,
            "num_samples": 1,
        }


class GridRandomRecipe(Recipe):
    """
    A recipe involves both grid search and random search.
       tsp = TimeSequencePredictor(...,recipe = GridRandomRecipe(1))

    @param num_rand_samples: number of hyper-param configurations sampled randomly
    @param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
    """

    def __init__(self, num_rand_samples=1, look_back=2):
        self.num_samples = num_rand_samples
        if isinstance(look_back, tuple) and len(look_back) == 2 and \
                isinstance(look_back[0], int) and isinstance(look_back[1], int):
            if look_back[1] < 2:
                raise ValueError("The max look back value should be at least 2")
            if look_back[0] < 2:
                print("The input min look back value is smaller than 2. "
                      "We sample from range (2, {}) instead.".format(look_back[1]))
            self.past_seq_config = RandomSample(
                lambda spec: np.random.randint(look_back[0], look_back[1]+1, size=1)[0])
        elif isinstance(look_back, int):
            if look_back < 2:
                raise ValueError("look back value should not be smaller than 2. "
                                 "Current value is ", look_back)
            self.past_seq_config = look_back
        else:
            raise ValueError("look back is {}.\n "
                             "look_back should be either a tuple with 2 int values:"
                             " (min_len, max_len) or a single int"
                             .format(look_back))

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(low=3, high=len(all_available_features), size=1),
                    replace=False)),

            # --------- model related parameters
            "model": "LSTM",
            "lr": 0.001,
            "lstm_1_units": GridSearch([16, 32]),
            "dropout_1": 0.2,
            "lstm_2_units": GridSearch([16, 32]),
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "batch_size": 1024,
            "epochs": 5,
            "past_seq_len": self.past_seq_config,
        }

    def runtime_params(self):
        return {
            "training_iteration": 10,
            "num_samples": self.num_samples,
        }


class RandomRecipe(Recipe):
    """
    Pure random sample Recipe. Often used as baseline.
       tsp = TimeSequencePredictor(...,recipe = RandomRecipe(5))

    @param num_rand_samples: number of hyper-param configurations sampled randomly
    @param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
    """

    def __init__(self, num_rand_samples=1, look_back=2, reward_metric=-0.05):
        self.num_samples = num_rand_samples
        self.reward_metric = reward_metric
        if isinstance(look_back, tuple) and len(look_back) == 2 and \
                isinstance(look_back[0], int) and isinstance(look_back[1], int):
            if look_back[1] < 2:
                raise ValueError("The max look back value should be at least 2")
            if look_back[0] < 2:
                print("The input min look back value is smaller than 2. "
                      "We sample from range (2, {}) instead.".format(look_back[1]))
            self.past_seq_config = \
                RandomSample(lambda spec:
                             np.random.randint(look_back[0], look_back[1]+1, size=1)[0])
        elif isinstance(look_back, int):
            if look_back < 2:
                raise ValueError("look back value should not be smaller than 2. "
                                 "Current value is ", look_back)
            self.past_seq_config = look_back
        else:
            raise ValueError("look back is {}.\n "
                             "look_back should be either a tuple with 2 int values:"
                             " (min_len, max_len) or a single int"
                             .format(look_back))

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(low=3, high=len(all_available_features), size=1))
            ),

            "model": RandomSample(lambda spec: np.random.choice(["LSTM", "Seq2seq"], size=1)[0]),
            # --------- Vanilla LSTM model parameters
            "lstm_1_units": RandomSample(lambda spec:
                                         np.random.choice([8, 16, 32, 64, 128], size=1)[0]),
            "dropout_1": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "lstm_2_units": RandomSample(lambda spec:
                                         np.random.choice([8, 16, 32, 64, 128], size=1)[0]),
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),

            # ----------- Seq2Seq model parameters
            "latent_dim": RandomSample(lambda spec:
                                       np.random.choice([32, 64, 128, 256], size=1)[0]),
            "dropout": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),

            # ----------- optimization parameters
            "lr": RandomSample(lambda spec: np.random.uniform(0.001, 0.01)),
            "batch_size": RandomSample(lambda spec:
                                       np.random.choice([32, 64, 1024], size=1, replace=False)[0]),
            "epochs": 5,
            "past_seq_len": self.past_seq_config,
        }

    def runtime_params(self):
        return {
            "reward_metric": self.reward_metric,
            "training_iteration": 10,
            "num_samples": self.num_samples,
        }


class BayesRecipe(Recipe):
    """
    A Bayes search Recipe.
       tsp = TimeSequencePredictor(...,recipe = BayesRecipe(5))

    @param num_samples: number of hyper-param configurations sampled
    @param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
    """
    def __init__(self, num_samples=1, look_back=2, reward_metric=-0.05):
        self.num_samples = num_samples
        self.reward_metric = reward_metric
        if isinstance(look_back, tuple) and len(look_back) == 2 and \
                isinstance(look_back[0], int) and isinstance(look_back[1], int):
            if look_back[1] < 2:
                raise ValueError("The max look back value should be at least 2")
            if look_back[0] < 2:
                print("The input min look back value is smaller than 2. "
                      "We sample from range (2, {}) instead.".format(look_back[1]))
            self.bayes_past_seq_config = {"past_seq_len_float": look_back}
            self.fixed_past_seq_config = {}
        elif isinstance(look_back, int):
            if look_back < 2:
                raise ValueError("look back value should not be smaller than 2. "
                                 "Current value is ", look_back)
            self.bayes_past_seq_config = {}
            self.fixed_past_seq_config = {"past_seq_len": look_back}
        else:
            raise ValueError("look back is {}.\n "
                             "look_back should be either a tuple with 2 int values:"
                             " (min_len, max_len) or a single int"
                             .format(look_back))

    def search_space(self, all_available_features):
        feature_space = {"bayes_feature_{}".format(feature): (0.3, 1)
                         for feature in all_available_features}
        other_space = {
            # --------- model parameters
            "lstm_1_units_float": (8, 128),
            "dropout_1": (0.2, 0.5),
            "lstm_2_units_float": (8, 128),
            "dropout_2": (0.2, 0.5),

            # ----------- optimization parameters
            "lr": (0.001, 0.01),
            "batch_size_log": (5, 10),
        }
        total_space = other_space.copy()
        total_space.update(feature_space)
        total_space.update(self.bayes_past_seq_config)
        return total_space

    def fixed_params(self):
        total_fixed_params = {
            "epochs": 5,
            # "batch_size": 1024,
        }
        total_fixed_params.update(self.fixed_past_seq_config)
        return total_fixed_params

    def runtime_params(self):
        return {
            "reward_metric": self.reward_metric,
            "training_iteration": 5,
            "num_samples": self.num_samples,
        }

    def search_algorithm_params(self):
        return {
            "utility_kwargs": {
                "kind": "ucb",
                "kappa": 2.5,
                "xi": 0.0
            }
        }

    def search_algorithm(self):
        return 'BayesOpt'


class TimeSequencePredictor(object):
    """
    Trains a model that predicts future time sequence from past sequence.
    Past sequence should be > 1. Future sequence can be > 1.
    For example, predict the next 2 data points from past 5 data points.
    Output have only one target value (a scalar) for each data point in the sequence.
    Input can have more than one features (value plus several features)
    Example usage:
        tsp = TimeSequencePredictor()
        tsp.fit(input_df)
        result = tsp.predict(test_df)

    """
    def __init__(self,
                 name="automl",
                 logs_dir="~/zoo_automl_logs",
                 future_seq_len=1,
                 dt_col="datetime",
                 target_col="value",
                 extra_features_col=None,
                 drop_missing=True,
                 ):
        """
        Constructor of Time Sequence Predictor
        :param logs_dir where the automl tune logs file located
        :param future_seq_len: the future sequence length to be predicted
        :param dt_col: the datetime index column
        :param target_col: the target col (to be predicted)
        :param extra_features_col: extra features
        :param drop_missing: whether to drop missing values in the input
        """
        self.logs_dir = logs_dir
        self.pipeline = None
        self.future_seq_len = future_seq_len
        self.dt_col = dt_col
        self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.drop_missing = drop_missing
        self.name = name

    def fit(self,
            input_df,
            validation_df=None,
            metric="mse",
            recipe=SmokeRecipe(),
            mc=False,
            resources_per_trial={"cpu": 2},
            distributed=False,
            hdfs_url=None
            ):
        """
        Trains the model for time sequence prediction.
        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param validation_df: validation data
        :param metric: String. Metric used for train and validation. Available values are
                       "mean_squared_error" or "r_square"
        :param recipe: a Recipe object. Various recipes covers different search space and stopping
                      criteria. Default is SmokeRecipe().
        :param resources_per_trial: Machine resources to allocate per trial,
            e.g. ``{"cpu": 64, "gpu": 8}`
        :param distributed: bool. Indicate if running in distributed mode. If true, we will upload
                            models to HDFS.
        :param hdfs_url: the hdfs url used to save file in distributed model. If None, the default
                         hdfs_url will be used.
        :return: self
        """

        self._check_input(input_df, validation_df, metric)
        if distributed:
            if hdfs_url is not None:
                remote_dir = os.path.join(hdfs_url, "ray_results", self.name)
            else:
                remote_dir = os.path.join(os.sep, "ray_results", self.name)
            if self.name not in get_remote_list(os.path.dirname(remote_dir)):
                cmd = "hadoop fs -mkdir -p {}".format(remote_dir)
                process(cmd)
        else:
            remote_dir = None

        self.pipeline = self._hp_search(input_df,
                                        validation_df=validation_df,
                                        metric=metric,
                                        recipe=recipe,
                                        mc=mc,
                                        resources_per_trial=resources_per_trial,
                                        remote_dir=remote_dir)
        return self.pipeline

    def evaluate(self,
                 input_df,
                 metric=None
                 ):
        """
        Evaluate the model on a list of metrics.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param metric: A list of Strings Available string values are "mean_squared_error",
                      "r_square".
        :return: a list of metric evaluation results.
        """
        if not Evaluator.check_metric(metric):
            raise ValueError("metric" + metric + "is not supported")
        return self.pipeline.evaluate(input_df, metric)

    def predict(self,
                input_df):
        """
        Predict future sequence from past sequence.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: a data frame with 2 columns, the 1st is the datetime, which is the last datetime of
            the past sequence.
            values are the predicted future sequence values.
            Example :
            datetime    value_0     value_1   ...     value_2
            2019-01-03  2           3                   9
        """
        return self.pipeline.predict(input_df)

    def _check_input_format(self, input_df):
        if isinstance(input_df, list) and all([isinstance(d, pd.DataFrame) for d in input_df]):
            input_is_list = True
            return input_is_list
        elif isinstance(input_df, pd.DataFrame):
            input_is_list = False
            return input_is_list
        else:
            raise ValueError("input_df should be a data frame or a list of data frames")

    def _check_missing_col(self, input_df):
        cols_list = [self.dt_col, self.target_col]
        if self.extra_features_col is not None:
            if not isinstance(self.extra_features_col, (list,)):
                raise ValueError("extra_features_col needs to be either None or a list")
            cols_list.extend(self.extra_features_col)

        missing_cols = set(cols_list) - set(input_df.columns)
        if len(missing_cols) != 0:
            raise ValueError("Missing Columns in the input data frame:" +
                             ','.join(list(missing_cols)))

    def _check_input(self, input_df, validation_df, metric):
        input_is_list = self._check_input_format(input_df)
        if not input_is_list:
            self._check_missing_col(input_df)
            if validation_df is not None:
                self._check_missing_col(validation_df)
        else:
            for d in input_df:
                self._check_missing_col(d)
            if validation_df is not None:
                for val_d in validation_df:
                    self._check_missing_col(val_d)

        if not Evaluator.check_metric(metric):
            raise ValueError("metric " + metric + " is not supported")

    def _hp_search(self,
                   input_df,
                   validation_df,
                   metric,
                   recipe,
                   mc,
                   resources_per_trial,
                   remote_dir):

        ft = TimeSequenceFeatureTransformer(self.future_seq_len,
                                            self.dt_col,
                                            self.target_col,
                                            self.extra_features_col,
                                            self.drop_missing)
        if isinstance(input_df, list):
            feature_list = ft.get_feature_list(input_df[0])
        else:
            feature_list = ft.get_feature_list(input_df)

        # model = VanillaLSTM(check_optional_config=False)
        model = TimeSequenceModel(check_optional_config=False, future_seq_len=self.future_seq_len)

        # prepare parameters for search engine
        search_space = recipe.search_space(feature_list)
        runtime_params = recipe.runtime_params()
        num_samples = runtime_params['num_samples']
        stop = dict(runtime_params)
        search_algorithm_params = recipe.search_algorithm_params()
        search_algorithm = recipe.search_algorithm()
        fixed_params = recipe.fixed_params()
        del stop['num_samples']

        searcher = RayTuneSearchEngine(logs_dir=self.logs_dir,
                                       resources_per_trial=resources_per_trial,
                                       name=self.name,
                                       remote_dir=remote_dir,
                                       )
        searcher.compile(input_df,
                         search_space=search_space,
                         stop=stop,
                         search_algorithm_params=search_algorithm_params,
                         search_algorithm=search_algorithm,
                         fixed_params=fixed_params,
                         # feature_transformers=TimeSequenceFeatures,
                         feature_transformers=ft,
                         # model=model,
                         future_seq_len=self.future_seq_len,
                         validation_df=validation_df,
                         metric=metric,
                         mc=mc,
                         num_samples=num_samples)
        # searcher.test_run()
        searcher.run()

        best = searcher.get_best_trials(k=1)[0]  # get the best one trial, later could be n
        pipeline = self._make_pipeline(best,
                                       feature_transformers=ft,
                                       model=model,
                                       remote_dir=remote_dir)
        return pipeline

    def _print_config(self, best_config):
        print("The best configurations are:")
        for name, value in best_config.items():
            print(name, ":", value)

    def _make_pipeline(self, trial, feature_transformers, model, remote_dir):
        isinstance(trial, TrialOutput)
        config = convert_bayes_configs(trial.config).copy()
        self._print_config(config)
        if remote_dir is not None:
            all_config = restore_hdfs(trial.model_path,
                                      remote_dir,
                                      feature_transformers,
                                      model,
                                      # config)
                                      )
        else:
            all_config = restore_zip(trial.model_path,
                                     feature_transformers,
                                     model,
                                     # config)
                                     )
        return TimeSequencePipeline(name=self.name,
                                    feature_transformers=feature_transformers,
                                    model=model,
                                    config=all_config)
