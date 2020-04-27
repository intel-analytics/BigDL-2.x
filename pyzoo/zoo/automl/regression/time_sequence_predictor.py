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
from zoo.automl.config.recipe import *


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

        self.pipeline = self._hp_search(
            input_df,
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
        if isinstance(input_df, list) and all(
                [isinstance(d, pd.DataFrame) for d in input_df]):
            input_is_list = True
            return input_is_list
        elif isinstance(input_df, pd.DataFrame):
            input_is_list = False
            return input_is_list
        else:
            raise ValueError(
                "input_df should be a data frame or a list of data frames")

    def _check_missing_col(self, input_df):
        cols_list = [self.dt_col, self.target_col]
        if self.extra_features_col is not None:
            if not isinstance(self.extra_features_col, (list,)):
                raise ValueError(
                    "extra_features_col needs to be either None or a list")
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

        allowed_fit_metrics = ["mse", "mae", "r2"]
        if metric not in allowed_fit_metrics:
            raise ValueError("metric " + metric + " is not supported")

    @staticmethod
    def _get_metric_mode(metric):
        max_mode_metrics = ["r2"]
        min_mode_metrics = ["mse", "mae"]
        if metric in min_mode_metrics:
            return "min"
        elif metric in max_mode_metrics:
            return "max"
        else:
            return ValueError, "metric " + metric + " is not supported"

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
        model = TimeSequenceModel(
            check_optional_config=False,
            future_seq_len=self.future_seq_len)

        # prepare parameters for search engine
        search_space = recipe.search_space(feature_list)
        runtime_params = recipe.runtime_params()
        num_samples = runtime_params['num_samples']
        stop = dict(runtime_params)
        search_algorithm_params = recipe.search_algorithm_params()
        search_algorithm = recipe.search_algorithm()
        fixed_params = recipe.fixed_params()
        del stop['num_samples']

        metric_mode = TimeSequencePredictor._get_metric_mode(metric)
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
                         feature_transformers=ft,
                         future_seq_len=self.future_seq_len,
                         validation_df=validation_df,
                         metric=metric,
                         metric_mode=metric_mode,
                         mc=mc,
                         num_samples=num_samples)
        # searcher.test_run()
        analysis = searcher.run()

        pipeline = self._make_pipeline(analysis,
                                       metric_mode,
                                       feature_transformers=ft,
                                       model=model,
                                       remote_dir=remote_dir)
        return pipeline

    def _print_config(self, best_config):
        print("The best configurations are:")
        for name, value in best_config.items():
            print(name, ":", value)

    def _make_pipeline(self, analysis, metric_mode, feature_transformers, model, remote_dir):
        metric = "reward_metric"
        best_config = analysis.get_best_config(metric=metric, mode=metric_mode)
        best_logdir = analysis.get_best_logdir(metric=metric, mode=metric_mode)
        print("best log dir is ", best_logdir)
        dataframe = analysis.dataframe(metric=metric, mode=metric_mode)
        # print(dataframe)
        model_path = os.path.join(best_logdir, dataframe["checkpoint"].iloc[0])
        config = convert_bayes_configs(best_config).copy()
        self._print_config(config)
        if remote_dir is not None:
            all_config = restore_hdfs(model_path,
                                      remote_dir,
                                      feature_transformers,
                                      model,
                                      # config)
                                      )
        else:
            all_config = restore_zip(model_path,
                                     feature_transformers,
                                     model,
                                     # config)
                                     )
        return TimeSequencePipeline(name=self.name,
                                    feature_transformers=feature_transformers,
                                    model=model,
                                    config=all_config)
