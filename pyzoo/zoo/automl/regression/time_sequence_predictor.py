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
import tempfile
import zipfile
import os
import shutil
from zoo.automl.search.abstract import *
from zoo.automl.search.RayTuneSearchEngine import RayTuneSearchEngine

from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.automl.model.time_sequence import TimeSequenceModel
from zoo.automl.model import VanillaLSTM
from zoo.automl.pipeline.time_sequence import TimeSequencePipeline
from zoo.automl.common.util import *


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
                 logs_dir="~/zoo_automl_logs",
                 future_seq_len=1,
                 dt_col="datetime",
                 target_col="value",
                 extra_features_col=None,
                 drop_missing=True):
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

    def fit(self,
            input_df,
            validation_df=None,
            metric="mean_squared_error"):
        """
        Trains the model for time sequence prediction.
        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param validation_df: validation data
        :param metric: String. Metric used for train and validation. Available values are "mean_squared_error" or
        "r_square"
        :return: self
        """
        self.pipeline = self._hp_search(input_df,
                                        validation_df=validation_df,
                                        metric=metric)
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
        :param metric: A list of Strings Available string values are "mean_squared_error", "r_square".
        :return: a list of metric evaluation results.
        """
        return self.pipeline.evaluate(input_df, metric)

    def predict(self,
                input_df):
        """
        Predict future sequence from past sequence.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: a data frame with 2 columns, the 1st is the datetime, which is the last datetime of the past sequence.
            values are the predicted future sequence values.
            Example :
            datetime    values
            2019-01-03  np.array([2, 3, ... 9])
        """
        return self.pipeline.predict(input_df)

    def _hp_search(self,
                   input_df,
                   validation_df,
                   metric):
        # features
        # feature_list = ["WEEKDAY(datetime)", "HOUR(datetime)",
        #                "PERCENTILE(value)", "IS_WEEKEND(datetime)",
        #                "IS_AWAKE(datetime)", "IS_BUSY_HOURS(datetime)"
        #                # "DAY(datetime)","MONTH(datetime)", #probabaly not useful
        #                ]
        # target_list = ["value"]
        # ft = TimeSequenceFeatures(self.future_seq_len, self.dt_col, self.target_col, self.extra_features_col)

        # ft = DummyTimeSequenceFeatures(file_path='../../../../data/nyc_taxi_rolled_split.npz')
        ft = TimeSequenceFeatureTransformer(self.future_seq_len, self.dt_col, self.target_col, self.extra_features_col, self.drop_missing)

        feature_list = ft.get_feature_list(input_df)
        # model
        model = VanillaLSTM(check_optional_config=False, future_seq_len=self.future_seq_len)

        search_space = {
            # -------- feature related parameters
            "selected_features": RandomSample(
               lambda spec: np.random.choice(
                   feature_list,
                   size=np.random.randint(low=3, high=len(feature_list), size=1),
                   replace=False)),

            # --------- model related parameters
            # 'input_shape_x': x_train.shape[1],
            # 'input_shape_y': x_train.shape[-1],
            'out_units': self.future_seq_len,
            "lr": 0.001,
            "lstm_1_units": GridSearch([16, 32]),
            "dropout_1": 0.2,
            "lstm_2_units": 10,
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "batch_size": 1024,
        }

        stop = {
            "reward_metric": -0.05,
            "training_iteration": 10
        }

        searcher = RayTuneSearchEngine(logs_dir=self.logs_dir, ray_num_cpus=6, resources_per_trial={"cpu": 2})
        searcher.compile(input_df,
                         search_space=search_space,
                         stop=stop,
                         # feature_transformers=TimeSequenceFeatures,
                         feature_transformers=ft,  # use dummy features for testing the rest
                         model=model,
                         validation_df=validation_df,
                         metric=metric)
        # searcher.test_run()

        trials = searcher.run()
        best = searcher.get_best_trials(k=1)[0]  # get the best one trial, later could be n
        pipeline = self._make_pipeline(best,
                                       feature_transformers=ft,
                                       # feature_transformers=TimeSequenceFeatures(
                                       #     file_path='../../../../data/nyc_taxi_rolled_split.npz'),
                                       model=VanillaLSTM(check_optional_config=False))
        return pipeline

    def _print_config(self, best_config):
        print("The best configurations are:")
        for name, value in best_config.items():
            print(name, ":", value)

    def _make_pipeline(self, trial, feature_transformers, model):
        isinstance(trial, TrialOutput)
        # TODO we need to save fitted parameters (not in config, e.g. min max for scalers, model weights)
        # for both transformers and model
        # temp restore from two files

        self._print_config(trial.config)
        dirname = tempfile.mkdtemp(prefix="automl_")
        try:
            with zipfile.ZipFile(trial.model_path) as zf:
                zf.extractall(dirname)
                # print("files are extracted into" + dirname)
                # print(os.listdir(dirname))

            model_path = os.path.join(dirname, "weights_tune.h5")
            config_path = os.path.join(dirname, "local_config.json")
            local_config = load_config(config_path)
            all_config = trial.config.copy()
            all_config.update(local_config)
            model.restore(model_path, **all_config)
            feature_transformers.restore(**all_config)
        finally:
            shutil.rmtree(dirname)

        # model.restore(model_path)
        # feature_transformers.restore(config_path, **trial.config)

        return TimeSequencePipeline(feature_transformers=feature_transformers, model=model, config=all_config)


if __name__ == "__main__":
    train_df, val_df, test_df = load_nytaxi_data_df("../../../../data/nyc_taxi.csv")
    # print(train_df.describe())
    # print(test_df.describe())

    tsp = TimeSequencePredictor(dt_col="datetime",
                                target_col="value",
                                extra_features_col=None, )
    pipeline = tsp.fit(train_df,
                       validation_df=val_df,
                       metric="mean_squared_error")

    print("evaluate:", pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"]))
    pred = pipeline.predict(test_df)
    print("predict:", pred.shape)

    save_pipeline_file = "../../../saved_pipeline/"
    pipeline.save(save_pipeline_file)

    new_pipeline = TimeSequencePipeline()
    new_pipeline.restore(save_pipeline_file)
    print("evaluate:", new_pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"]))

    new_pred = new_pipeline.predict(test_df)
    print("predict:", pred.shape)
    np.testing.assert_allclose(pred["value"].values, new_pred["value"].values)
