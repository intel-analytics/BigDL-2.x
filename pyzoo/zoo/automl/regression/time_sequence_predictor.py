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
from zoo.automl.regression.base_predictor import BasePredictor
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer

from zoo.automl.model.time_sequence import TimeSequenceModel
import pandas as pd


class TimeSequencePredictor(BasePredictor):
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
                 target_col=["value"],
                 extra_features_col=None,
                 drop_missing=True,
                 search_alg=None,
                 search_alg_params=None,
                 scheduler=None,
                 scheduler_params=None,
                 ):
        self.pipeline = None
        self.future_seq_len = future_seq_len
        self.dt_col = dt_col
        if isinstance(target_col, str):
            self.target_col = [target_col]
        else:
            self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.drop_missing = drop_missing
        super().__init__(name=name,
                         logs_dir=logs_dir,
                         search_alg=search_alg,
                         search_alg_params=search_alg_params,
                         scheduler=scheduler,
                         scheduler_params=scheduler_params)

    def create_feature_transformer(self):
        ft = TimeSequenceFeatureTransformer(self.future_seq_len,
                                            self.dt_col,
                                            self.target_col,
                                            self.extra_features_col,
                                            self.drop_missing)
        return ft

    def make_model_fn(self, resources_per_trial):
        future_seq_len = self.future_seq_len

        def create_model():
            _model = TimeSequenceModel(
                check_optional_config=False,
                future_seq_len=future_seq_len)
            return _model
        return create_model

    def _check_missing_col(self, df):
        cols_list = [self.dt_col] + self.target_col
        if self.extra_features_col is not None:
            if not isinstance(self.extra_features_col, (list,)):
                raise ValueError(
                    "extra_features_col needs to be either None or a list")
            cols_list.extend(self.extra_features_col)

        missing_cols = set(cols_list) - set(df.columns)
        if len(missing_cols) != 0:
            raise ValueError("Missing Columns in the input data frame:" +
                             ','.join(list(missing_cols)))

    def _check_df(self, df):
        super()._check_df(df)
        self._check_missing_col(df=df)
