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
from zoo.automl.feature.identity_transformer import IdentityTransformer

from zoo.automl.model.XGBoost import XGBoost


class XgbPredictor(BasePredictor):
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
                 feature_cols,
                 target_col,
                 model_type="regressor",
                 config=None,
                 name="automl",
                 logs_dir="~/zoo_automl_logs",
                 search_alg=None,
                 search_alg_params=None,
                 scheduler=None,
                 scheduler_params=None,
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
        self.pipeline = None
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.config = config
        self.model_type = model_type
        super().__init__(name=name,
                         logs_dir=logs_dir,
                         search_alg=search_alg,
                         search_alg_params=search_alg_params,
                         scheduler=scheduler,
                         scheduler_params=scheduler_params)

    def create_feature_transformer(self):
        ft = IdentityTransformer(self.feature_cols, self.target_col)
        return ft

    def make_model_fn(self, resources_per_trial):
        model_type = self.model_type
        config = self.config

        def create_model():
            _model = XGBoost(model_type=model_type, config=config)
            if "cpu" in resources_per_trial:
                _model.set_params(n_jobs=resources_per_trial.get("cpu"))
            return _model
        return create_model
