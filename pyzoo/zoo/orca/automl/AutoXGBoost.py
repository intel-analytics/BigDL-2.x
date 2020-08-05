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

from zoo.automl.regression.xgbregressor_predictor import XgbRegressorPredictor
from zoo.automl.config.recipe import *


class AutoXGBoost(object):
    def regressor(self,
                  feature_cols,
                  target_col,
                  name="automl",
                  logs_dir="~/zoo_automl_logs"):
        # models
        self.tsp = XgbRegressorPredictor(feature_cols, target_col,
                                         name, logs_dir)
        return self.tsp

    def fit(self,
            input_df,
            validation_df,
            metric="rmse",
            recipe=XgbRegressorGridRandomRecipe(),
            mc=False,
            resources_per_trial={"cpu": 2},
            distributed=False,
            hdfs_url=None):
        pipeline = self.tsp.fit(input_df, validation_df, metric,
                                recipe, mc, resources_per_trial,
                                distributed, hdfs_url)
        return pipeline

    def predict(self, input_df):
        output = self.tsp.predict(input_df)
        return output

    def evaluate(self,
                 input_df,
                 metric=None):
        res = self.tsp.fit(input_df, metric)
        return res

    def getModel(self):
        return self.tsp.pipeline.model
