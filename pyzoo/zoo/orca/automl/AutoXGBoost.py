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

from zoo.automl.regression.xgb_predictor import XgbPredictor
from zoo.automl.config.recipe import *


class AutoXGBoost(object):
    @staticmethod
    def regressor(feature_cols,
                  target_col,
                  config=None,
                  name="automl",
                  logs_dir="~/zoo_automl_logs",
                  search_alg=None,
                  search_alg_params=None,
                  scheduler=None,
                  scheduler_params=None,
                  ):
        tsp = XgbPredictor(feature_cols, target_col, 'regressor',
                           config, name, logs_dir,
                           search_alg=search_alg,
                           search_alg_params=search_alg_params,
                           scheduler=scheduler,
                           scheduler_params=scheduler_params)
        return tsp

    @staticmethod
    def classifier(feature_cols,
                   target_col,
                   config=None,
                   name="automl",
                   logs_dir="~/zoo_automl_logs",
                   search_alg=None,
                   search_alg_params=None,
                   scheduler=None,
                   scheduler_params=None,
                   ):
        tsp = XgbPredictor(feature_cols, target_col, 'classifier',
                           config, name, logs_dir,
                           search_alg=search_alg,
                           search_alg_params=search_alg_params,
                           scheduler=scheduler,
                           scheduler_params=scheduler_params)
        return tsp
