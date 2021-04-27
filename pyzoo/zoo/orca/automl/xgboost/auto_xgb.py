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
from zoo.automl.model import XGBoostModelBuilder
from zoo.orca.automl.auto_estimator import AutoEstimator


class AutoXGBClassifier(AutoEstimator):
    def __init__(self,
                 logs_dir="~/auto_xgb_classifier_logs",
                 n_cpus=None,
                 name=None,
                 **xgb_configs
                 ):
        xgb_model_builder = XGBoostModelBuilder(model_type='classifier',
                                                n_cpus=n_cpus,
                                                **xgb_configs)
        resources_per_trial = {"cpu": n_cpus} if n_cpus else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)


class AutoXGBRegressor(AutoEstimator):
    def __init__(self,
                 logs_dir="~/auto_xgb_regressor_logs",
                 n_cpus=None,
                 name=None,
                 **xgb_configs
                 ):
        xgb_model_builder = XGBoostModelBuilder(model_type='regressor',
                                                n_cpus=n_cpus,
                                                **xgb_configs)
        resources_per_trial = {"cpu": n_cpus} if n_cpus else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)
