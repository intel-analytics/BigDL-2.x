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
from zoo.orca.automl.xgboost.XGBoost import XGBoostModelBuilder
from zoo.orca.automl.auto_estimator import AutoEstimator


class AutoXGBClassifier(AutoEstimator):
    def __init__(self,
                 logs_dir="/tmp/auto_xgb_classifier_logs",
                 cpus_per_trial=1,
                 name=None,
                 **xgb_configs
                 ):
        """
        Automated xgboost classifier

        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_xgb_classifier_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
            The value will also be assigned to n_jobs in xgboost,
            which is the number of parallel threads used to run xgboost.
        :param name: Name of the auto xgboost classifier.
        :param xgb_configs: Other scikit learn xgboost parameters. You may refer to
           https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
           for the parameter names to specify. Note that we will directly use cpus_per_trial value
           for n_jobs in xgboost and you shouldn't specify n_jobs again.
        """
        xgb_model_builder = XGBoostModelBuilder(model_type='classifier',
                                                cpus_per_trial=cpus_per_trial,
                                                **xgb_configs)
        resources_per_trial = {"cpu": cpus_per_trial} if cpus_per_trial else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)


class AutoXGBRegressor(AutoEstimator):
    def __init__(self,
                 logs_dir="~/auto_xgb_regressor_logs",
                 cpus_per_trial=1,
                 name=None,
                 **xgb_configs
                 ):
        """
        Automated xgboost regressor

        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_xgb_classifier_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. The value will also be assigned
            to n_jobs, which is the number of parallel threads used to run xgboost.
        :param name: Name of the auto xgboost classifier.
        :param xgb_configs: Other scikit learn xgboost parameters. You may refer to
           https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
           for the parameter names to specify. Note that we will directly use cpus_per_trial value
           for n_jobs in xgboost and you shouldn't specify n_jobs again.
        """
        xgb_model_builder = XGBoostModelBuilder(model_type='regressor',
                                                cpus_per_trial=cpus_per_trial,
                                                **xgb_configs)
        resources_per_trial = {"cpu": cpus_per_trial} if cpus_per_trial else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)
