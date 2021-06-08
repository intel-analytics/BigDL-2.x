# +
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

from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.chronos.model.arima import ARIMABuilder


# -

class AutoARIMA(AutoEstimator):

    def __init__(self,
                 logs_dir="/tmp/auto_arima_logs",
                 **arima_config
                 ):
        """
        Automated ARIMA Model
        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_arima_logs"
        :param arima_config: Other ARIMA hyperparameters. You may refer to
           https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        for the parameter names to specify.
        """
        arima_model_builder = ARIMABuilder(**arima_config)
        super().__init__(model_builder=arima_model_builder,
                         logs_dir=logs_dir)
