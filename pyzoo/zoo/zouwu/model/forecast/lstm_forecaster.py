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

from zoo.automl.model.VanillaLSTM import VanillaLSTM as LSTMKerasModel
from zoo.zouwu.model.forecast.tfpark_forecaster import TFParkForecaster


class LSTMForecaster(TFParkForecaster):
    """
    Vanilla LSTM Forecaster
    """

    def __init__(self,
                 target_dim=1,
                 feature_dim=1,
                 lstm_1_units=16,
                 dropout_1=0.2,
                 lstm_2_units=8,
                 dropout_2=0.2,
                 metric="mean_squared_error",
                 lr=0.001,
                 loss="mse",
                 uncertainty: bool = False
                 ):
        """
        Build a LSTM Forecast Model.

        :param target_dim: dimension of model output
        :param feature_dim: dimension of input feature
        :param lstm_1_units: num of units for the 1st LSTM layer
        :param dropout_1: p for the 1st dropout layer
        :param lstm_2_units: num of units for the 2nd LSTM layer
        :param dropout_2: p for the 2nd dropout layer
        :param metric: the metric for validation and evaluation
        :param lr: learning rate
        :param uncertainty: whether to return uncertainty
        :param loss: the target function you want to optimize on
        """
        #
        self.target_dim = target_dim
        self.check_optional_config = False
        self.uncertainty = uncertainty

        self.model_config = {
            "lr": lr,
            "lstm_1_units": lstm_1_units,
            "dropout_1": dropout_1,
            "lstm_2_units": lstm_2_units,
            "dropout_2": dropout_2,
            "metric": metric,
            "feature_num": feature_dim,
            "loss": loss
        }
        self.internal = None

        super().__init__()

    def _build(self):
        """
        Build LSTM Model in tf.keras
        """
        # build model with TF/Keras
        self.internal = LSTMKerasModel(
            check_optional_config=self.check_optional_config,
            future_seq_len=self.target_dim)
        return self.internal._build(mc=self.uncertainty,
                                    **self.model_config)
