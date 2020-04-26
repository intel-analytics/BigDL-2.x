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

from abc import ABCMeta, abstractmethod

from zoo.automl.model.MTNet_keras import MTNetKeras as MTNetKerasModel
from zoo.automl.model.VanillaLSTM import VanillaLSTM as LSTMKerasModel
from zoo.tfpark import KerasModel as TFParkKerasModel

import tensorflow as tf


class Forecaster(TFParkKerasModel, metaclass=ABCMeta):
    """
    Base class for TFPark KerasModel based Forecast models.
    """

    def __init__(self):
        """
        Initializer.
        Turns the tf.keras model returned from _build into a tfpark.KerasModel
        """
        self.model = self._build()
        assert (isinstance(self.model, tf.keras.Model))
        super().__init__(self.model)

    @abstractmethod
    def _build(self):
        """
        Build a tf.keras model.
        :return: a tf.keras model (compiled)
        """
        pass


class LSTMForecaster(Forecaster):
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
            "feature_num": feature_dim
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


class MTNetForecaster(Forecaster):
    """
    MTNet Forecast Model
    """

    def __init__(self,
                 target_dim=1,
                 feature_dim=1,
                 lb_long_steps=1,
                 lb_long_stepsize=1,
                 ar_window_size=1,
                 cnn_kernel_size=1,
                 metric="mean_squared_error",
                 uncertainty: bool = False,
                 ):
        """
        Build a MTNet Forecast Model.
        :param target_dim: the dimension of model output
        :param feature_dim: the dimension of input feature
        :param lb_long_steps: the number of steps for the long-term memory series
        :param lb_long_stepsize: the step size for long-term memory series
        :param ar_window_size：the auto regression window size in MTNet
        :param cnn_kernel_size: cnn filter height in MTNet
        :param metric: the metric for validation and evaluation
        :param uncertainty: whether to enable calculation of uncertainty
        """
        self.check_optional_config = False
        self.mc = uncertainty
        self.model_config = {
            "feature_num": feature_dim,
            "output_dim": target_dim,
            "metrics": [metric],
            "mc": uncertainty,
            "time_step": lb_long_stepsize,
            "long_num": lb_long_steps,
            "ar_window": ar_window_size,
            "cnn_height": cnn_kernel_size,
            "past_seq_len": (lb_long_steps + 1) * lb_long_stepsize

        }
        self._internal = None

        super().__init__()

    def _build(self):
        """
        build a MTNet model in tf.keras
        :return: a tf.keras MTNet model
        """
        # TODO change this function call after MTNet fixes
        self.internal = MTNetKerasModel(
            check_optional_config=self.check_optional_config,
            future_seq_len=self.model_config.get('output_dim'))
        self.internal.apply_config(config=self.model_config)
        return self.internal.build()

    def preprocess_input(self, x):
        """
        The original rolled features needs an extra step to process.
        This should be called before train_x, validation_x, and test_x
        :param x: the original samples from rolling
        :return: a tuple (long_term_x, short_term_x)
                which are long term and short term history respectively
        """
        return self.internal._reshape_input_x(x)
