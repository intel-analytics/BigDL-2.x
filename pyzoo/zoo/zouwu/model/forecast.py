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
from zoo.automl.model import TCMF

import tensorflow as tf


class Forecaster(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass


class TCMFForecaster(Forecaster):
    def __init__(self,
                 vbsize=128,
                 hbsize=256,
                 num_channels_X=[32, 32, 32, 32, 32, 1],
                 num_channels_Y=[16, 16, 16, 16, 16, 1],
                 kernel_size=7,
                 dropout=0.1,
                 rank=64,
                 kernel_size_Y=7,
                 learning_rate=0.0005,
                 val_len=24,
                 end_index=-24,
                 normalize=False,
                 start_date="2020-4-1",
                 freq="1H",
                 covariates=None,
                 use_time=True,
                 dti=None,
                 svd=None,
                 period=24,
                 forward_cov=None,
                 max_y_iterations=300,
                 init_XF_epoch=100,
                 max_FX_epoch=300,
                 max_TCN_epoch=300):
        """
        Initialize
        :param vbsize:
        :param hbsize:
        :param num_channels_X:
        :param num_channels_Y:
        :param kernel_size:
        :param dropout:
        :param rank:
        :param kernel_size_Y:
        :param learning_rate:
        :param val_len:
        :param end_index:
        :param normalize:
        :param start_date:
        :param freq:
        :param use_time:
        :param dti:
        :param svd:
        :param period:
        :param forward_cov:
        :param max_y_iterations,
        :param init_XF_epoch,
        :param max_FX_epoch,
        :param max_TCN_epoch
        """
        self.internal = None
        self.config = {
            "vbsize" : vbsize ,
            "hbsize" : hbsize,
            "num_channels_X" : num_channels_X,
            "num_channels_Y" : num_channels_Y,
            "kernel_size" : kernel_size,
            "dropout" : dropout,
            "rank" : rank,
            "kernel_size_Y" : kernel_size_Y,
            "learning_rate" : learning_rate,
            "val_len" : val_len,
            "end_index" : end_index,
            "normalize" : normalize,
            "start_date" : start_date,
            "freq" : freq,
            "covariates" : covariates,
            "use_time" : use_time,
            "dti" : dti,
            "svd" : svd,
            "period" : period,
            "forward_cov" : forward_cov,
            "max_y_iterations" : max_y_iterations,
            "init_XF_epoch" : init_XF_epoch,
            "max_FX_epoch" : max_FX_epoch,
            "max_TCN_epoch" : max_TCN_epoch
        }
        self.model = self._build()

    def _build(self):
        self.internal = TCMF()
        return self.internal._build(**self.config)

    def fit(self,
            x,
            incremental=False):
        """
        fit the model
        :param x: the input
        :param covariates: the global covariates
        :param lr: learning rate
        :param incremental: if the fit is incremental
        :return:
        """
        if incremental:
            self.internal.fit_incremental(x)
        else:
            self.internal.fit_eval(x)

    def evaluate(self,
                 x,
                 metric=['mae'],
                 covariates=None,
                 ):
        """
        evaluate the model
        :param covariates: global covariates
        :param x: the input
        :param metric: the metrics
        :return:
        """
        self.internal.evaluate(x, metric=metric)

    def predict(self,
                x,
                covariates=None,
                ):
        """
        predict
        :param x: the input
        :param covariates: the global covariates
        :return:
        """
        self.internal.predict(x)


class TFParkForecaster(TFParkKerasModel, Forecaster, metaclass=ABCMeta):
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


class MTNetForecaster(TFParkForecaster):
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
