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
                 normalize=False,
                 start_date="2020-4-1",
                 freq="1H",
                 covariates=None,
                 use_time=True,
                 dti=None,
                 svd=True,
                 period=24,
                 max_y_iterations=300,
                 init_FX_epoch=100,
                 max_FX_epoch=300,
                 max_TCN_epoch=300,
                 alt_iters=10):
        """
        Initialize
        :param vbsize: int, default is 128.
            Vertical batch size, which is the number of cells per batch.
        :param hbsize: int, default is 256.
            Horizontal batch size, which is the number of time series per batch.
        :param num_channels_X: list, default=[32, 32, 32, 32, 32, 1].
            List containing channel progression of temporal convolution network for local model
        :param num_channels_Y: list, default=[16, 16, 16, 16, 16, 1]
            List containing channel progression of temporal convolution network for hybrid model.
        :param kernel_size: int, default is 7.
            Kernel size for local models
        :param dropout: float, default is 0.1.
            Dropout rate during training
        :param rank: int, default is 64.
            The rank in matrix factorization of global model.
        :param kernel_size_Y: int, default is 7.
            Kernel size of hybrid model
        :param learning_rate: float, default is 0.0005
        :param val_len: int, default is 24.
            Validation length. We will use the last val_len time points as validation data.
        :param normalize: boolean, false by default.
            Whether to normalize input data for training.
        :param start_date: str or datetime-like.
            Start date time for the time-series. e.g. "2020-01-01"
        :param freq: str or DateOffset, default is 'H'
            Frequency of data
        :param use_time: boolean, default is True.
            Whether to use time coveriates.
        :param covariates: 2-D ndarray or None. The shape of ndarray should be (r, T), where r is
            the number of covariates and T is the number of time points.
            Global covariates for all time series. If None, only default time coveriates will be
            used while use_time is True. If not, the time coveriates used is the stack of input
            covariates and default time coveriates.
        :param dti: DatetimeIndex or None.
            If None, use default fixed frequency DatetimeIndex generated with start_date and freq.
        :param svd: boolean, default is False.
            Whether factor matrices are initialized by NMF
        :param period: int, default is 24.
            Periodicity of input time series, leave it out if not known
        :param max_y_iterations: int, default is 300.
            Max number of iterations while training the hybrid model.
        :param init_FX_epoch: int, default is 100.
            Number of iterations while initializing factors
        :param max_FX_epoch: int, default is 300.
            Max number of iterations while training factors.
        :param max_TCN_epoch: int, default is 300.
            Max number of iterations while training the local model.
        :param alt_iters: int, default is 10.
            Number of iterations while alternate training.
        """
        self.internal = None
        self.config = {
            "vbsize": vbsize,
            "hbsize": hbsize,
            "num_channels_X": num_channels_X,
            "num_channels_Y": num_channels_Y,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "rank": rank,
            "kernel_size_Y": kernel_size_Y,
            "learning_rate": learning_rate,
            "val_len": val_len,
            "normalize": normalize,
            "start_date": start_date,
            "freq": freq,
            "covariates": covariates,
            "use_time": use_time,
            "dti": dti,
            "svd": svd,
            "period": period,
            "max_y_iterations": max_y_iterations,
            "init_FX_epoch": init_FX_epoch,
            "max_FX_epoch": max_FX_epoch,
            "max_TCN_epoch": max_TCN_epoch,
            "alt_iters": alt_iters,
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
                 target_value,
                 x=None,
                 metric=['mae'],
                 covariates=None,
                 ):
        """
        evaluate the model
        :param target_value: target value for evaluation. We interpret its second dimension of
        as the horizon length for evaluation.
        :param covariates: global covariates
        :param x: the input
        :param metric: the metrics
        :return:
        """
        self.internal.evaluate(y=target_value, x=x, metrics=metric)

    def predict(self,
                x=None,
                horizon=24,
                covariates=None,
                ):
        """
        predict
        :param x: the input. We don't support input x directly
        :param horizon: horizon length to look forward.
        :param covariates: the global covariates
        :return:
        """
        return self.internal.predict(x=x, horizon=horizon)


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


class MTNetForecaster(TFParkForecaster):
    """
    MTNet Forecast Model
    """

    def __init__(self,
                 target_dim=1,
                 feature_dim=1,
                 long_series_num=1,
                 series_length=1,
                 ar_window_size=1,
                 cnn_height=1,
                 cnn_hid_size=32,
                 rnn_hid_sizes=[16, 32],
                 lr=0.001,
                 loss="mae",
                 cnn_dropout=0.2,
                 rnn_dropout=0.2,
                 metric="mean_squared_error",
                 uncertainty: bool = False,
                 ):
        """
        Build a MTNet Forecast Model.
        :param target_dim: the dimension of model output
        :param feature_dim: the dimension of input feature
        :param long_series_num: the number of series for the long-term memory series
        :param series_length: the series size for long-term and short-term memory series
        :param ar_window_size: the auto regression window size in MTNet
        :param cnn_hid_size: the hidden layer unit for cnn in encoder
        :param rnn_hid_sizes: the hidden layers unit for rnn in encoder
        :param cnn_height: cnn filter height in MTNet
        :param metric: the metric for validation and evaluation
        :param uncertainty: whether to enable calculation of uncertainty
        :param lr: learning rate
        :param loss: the target function you want to optimize on
        :param cnn_dropout: the dropout possibility for cnn in encoder
        :param rnn_dropout: the dropout possibility for rnn in encoder
        """
        self.check_optional_config = False
        self.mc = uncertainty
        self.model_config = {
            "feature_num": feature_dim,
            "output_dim": target_dim,
            "metrics": [metric],
            "mc": uncertainty,
            "time_step": series_length,
            "long_num": long_series_num,
            "ar_window": ar_window_size,
            "cnn_height": cnn_height,
            "past_seq_len": (long_series_num + 1) * series_length,
            "cnn_hid_size": cnn_hid_size,
            "rnn_hid_sizes": rnn_hid_sizes,
            "lr": lr,
            "cnn_dropout": cnn_dropout,
            "rnn_dropout": rnn_dropout,
            "loss": loss
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
