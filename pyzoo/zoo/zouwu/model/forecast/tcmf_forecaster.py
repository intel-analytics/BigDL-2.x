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

from zoo.automl.model.tcmf_model import TCMFNdarrayModelWrapper, TCMFXshardsModelWrapper
from zoo.orca.data import SparkXShards
from zoo.zouwu.model.forecast.abstract import Forecaster


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
                 y_iters=10,
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
        :param y_iters: int, default is 10.
            Number of iterations while training the hybrid model.
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
            "y_iters": y_iters,
            "init_FX_epoch": init_FX_epoch,
            "max_FX_epoch": max_FX_epoch,
            "max_TCN_epoch": max_TCN_epoch,
            "alt_iters": alt_iters,
        }

    def fit(self,
            x, num_workers=None):
        """
        fit the model on x from scratch
        :param x: the input for fit. Only dict of ndarray and SparkXShards of dict of ndarray
            are supported. Example: {'id': id_arr, 'y': data_ndarray}, and data_ndarray is of shape
            (n, T), where n is the number f target time series and T is the number of time steps.
        :param num_workers: the number of workers you want to use for fit. If None, it defaults to
        num_ray_nodes in the created RayContext or 1 if there is no active RayContext.
        :return: None
        """
        if self.internal is None:
            if isinstance(x, SparkXShards):
                self.internal = TCMFXshardsModelWrapper(self.config)
            elif isinstance(x, dict):
                self.internal = TCMFNdarrayModelWrapper(self.config)
            else:
                raise ValueError("value of x should be a dict of ndarray or "
                                 "an xShards of dict of ndarray")

            try:
                self.internal.fit(x, num_workers=num_workers)
            except Exception as inst:
                self.internal = None
                raise inst
        else:
            raise Exception("This model has already been fully trained, "
                            "you can only run full training once.")

    def fit_incremental(self, x_incr):
        """
        incrementally fit the model. Note that we only incrementally fit X_seq (TCN in global model)
        :param x_incr: incremental data to be fitted. It should be of the same format as input x in
         fit, which is a dict of ndarray or SparkXShards of dict of ndarray.
        Example: {'id': id_arr, 'y': incr_ndarray}, and incr_ndarray is of shape (n, T_incr), where
        n is the number of target time series, T_incr is the number of time steps incremented. You
        can choose not to input 'id' in x_incr, but if you do, the elements of id in x_incr should
        be the same as id in x of fit.
        :return: None.
        """
        self.internal.fit_incremental(x_incr)

    def evaluate(self,
                 target_value,
                 x=None,
                 metric=['mae'],
                 covariates=None,
                 num_workers=None,
                 ):
        """
        evaluate the model
        :param target_value: target value for evaluation. We interpret its second dimension of
        as the horizon length for evaluation.
        :param covariates: global covariates
        :param x: the input
        :param metric: the metrics
        :param num_workers: the number of workers to use in evaluate. If None, it defaults to
        num_ray_nodes in the created RayContext or 1 if there is no active RayContext.
        :return:
        """
        return self.internal.evaluate(y=target_value, x=x, metric=metric, num_workers=num_workers)

    def predict(self,
                x=None,
                horizon=24,
                num_workers=None,
                ):
        """
        predict
        :param x: the input. We don't support input x directly
        :param horizon: horizon length to look forward.
        :param num_workers: the number of workers to use in predict. If None, it defaults to
        num_ray_nodes in the created RayContext or 1 if there is no active RayContext.
        :return:
        """
        if self.internal is None:
            raise Exception("You should run fit before calling predict()")
        else:
            return self.internal.predict(x, horizon, num_workers=num_workers)

    def save(self, path):
        if self.internal is None:
            raise Exception("You should run fit before calling save()")
        else:
            self.internal.save(path)

    def is_xshards_distributed(self):
        if self.internal is None:
            raise ValueError("You should run fit before calling is_xshards_distributed()")
        else:
            return self.internal.is_xshards_distributed()

    @classmethod
    def load(cls, path, is_xshards_distributed=False, minPartitions=None):
        loaded_model = TCMFForecaster()
        if is_xshards_distributed:
            loaded_model.internal = TCMFXshardsModelWrapper(loaded_model.config)
            loaded_model.internal.load(path, minPartitions=minPartitions)
        else:
            loaded_model.internal = TCMFNdarrayModelWrapper(loaded_model.config)
            loaded_model.internal.load(path)
        return loaded_model
