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
from abc import ABC

from zoo.automl.model.tcmf.DeepGLO import *

from zoo.automl.common.metrics import Evaluator
import pandas as pd
from zoo.automl.model.abstract import BaseModel
from zoo.automl.common.util import save_config


class TCMF(BaseModel):
    """
    MF regularized TCN + TCN. This version is not for automated searching yet.
    """

    def __init__(self):
        """
        Initialize hyper parameters
        :param check_optional_config:
        :param future_seq_len:
        """
        # models
        self.model_init = False

    def set_params(self, **config):
        self.vbsize = config.get("vbsize", 128)
        self.hbsize = config.get("hbsize", 256)
        self.num_channels_X = config.get("num_channels_X", [32, 32, 32, 32, 32, 1])
        self.num_channels_Y = config.get("num_channels_Y", [16, 16, 16, 16, 16, 1])
        self.kernel_size = config.get("kernel_size", 7)
        self.dropout = config.get("dropout", 0.1)
        self.rank = config.get("rank", 64)
        self.kernel_size_Y = config.get("kernel_size_Y", 7)
        self.lr = config.get("learning_rate", 0.0005)
        self.val_len = config.get("val_len", 24)
        self.end_index = config.get("end_index", -24)
        self.normalize = config.get("normalize", False)
        self.start_date = config.get("start_date", "2020-4-1")
        self.freq = config.get("freq", "1H")
        self.covariates = config.get('covariates', None)
        self.use_time = config.get("use_time", True)
        self.dti = config.get("dti", None)
        self.svd = config.get("svd", None)
        self.period = config.get("period", 24)
        self.forward_cov = config.get("forward_cov", None)
        self.alt_iters = config.get("alt_iters", 10)
        self.y_iters = config.get("max_y_iterations", 300)
        self.init_epoch = config.get("init_XF_epoch", 100)
        self.max_FX_epoch = config.get("max_FX_epoch", 300)
        self.max_TCN_epoch = config.get("max_TCN_epoch", 300)

    def _build(self, **config):
        """
        build the models and initialize.
        :param config: hyper parameters for building the model
        :return:
        """
        self.set_params(**config)
        self.model = DeepGLO(
            vbsize=self.vbsize,
            hbsize=self.hbsize,
            num_channels_X=self.num_channels_X,
            num_channels_Y=self.num_channels_Y,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            rank=self.rank,
            kernel_size_Y=self.kernel_size_Y,
            lr=self.lr,
            val_len=self.val_len,
            end_index=self.end_index,
            normalize=self.normalize,
            start_date=self.start_date,
            freq=self.freq,
            covariates=self.covariates,
            use_time=self.use_time,
            dti=self.dti,
            svd=self.svd,
            period=self.period,
            forward_cov=self.forward_cov
        )
        self.model_init = True

    def fit_eval(self, x, y=None, verbose=0, **config):
        """
        Fit on the training data from scratch.
        Since the rolling process is very customized in this model,
        we enclose the rolling process inside this method.

        :param x: training data, an array in shape (nd, Td),
            nd is the number of series, Td is the time dimension
        :param y: None. target is extracted from x directly
        :param verbose:
        :return: the evaluation metric value
        """
        if not self.model_init:
            self._build(**config)
        self.model.train_all_models(x,
                                    alt_iters=self.alt_iters,
                                    y_iters=self.y_iters,
                                    init_epochs=self.init_epoch,
                                    max_FX_epoch=self.max_FX_epoch,
                                    max_TCN_epoch=self.max_TCN_epoch)
        return self.model.Yseq.val_loss

    def fit_incremental(self, x):
        """
        Incremental fitting given a pre-trained model.
        :param x: incremental data
        :param config: fitting parameters
        :return:
        """
        # TODO incrementally train models
        pass

    def predict(self, x, mc=False, ):
        """

        :param x:
        :param mc:
        :return:
        """
        # TODO predict on new data
        pass

    def evaluate(self, x, y, metric=None):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metric: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        pass

    def save(self, model_path, config_path):
        raise NotImplementedError

    def restore(self, model_path, **config):
        raise NotImplementedError

    def _get_optional_parameters(self):
        return {}

    def _get_required_parameters(self):
        return {}
