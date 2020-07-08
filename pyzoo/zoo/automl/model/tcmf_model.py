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

from zoo.automl.model.tcmf import DeepGLO
from zoo.automl.common.metrics import Evaluator
from zoo.automl.model.abstract import BaseModel
import pickle


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
        self.model = None
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
        self.normalize = config.get("normalize", False)
        self.start_date = config.get("start_date", "2020-4-1")
        self.freq = config.get("freq", "1H")
        self.covariates = config.get('covariates', None)
        self.use_time = config.get("use_time", True)
        self.dti = config.get("dti", None)
        self.svd = config.get("svd", True)
        self.period = config.get("period", 24)
        self.alt_iters = config.get("alt_iters", 10)
        self.y_iters = config.get("max_y_iterations", 300)
        self.init_epoch = config.get("init_FX_epoch", 100)
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
            normalize=self.normalize,
            start_date=self.start_date,
            freq=self.freq,
            covariates=self.covariates,
            use_time=self.use_time,
            dti=self.dti,
            svd=self.svd,
            period=self.period,
            forward_cov=False
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

    def predict(self, x=None, horizon=24, mc=False, ):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: We don't support input x currently.
        :param horizon: horizon length to predict
        :param mc:
        :return:
        """
        if x is not None:
            raise ValueError("We don't support input x directly.")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        out = self.model.predict_horizon(
            future=horizon,
            bsize=90,
            normalize=False,
        )
        return out[:, -horizon::]

    def evaluate(self, x=None, y=None, metrics=None):
        """
        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        y.
        :param x: We don't support input x currently.
        :param y: target. We interpret the second dimension of y as the horizon length for
            evaluation.
        :param metrics: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        if x is not None:
            raise ValueError("We don't support input x directly.")
        if y is None:
            raise ValueError("Input invalid y of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
            horizon = 1
        else:
            horizon = y.shape[1]
        result = self.predict(horizon=horizon)

        if y.shape[1] == 1:
            multioutput = 'uniform_average'
        else:
            multioutput = 'raw_values'
        return [Evaluator.evaluate(m, y, result, multioutput=multioutput) for m in metrics]

    def save(self, model_file):
        pickle.dump(self.model, open(model_file, "wb"))

    def restore(self, model_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        self.model_init = True

    def _get_optional_parameters(self):
        return {}

    def _get_required_parameters(self):
        return {}
