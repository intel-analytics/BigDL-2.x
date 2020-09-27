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

from zoo.automl.model.tcmf import DeepGLO
from zoo.automl.common.metrics import Evaluator
from zoo.automl.model.abstract import BaseModel
from zoo.orca.data import SparkXShards, XShards
import pickle
import numpy as np


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
        self.y_iters = config.get("y_iters", 10)
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

    def fit_eval(self, x, y=None, verbose=0, num_workers=None, **config):
        """
        Fit on the training data from scratch.
        Since the rolling process is very customized in this model,
        we enclose the rolling process inside this method.

        :param x: training data, an array in shape (nd, Td),
            nd is the number of series, Td is the time dimension
        :param y: None. target is extracted from x directly
        :param verbose:
        :param num_workers: number of workers to use.
        :return: the evaluation metric value
        """
        if not self.model_init:
            self._build(**config)
        if num_workers is None:
            num_workers = TCMF.get_default_num_workers()
        val_loss = self.model.train_all_models(x,
                                               alt_iters=self.alt_iters,
                                               y_iters=self.y_iters,
                                               init_epochs=self.init_epoch,
                                               max_FX_epoch=self.max_FX_epoch,
                                               max_TCN_epoch=self.max_TCN_epoch,
                                               num_workers=num_workers,
                                               )
        return val_loss

    def fit_incremental(self, x):
        """
        Incremental fitting given a pre-trained model.
        :param x: incremental data
        :param config: fitting parameters
        :return:
        """
        # TODO incrementally train models
        pass

    @staticmethod
    def get_default_num_workers():
        from zoo.ray import RayContext
        try:
            ray_ctx = RayContext.get(initialize=False)
            num_workers = ray_ctx.num_ray_nodes
        except:
            num_workers = 1
        return num_workers

    def predict(self, x=None, horizon=24, mc=False, num_workers=None):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: We don't support input x currently.
        :param horizon: horizon length to predict
        :param mc:
        :param num_workers: the number of workers to use. Note that there has to be an activate
            RayContext if num_workers > 1.
        :return:
        """
        if x is not None:
            raise ValueError("We don't support input x directly.")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        if num_workers is None:
            num_workers = TCMF.get_default_num_workers()
        if num_workers > 1:
            import ray
            from zoo.ray import RayContext
            try:
                RayContext.get(initialize=False)
            except:
                try:
                    # detect whether ray has been started.
                    ray.put(None)
                except:
                    raise RuntimeError(f"There must be an activate ray context while running with "
                                       f"{num_workers} workers. You can either start and init a "
                                       f"RayContext by init_orca_context(..., init_ray_on_spark="
                                       f"True) or start Ray with ray.init()")

        out = self.model.predict_horizon(
            future=horizon,
            bsize=90,
            normalize=False,
            num_workers=num_workers,
        )
        return out[:, -horizon::]

    def evaluate(self, x=None, y=None, metrics=None, num_workers=None):
        """
        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        y.
        :param x: We don't support input x currently.
        :param y: target. We interpret the second dimension of y as the horizon length for
            evaluation.
        :param metrics: a list of metrics in string format
        :param num_workers: the number of workers to use in evaluate. It defaults to 1.
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
        result = self.predict(x=None, horizon=horizon, num_workers=num_workers)

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


class ModelWrapper(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def is_distributed(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass


class TCMFDistributedModelWrapper(ModelWrapper):

    def __init__(self, config):
        self.internal = None
        self.config = config

    def fit(self, x, incremental=False, num_workers=None):
        if num_workers:
            raise ValueError("We don't support passing num_workers in fit "
                             "with input of xShards of dict")

        def orca_train_model(d, config):
            tcmf = TCMF()
            tcmf._build(**config)
            id_arr, train_data = split_id_and_train_data(d, True)
            if incremental:
                tcmf.fit_incremental(train_data)
            else:
                tcmf.fit_eval(train_data)
            return [id_arr, tcmf]

        if isinstance(x, SparkXShards):
            if x._get_class_name() == "dict":
                self.internal = x.transform_shard(orca_train_model, self.config)
            else:
                raise ValueError("value of x should be an xShards of dict, "
                                 "but is an xShards of " + x._get_class_name())
        else:
            raise ValueError("value of x should be an xShards of dict, "
                             "but isn't an xShards")

    def evaluate(self, x, y, metric=None, num_workers=None):
        """
        Evaluate the model
        :param x: input
        :param y: target
        :param metric:
        :param num_workers:
        :return: a list of metric evaluation results
        """
        raise NotImplementedError

    def predict(self, x, horizon=24, num_workers=None):
        """
        Prediction.
        :param x: input
        :param horizon:
        :param num_workers
        :return: result
        """
        if num_workers and num_workers != 1:
            raise ValueError("We don't support passing num_workers in predict "
                             "with input of xShards of dict")

        def orca_predict(data):
            id_arr = data[0]
            tcmf = data[1]
            predict_results = tcmf.predict(x=None, horizon=horizon)
            result = dict()
            result['id'] = id_arr
            result["prediction"] = predict_results
            return result

        return self.internal.transform_shard(orca_predict)

    def is_distributed(self):
        return True

    def save(self, model_path):
        """
        save model to file.
        :param model_path: the model file path to be saved to.
        :return:
        """
        if self.internal is not None:
            self.internal.save_pickle(model_path)

    def load(self, model_path, minPartitions=None):
        """
        restore model from model file and config.
        :param model_path: the model file
        :return: the restored model
        """
        self.internal = XShards.load_pickle(model_path, minPartitions=minPartitions)


class TCMFLocalModelWrapper(ModelWrapper):

    def __init__(self, config):
        self.internal = TCMF()
        self.config = config
        self.internal._build(**self.config)
        self.id_arr = None

    def fit(self, x, incremental=False, num_workers=None):
        if isinstance(x, dict):
            self.id_arr, train_data = split_id_and_train_data(x, False)
            if incremental:
                self.internal.fit_incremental(train_data)
            else:
                self.internal.fit_eval(train_data, num_workers=num_workers)
        else:
            raise ValueError("value of x should be a dict of ndarray")

    def evaluate(self, x, y, metric=None, num_workers=None):
        """
        Evaluate the model
        :param x: input
        :param y: target
        :param metric:
        :param num_workers:
        :return: a list of metric evaluation results
        """
        if isinstance(y, dict):
            if 'y' in y:
                y = y['y']
                if not isinstance(y, np.ndarray):
                    raise ValueError("the value of y should be an ndarray")
            else:
                raise ValueError("key y doesn't exist in y")
            return self.internal.evaluate(y=y, x=x, metrics=metric, num_workers=num_workers)
        else:
            raise ValueError("value of y should be a dict of ndarray")

    def predict(self, x, horizon=24, num_workers=None):
        """
        Prediction.
        :param x: input
        :param horizon
        :param num_workers
        :return: result
        """
        pred = self.internal.predict(x=x, horizon=horizon, num_workers=num_workers)
        result = dict()
        if self.id_arr is not None:
            result['id'] = self.id_arr
        result["prediction"] = pred
        return result

    def is_distributed(self):
        return False

    def save(self, model_path):
        """
        save model to file.
        :param model_path: the model file path to be saved to.
        :return:
        """
        with open(model_path + '/id.pkl', 'wb') as f:
            pickle.dump(self.id_arr, f)
        self.internal.save(model_path + "/model")

    def load(self, model_path):
        """
        restore model from model file and config.
        :param model_path: the model file
        :return: the restored model
        """
        self.internal = TCMF()
        with open(model_path + '/id.pkl', 'rb') as f:
            self.id_arr = pickle.load(f)
        self.internal.restore(model_path + "/model")


def split_id_and_train_data(d, is_distributed):
    if 'y' in d:
        train_data = d['y']
        if not isinstance(train_data, np.ndarray):
            raise ValueError("the value of y should be an ndarray")
    else:
        raise ValueError("key `y` doesn't exist in x")
    id_arr = None
    if 'id' in d:
        id_arr = d['id']
        if len(id_arr) != train_data.shape[0]:
            raise ValueError("the length of the id array should be equal to the number of "
                             "rows in the y")
    elif is_distributed:
        raise ValueError("key `id` doesn't exist in x")
    return id_arr, train_data
