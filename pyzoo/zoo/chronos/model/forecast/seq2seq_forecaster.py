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

from zoo.chronos.model.forecast.abstract import Forecaster
from zoo.chronos.model.forecast.utils import np_to_creator
from zoo.chronos.model.Seq2Seq_pytorch import Seq2SeqPytorch
from zoo.chronos.model.Seq2Seq_pytorch import model_creator, optimizer_creator, loss_creator
from zoo.orca.data import XShards
from zoo.orca.learn.pytorch.estimator import Estimator
from zoo.orca.learn.metrics import MSE, MAE

import torch
import numpy as np
import os

ORCA_METRICS = {"mse": MSE, "mae": MAE}


class Seq2SeqForecaster(Forecaster):
    """
        Example:
            >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
            >>> forecaster = Seq2SeqForecaster(future_seq_len=5,
                                               input_feature_num=3,
                                               output_feature_num=2,
                                               lstm_layer_num=2,
                                               ...)
            >>> forecaster.fit(x_train, y_train)
            >>> forecaster.to_local()  # if you set distributed=True
            >>> test_pred = forecaster.predict(x_test)
            >>> test_eval = forecaster.evaluate(x_test, y_test)
            >>> forecaster.save({ckpt_name})
            >>> forecaster.restore({ckpt_name})
    """
    def __init__(self,
                 input_feature_num,
                 future_seq_len,
                 output_feature_num,
                 lstm_hidden_dim=128,
                 lstm_layer_num=1,
                 teacher_forcing=False,
                 dropout=0.25,
                 lr=0.001,
                 loss="mse",
                 optimizer="Adam",
                 metrics=["mse"],
                 distributed=False,
                 workers_per_node=1,
                 distributed_backend="torch_distributed"):
        """
        Build a LSTM Sequence to Sequence Forecast Model.

        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param input_feature_num: Specify the feature dimension.
        :param output_feature_num: Specify the output dimension.
        :param lstm_hidden_dim: LSTM hidden channel for decoder and encoder.
        :param lstm_layer_num: LSTM layer number for decoder and encoder.
        :param teacher_forcing: If use teacher forcing in training.
        :param dropout: Specify the dropout close possibility (i.e. the close
               possibility to a neuron). This value defaults to 0.25.
        :param optimizer: Specify the optimizer used for training. This value
               defaults to "Adam".
        :param loss: Specify the loss function used for training. This value
               defaults to "mse". You can choose from "mse", "mae" and
               "huber_loss".
        :param lr: Specify the learning rate. This value defaults to 0.001.
        :param metrics: A list contains metrics for evaluating the quality of
               forecasting. You may only choose from "mse" and "mae" for a
               distributed forecaster. You may choose from "mse", "me", "mae",
               "mse","rmse","msle","r2", "mpe", "mape", "mspe", "smape", "mdape"
               and "smdape" for a non-distributed forecaster.
        :param distributed: bool, if init the forecaster in a distributed
               fashion. If True, the internal model will use an Orca Estimator.
               If False, the internal model will use a pytorch model. The value
               defaults to False.
        :param workers_per_node: int, the number of worker you want to use.
               The value defaults to 1. The param is only effective when
               distributed is set to True.
        :param distributed_backend: str, select from "torch_distributed" or
               "horovod". The value defaults to "torch_distributed".
        """
        # config setting
        self.model_config = {
            "input_feature_num": input_feature_num,
            "future_seq_len": future_seq_len,
            "output_feature_num": output_feature_num,
            "lstm_hidden_dim": lstm_hidden_dim,
            "lstm_layer_num": lstm_layer_num,
            "teacher_forcing": teacher_forcing,
            "dropout": dropout,
            "lr": lr,
            "loss": loss,
            "optimizer": optimizer,
        }
        self.metrics = metrics

        # create internal implementation
        self.internal = None
        self.distributed = distributed
        if self.distributed:
            def model_creator_seq2seq(config):
                model = model_creator(self.model_config)
                model.train()
                return model

            self.internal = Estimator.from_torch(model=model_creator_seq2seq,
                                                 optimizer=optimizer_creator,
                                                 loss=loss_creator,
                                                 metrics=[ORCA_METRICS[name]()
                                                          for name in self.metrics],
                                                 backend=distributed_backend,
                                                 use_tqdm=True,
                                                 config={"lr": lr},
                                                 workers_per_node=workers_per_node)
        else:
            self.internal = Seq2SeqPytorch(check_optional_config=False)

    def _check_data(self, x, y):
        assert self.model_config["future_seq_len"] == y.shape[-2], \
            "The y shape should be (batch_size, future_seq_len, target_col_num), \
            Got future_seq_len of {} in config while y input shape of {}."\
            .format(self.model_config["future_seq_len"], y.shape[-2])
        assert self.model_config["input_feature_num"] == x.shape[-1],\
            "The x shape should be (batch_size, past_seq_len, input_feature_num), \
            Got input_feature_num of {} in config while x input shape of {}."\
            .format(self.model_config["input_feature_num"], x.shape[-1])
        assert self.model_config["output_feature_num"] == y.shape[-1], \
            "The y shape should be (batch_size, future_seq_len, output_feature_num), \
            Got output_feature_num of {} in config while y input shape of {}."\
            .format(self.model_config["output_feature_num"], y.shape[-1])

    def fit(self, x, y, epochs=1, batch_size=32):
        """
        Fit(Train) the forecaster.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
               lookback and feature_dim should be the same as past_seq_len and input_feature_num.
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
               horizon and target_dim should be the same as future_seq_len and output_feature_num.
        :param epochs: Number of epochs you want to train. The value defaults to 1.
        :param batch_size: Number of batch size you want to train. The value defaults to 32.

        :return: Evaluation results on validation data.
        """
        # input check
        validation_data = (x, y)
        self.model_config["batch_size"] = batch_size
        self._check_data(x, y)
        # fit on internal
        if self.distributed:
            return self.internal.fit(data=np_to_creator((x, y)),
                                     epochs=epochs,
                                     batch_size=batch_size)
        else:
            return self.internal.fit_eval(data=(x, y),
                                          validation_data=validation_data,
                                          epochs=epochs,
                                          metric=self.metrics[0],  # only use the first metric
                                          **self.model_config)

    def predict(self, x, batch_size=32):
        """
        Predict using a trained forecaster.

        if you want to predict on a single node(which is common practice), please call
        .to_local().predict(x, ...)

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).

        :return: A numpy array with shape (num_samples, lookback, feature_dim).
        """
        if self.distributed:
            # map input to a xshard
            x = XShards.partition(x)

            def transform_to_dict(train_data):
                return {"x": train_data}

            x = x.transform_shard(transform_to_dict)
            # predict with distributed fashion
            yhat = self.internal.predict(x, batch_size=batch_size)
            # collect result from xshard to numpy
            yhat = yhat.collect()
            yhat = np.concatenate([yhat[i]['prediction'] for i in range(len(yhat))], axis=0)
            if yhat.ndim == 2:
                yhat = np.expand_dims(yhat, axis=2)
            return yhat
        else:
            if not self.internal.model_built:
                raise RuntimeError("You must call fit or restore first before calling predict!")
            return self.internal.predict(x, batch_size=batch_size)

    def predict_with_onnx(self, x,  batch_size=32, dirname=None):
        """
        Predict using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.

        :return: A numpy array with shape (num_samples, lookback, feature_dim).
        """
        if self.distributed:
            raise NotImplementedError("ONNX inference has not been supported for distributed\
                                       forecaster. You can call .to_local() to transform the\
                                       forecaster to a non-distributed version.")
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict_with_onnx(x, batch_size=batch_size, dirname=dirname)

    def evaluate(self, x, y, batch_size=32, multioutput="raw_values"):
        """
        Evaluate using a trained forecaster.
        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.
        if you want to evaluate on a single node(which is common practice), please call
        .to_local().evaluate(x, y, ...)
        >>> from zoo.automl.common.metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)
        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.The param is only effective when the forecaster is a
               non-distribtued version.
        :return: A list of evaluation results. Each item represents a metric.
        """
        if self.distributed:
            return self.internal.evaluate(data=np_to_creator((x, y)),
                                          batch_size=batch_size)
        else:
            if not self.internal.model_built:
                raise RuntimeError("You must call fit or restore first before calling evaluate!")
            return self.internal.evaluate(x, y, metrics=self.metrics,
                                          multioutput=multioutput, batch_size=batch_size)

    def evaluate_with_onnx(self, x, y,
                           batch_size=32,
                           dirname=None,
                           multioutput="raw_values"):
        """
        Evaluate using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.
        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.
        >>> from zoo.automl.common.metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)
        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.
        :return: A list of evaluation results. Each item represents a metric.
        """
        if self.distributed:
            raise NotImplementedError("ONNX inference has not been supported for distributed\
                                       forecaster. You can call .to_local() to transform the\
                                       forecaster to a non-distributed version.")
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate_with_onnx(x, y,
                                                metrics=self.metrics,
                                                dirname=dirname,
                                                multioutput=multioutput,
                                                batch_size=batch_size)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if self.distributed:
            self.internal.save(checkpoint_file)
        else:
            if not self.internal.model_built:
                raise RuntimeError("You must call fit or restore first before calling save!")
            self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        if self.distributed:
            self.internal.load(checkpoint_file)
        else:
            self.internal.restore(checkpoint_file)

    def to_local(self):
        """
        Transform a distributed forecaster to a local (non-distributed) one.
        Common practice is to use distributed training (fit) and predict/
        evaluate with onnx or other frameworks on a single node. To do so,
        you need to call .to_local() and transform the forecaster to a non-
        distributed one.
        The optimizer is refreshed, incremental training after to_local
        might have some problem.
        :return: a forecaster instance.
        """
        # TODO: optimizer is refreshed, which is not reasonable
        if not self.distributed:
            raise RuntimeError("The forecaster has become local.")
        model = self.internal.get_model()
        state = {
            "config": self.model_config,
            "model": model.state_dict(),
            "optimizer": optimizer_creator(model, {"lr": self.model_config["lr"]}).state_dict(),
        }
        self.shutdown()
        self.internal = Seq2SeqPytorch(check_optional_config=False)
        self.internal.load_state_dict(state)
        self.distributed = False
        return self

    def shutdown(self, force=False):
        """
        Only used when you what to shut down a distributed forecaster's
        workers and releases resources.
        :param force: bool, if force to shut down the resources.
        """
        if not self.distributed:
            raise RuntimeError("A local forecaster does not need shutdown.")
        self.internal.shutdown(force)
