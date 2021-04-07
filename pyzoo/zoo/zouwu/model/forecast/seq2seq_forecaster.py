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

from zoo.zouwu.model.Seq2Seq_pytorch import Seq2SeqPytorch
from zoo.zouwu.model.forecast.abstract import Forecaster
from zoo.automl.common.util import load_config

import os


class Seq2SeqForecaster(Forecaster):

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
                 ):
        self.check_optional_config = False
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

    def fit(self, x, y, validation_data=None, epochs=1, metric="mse", batch_size=32):
        if validation_data is None:
            validation_data = (x, y)
        self.model_config["batch_size"] = batch_size
        self._check_data(x, y)
        return self.internal.fit_eval(x,
                                      y,
                                      validation_data=validation_data,
                                      epochs=epochs,
                                      metric=metric,
                                      **self.model_config)
    
    def predict(self, x):
        """
        Predict using a trained forecaster.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict(x)

    def predict_with_onnx(self, x, dirname=None):
        """
        Predict using a trained forecaster with onnxruntime.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict_with_onnx(x, dirname=dirname)

    def evaluate(self, x, y, metrics=['mse']):
        """
        Evaluate using a trained forecaster.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
        :param metrics: A list contains metrics for test/valid data.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate(x, y, metrics=metrics)
    
    def evaluate_with_onnx(self, x, y, metrics=['mse'], dirname=None):
        """
        Evaluate using a trained forecaster with onnxruntime.

        :param x: A numpy array with shape (num_samples, lookback, feature_dim).
        :param y: A numpy array with shape (num_samples, horizon, target_dim).
        :param metrics: A list contains metrics for test/valid data.
        :param dirname: The directory to save onnx model file. This value defaults
               to None for no saving file.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate_with_onnx(x, y, metrics=metrics, dirname=dirname)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if not self.internal.model_built:
            raise RuntimeError("You must call fit or restore first before calling save!")
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.internal.restore(checkpoint_file)