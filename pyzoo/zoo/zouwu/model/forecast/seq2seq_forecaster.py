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

from zoo.automl.model.Seq2Seq import LSTMSeq2Seq as Seq2SeqKerasModel
from zoo.zouwu.model.forecast.abstract import Forecaster
from zoo.automl.common.util import load_config

import os


class Seq2SeqForecaster(Forecaster):
    """
    Seq2Seq Forecaster
    """

    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 feature_num,
                 target_col_num,
                 latent_dim=128,
                 dropout=0.2,
                 lr=0.001,
                 loss='mse',
                 metric="mean_squared_error",
                 ):
        self.check_optional_config = False
        self.model_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": future_seq_len,
            "feature_num": feature_num,
            "target_col_num": target_col_num,
            "latent_dim": latent_dim,
            "dropout": dropout,
            "lr": lr,
            "loss": loss,
            "metric": metric
        }
        self.internal = Seq2SeqKerasModel(check_optional_config=False,
                                   future_seq_len=self.model_config["future_seq_len"])

    def _check_data(self, x, y):
        assert self.model_config["past_seq_len"] == x.shape[-2], \
            "The x shape should be (batch_size, past_seq_len, feature_num), \
            Got past_seq_len of {} in config while x input shape of {}."\
            .format(self.model_config["past_seq_len"], x.shape[-2])
        assert self.model_config["future_seq_len"] == y.shape[-2], \
            "The y shape should be (batch_size, future_seq_len, target_col_num), \
            Got future_seq_len of {} in config while y input shape of {}."\
            .format(self.model_config["future_seq_len"], y.shape[-2])
        assert self.model_config["feature_num"] == x.shape[-1],\
            "The x shape should be (batch_size, past_seq_len, feature_num), \
            Got feature_num of {} in config while x input shape of {}."\
            .format(self.model_config["input_feature_num"], x.shape[-1])
        assert self.model_config["target_col_num"] == y.shape[-1], \
            "The y shape should be (batch_size, future_seq_len, target_col_num), \
            Got target_col_num of {} in config while y input shape of {}."\
            .format(self.model_config["target_col_num"], y.shape[-1])

    def fit(self, x, y, epochs=1, batch_size=32):
        self.model_config["batch_size"] = batch_size
        self._check_data(x, y)
        return self.internal.fit_eval(x,
                                      y,
                                      validation_data=(x, y),
                                      epochs=epochs,
                                      **self.model_config)
    
    def predict(self, x):
        if not self.internal.model:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict(x)

    def evaluate(self, x, y, metrics=['mse']):
        if not self.internal.model:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate(x, y, metric=metrics)
    
    def save(self, checkpoint_dir):
        if not self.internal.model:
            raise RuntimeError("You must call fit or restore first before calling save!")
        model_path = os.path.join(checkpoint_dir, "seq2seqforecaster.model")
        config_path = os.path.join(checkpoint_dir, "seq2seqforecaster.config")
        self.internal.save(model_path, config_path)

    def restore(self, checkpoint_dir):
        model_path = os.path.join(checkpoint_dir, "seq2seqforecaster.model")
        config_path = os.path.join(checkpoint_dir, "seq2seqforecaster.config")
        config = load_config(config_path)
        self.internal.restore(model_path, **config)