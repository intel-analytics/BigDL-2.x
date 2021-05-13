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

import json
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from zoo.automl.common.metrics import Evaluator
from zoo.automl.model.abstract import BaseModel
from zoo.automl.model import ModelBuilder


class ProphetModel(BaseModel):

    def __init__(self, config={}):
        """
        Initialize Model
        """

        self.changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        self.holidays_prior_scale = config.get('holidays_prior_scale', 10.0)
        self.seasonality_mode = config.get('seasonality_mode', 'additive')
        self.changepoint_range = config.get('changepoint_range', 0.8)
        self.metric = config.get('metric', 'mse')
        self.train_data = None
        self.train_mode = True
        self.fit_done = False
        self.model = None
        self.model_init = False

    def set_params(self, **config):
        self.changepoint_prior_scale = config.get('changepoint_prior_scale', self.changepoint_prior_scale)
        self.seasonality_prior_scale = config.get('seasonality_prior_scale', self.seasonality_prior_scale)
        self.holidays_prior_scale = config.get('holidays_prior_scale', self.holidays_prior_scale)
        self.seasonality_mode = config.get('seasonality_mode', self.seasonality_mode)
        self.changepoint_range = config.get('changepoint_range', self.changepoint_range)
        self.metric = config.get('metric', self.metric)
        
    def _build(self, **config):
        """
        build the models and initialize.
        :param config: hyperparameters for the model
        """
        self.set_params(**config)
        self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                             seasonality_prior_scale=self.seasonality_prior_scale,
                             holidays_prior_scale=self.holidays_prior_scale,
                             changepoint_range=self.changepoint_range,
                             seasonality_mode=self.seasonality_mode)
        self.model_init = True

    def fit_eval(self, x, y=None, **config):
        """
        Fit on the training data from scratch.
        Since the rolling process is very customized in this model,
        we enclose the rolling process inside this method.

        :param x: training data, an dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating target
            and Td is the time dimension
        :param y: None. target is extracted from x directly
        :return: the evaluation metric value
        """
        if self.train_data is None:
            self.train_data = x.copy()
        
        if not self.model_init:
            self._build(**config)
        val_len = config.get("val_len", 24)
        self.model.fit(x[:-val_len])
        
        val_pred = self.predict(horizon=val_len)
        val_metric = Evaluator.evaluate(self.metric, x[['y']].values[-val_len:], val_pred[['yhat']].values)
        return val_metric

    def predict(self, x=None, horizon=24):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: We don't support input x currently.
        :param horizon: horizon length to predict
        :return: predicted result of length horizon
        """
        if not self.train_mode and not self.fit_done:
            self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                                 seasonality_prior_scale=self.seasonality_prior_scale,
                                 holidays_prior_scale=self.holidays_prior_scale,
                                 changepoint_range=self.changepoint_range,
                                 seasonality_mode=self.seasonality_mode)
            self.model.fit(self.train_data)
            self.fit_done = True

        if x is not None:
            raise Exception("We don't support input x currently")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        future = self.model.make_future_dataframe(periods=horizon)
        out = self.model.predict(future)[-horizon:]

        return out

    def evaluate(self, x, y, metrics=['mse']):
        """
        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        y.
        :param x: We don't support input x currently.
        :param y: target for evaluation.
        :param metrics: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        if x is not None:
            raise ValueError("We don't support input x currently")
        if y is None:
            raise ValueError("Input invalid y of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")

        self.train_mode = False        
        horizon = len(y)
        y = y[['y']]
        future = self.model.make_future_dataframe(periods=horizon)
        y_pred = self.predict(horizon=horizon)[['yhat']]
        return [Evaluator.evaluate(m, y.values, y_pred.values) for m in metrics]

    def save(self, checkpoint_file):
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling save")
        if not self.fit_done:
            self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                                 seasonality_prior_scale=self.seasonality_prior_scale,
                                 holidays_prior_scale=self.holidays_prior_scale,
                                 changepoint_range=self.changepoint_range,
                                 seasonality_mode=self.seasonality_mode)
            self.model.fit(self.train_data)
            self.fit_done = True
        with open(checkpoint_file, 'w') as fout:
            json.dump(model_to_json(self.model), fout)

    def restore(self, checkpoint_file):
        with open(checkpoint_file, 'r') as fin:
            self.model = model_from_json(json.load(fin))
        self.model_init = True
        self.train_mode = False
        self.fit_done = True

    def _get_required_parameters(self):
        return {}

    def _get_optional_parameters(self):
        return {}

class ProphetBuilder(ModelBuilder):

    def __init__(self, **prophet_config):
        self.model_config = prophet_config.copy()

    def build(self, config):
        from Prophet import ProphetModel
        model = ProphetModel(config=self.model_config)
        model._build(**config)
        return model
    
    def build_from_ckpt(self, checkpoint_filename):
        from Prophet import ProphetModel
        model = ProphetModel(config=self.model_config)
        model.restore(checkpoint_filename)
        return model
