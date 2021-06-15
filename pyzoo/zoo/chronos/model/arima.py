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
import os
import pickle
from pmdarima.arima import ARIMA
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs

from zoo.automl.common.metrics import Evaluator
from zoo.automl.model.abstract import BaseModel
from zoo.automl.model import ModelBuilder


class ARIMAModel(BaseModel):

    def __init__(self, config={}):
        """
        Initialize Model
        """
        self.p = config.get('p', 2)
        self.d = config.get('d', 0)
        self.q = config.get('q', 2)
        self.seasonal = config.get('seasonality_mode', True)
        self.P = config.get('P', 1)
        self.D = config.get('D', 0)
        self.Q = config.get('Q', 1)
        self.m = config.get('m', 7)
        self.metric = config.get('metric', 'mse')
        self.model = None
        self.model_init = False

    def set_params(self, **config):
        self.p = config.get('p', self.p)
        self.d = config.get('d', self.d)
        self.q = config.get('q', self.q)
        self.seasonal = config.get('seasonality_mode', self.seasonal)
        self.P = config.get('P', self.P) 
        self.D = config.get('D', self.D)
        self.Q = config.get('Q', self.Q)
        self.m = config.get('m', self.m)
        self.metric = config.get('metric', self.metric)

    def _build(self, **config):
        """
        build the models and initialize.
        :param config: hyperparameters for the model
        """
        self.set_params(**config)
        order=(self.p, self.d, self.q)
        if self.seasonal == False:
            seasonal_order=(0, 0, 0, 0)
        else:
            seasonal_order=(self.P, self.D, self.Q, self.m)
        self.model = ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
        self.model_init = True

    def fit_eval(self, data, **config):
        """
        Fit on the training data from scratch.
        :param data: a dict with data['x'] for training data and
            data['val_y'] for evaluation adata
        :return: the evaluation metric value
        """
        x = data['x']
        target = data['val_y']
        
        # Estimating differencing term (d) and seasonal differencing term (D)
        kpss_diffs = ndiffs(x, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(x, alpha=0.05, test='adf', max_d=6)
        d = max(adf_diffs, kpss_diffs)
        D = 0 if self.seasonal == False else nsdiffs(x, m=7, max_D=12)
        config.update(d=d, D=D)        

        if not self.model_init:
            self._build(**config)

        self.model.fit(x)
        val_metric = self.evaluate(x=None, target=target, metrics=[self.metric])[0].item()
        return val_metric

    def predict(self, x=None, horizon=24, update=False, rolling=False):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: ARIMA predicts the horizon steps foreward from the training data. 
            So x should be None as it is not used.
        :param horizon: the number of steps forward to predict
        :param update: whether to update the original model
        :param rolling: whether to use rolling prediction
        :return: predicted result of length horizon
        """
        if x is not None:
            raise ValueError("x should be None")
        if update==True and rolling==False:
            raise Exception("We don't support updating model without rolling prediction currently")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        
        if update==False and rolling==False:
            forecasts = self.model.predict(n_periods=horizon)
        elif rolling==True:
            if update==False:
                self.save("tmp.pkl")

            forecasts = []
            for step in range(horizon):
                fc = self.model.predict(n_periods=1).item()
                forecasts.append(fc)

                # Updates the existing model with a small number of MLE steps for rolling prediction
                self.model.update(fc)

            if update==False:
                self.restore("tmp.pkl")
                os.remove("tmp.pkl")
                
        return forecasts
    
    def evaluate(self, x, target, metrics=['mse'], rolling=False):
        """
        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        y.
        :param x: ARIMA predicts the horizon steps foreward from the training data.
            So x should be None as it is not used.
        :param target: target for evaluation.
        :param metrics: a list of metrics in string format
        :param rolling: whether to use rolling prediction
        :return: a list of metric evaluation results
        """
        if x is not None:
            raise ValueError("We don't support input x currently")
        if target is None:
            raise ValueError("Input invalid target of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling evaluate")
    
        forecasts = self.predict(horizon=len(target), rolling=rolling)
        
        return [Evaluator.evaluate(m, target, forecasts) for m in metrics]

    def save(self, checkpoint_file):
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling save")
        with open(checkpoint_file, 'wb') as fout:
            pickle.dump(self.model, fout)

    def restore(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as fin:
            self.model = pickle.load(fin)
        self.model_init = True

    def _get_required_parameters(self):
        return {}

    def _get_optional_parameters(self):
        return {}


class ARIMABuilder(ModelBuilder):

    def __init__(self, **arima_config):
        """
        Initialize ARIMA Model Builder
        :param ARIMA_config: Other ARIMA hyperparameters. You may refer to
           https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        for the parameter names to specify.
        """
        self.model_config = arima_config.copy()

    def build(self, config):
        """
        Build ARIMA Model
        :param config: Other ARIMA hyperparameters. You may refer to
           https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        for the parameter names to specify.
        """
        from zoo.chronos.model.arima import ARIMAModel
        model = ARIMAModel(config=self.model_config)
        model._build(**config)
        return model

    def build_from_ckpt(self, checkpoint_filename):
        """
        Build ARIMA Model from checkpoint
        :param checkpoint_filename: model checkpoint filename
        """
        from zoo.chronos.model.arima import ARIMAModel
        model = ARIMAModel(config=self.model_config)
        model.restore(checkpoint_filename)
        return model
