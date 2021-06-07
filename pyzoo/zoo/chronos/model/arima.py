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

    def fit_eval(self, x, target, **config):
        """
        Fit on the training data from scratch.

        :param x: training data
        :param target: evaluation data for evaluation
        :return: the evaluation metric value
        """
        
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

    def predict(self, x=None, horizon=24):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: We don't support input x currently.
        :param horizon: horizon length to predict
        :return: predicted result of length horizon
        """
        if x is not None:
            raise ValueError("We don't support input x currently")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        
        forecasts = self.model.predict(n_periods=horizon)

        return forecasts
    
    def roll_predict(self, target):
        """
        Rolling predict horizon time-points ahead the input x in fit_eval, note that 
        the model will be updated for rolling prediction
        :param x: We don't support input x currently.
        :param target: target for rolling prediction
        :return: predicted result of length same as target
        """
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        
        forecasts = []
        for new_ob in target:
            fc = self.model.predict(n_periods=1).item()
            forecasts.append(fc)
            
            # Updates the existing model with a small number of MLE steps for rolling prediction
            self.model.update(new_ob)

        return forecasts
    
    def evaluate(self, x, target, metrics=['mse']):
        """
        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x
        in fit_eval before evaluation, where the horizon length equals the second dimension size of
        y.
        :param x: We don't support input x currently.
        :param target: target for evaluation.
        :param metrics: a list of metrics in string format
        :return: a list of metric evaluation results
        """
        if x is not None:
            raise ValueError("We don't support input x currently")
        if target is None:
            raise ValueError("Input invalid target of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling evaluate")
    
        forecasts = self.predict(horizon=len(target))
        
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
        Initialize ARIMA Model
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
        from zoo.zouwu.model.arima import ARIMAModel
        model = ARIMAModel(config=self.model_config)
        model._build(**config)
        return model

    def build_from_ckpt(self, checkpoint_filename):
        """
        Build ARIMA Model from checkpoint
        :param checkpoint_filename: model checkpoint filename
        """
        from zoo.zouwu.model.arima import ARIMAModel
        model = ARIMAModel(config=self.model_config)
        model.restore(checkpoint_filename)
        return model
