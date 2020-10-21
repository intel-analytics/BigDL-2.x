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

import pickle
import pandas as pd
from xgboost.sklearn import XGBRegressor

from xgboost.sklearn import XGBClassifier
from zoo.automl.common.metrics import Evaluator
from zoo.automl.model.abstract import BaseModel


class XGBoost(BaseModel):

    def __init__(self, model_type="regressor", config=None):
        """
        Initialize hyper parameters
        :param check_optional_config:
        :param future_seq_len:
        """
        # models
        if not config:
            config = {}

        self.model_type = model_type
        self.n_estimators = config.get('n_estimators', 1000)
        self.max_depth = config.get('max_depth', 5)
        self.tree_method = config.get('tree_method', 'hist')
        self.n_jobs = config.get('n_jobs', -1)
        self.random_state = config.get('random_state', 2)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.min_child_weight = config.get('min_child_weight', 1)
        self.seed = config.get('seed', 0)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.gamma = config.get('gamma', 0)
        self.reg_alpha = config.get('reg_alpha', 0)
        self.reg_lambda = config.get('reg_lambda', 1)
        self.verbosity = config.get('verbosity', 0)

        if 'metric' not in config:
            if self.model_type == 'regressor':
                self.metric = 'rmse'
            elif self.model_type == 'classifier':
                self.metric = 'logloss'
        else:
            self.metric = config['metric']

        self.model = None
        self.model_init = False

    def set_params(self, **config):
        self.n_estimators = config.get('n_estimators', self.n_estimators)
        self.max_depth = config.get('max_depth', self.max_depth)
        self.tree_method = config.get('tree_method', self.tree_method)
        self.n_jobs = config.get('n_jobs', self.n_jobs)
        self.random_state = config.get('random_state', self.random_state)
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self.min_child_weight = config.get('min_child_weight', self.min_child_weight)
        self.seed = config.get('seed', self.seed)

        self.subsample = config.get('subsample', self.subsample)
        self.colsample_bytree = config.get('colsample_bytree', self.colsample_bytree)
        self.gamma = config.get('gamma', self.gamma)
        self.reg_alpha = config.get('reg_alpha', self.reg_alpha)
        self.reg_lambda = config.get('reg_lambda', self.reg_lambda)
        self.verbosity = config.get('verbosity', self.verbosity)
        self.metric = config.get('metric', self.metric)

    def _build(self, **config):
        """
        build the models and initialize.
        :param config: hyper parameters for building the model
        :return:
        """
        self.set_params(**config)
        if self.model_type == "regressor":
            self.model = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                      n_jobs=self.n_jobs, tree_method=self.tree_method,
                                      random_state=self.random_state,
                                      learning_rate=self.learning_rate,
                                      min_child_weight=self.min_child_weight, seed=self.seed,
                                      subsample=self.subsample,
                                      colsample_bytree=self.colsample_bytree,
                                      gamma=self.gamma, reg_alpha=self.reg_alpha,
                                      reg_lambda=self.reg_lambda, verbosity=self.verbosity)
        elif self.model_type == "classifier":
            self.model = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                       n_jobs=self.n_jobs, tree_method=self.tree_method,
                                       random_state=self.random_state,
                                       learning_rate=self.learning_rate,
                                       min_child_weight=self.min_child_weight, seed=self.seed,
                                       subsample=self.subsample,
                                       colsample_bytree=self.colsample_bytree,
                                       gamma=self.gamma, reg_alpha=self.reg_alpha,
                                       objective='binary:logistic',
                                       reg_lambda=self.reg_lambda, verbosity=self.verbosity)
        else:
            raise ValueError("model_type can only be \"regressor\" or \"classifier\"")

        self.model_init = True

    def fit_eval(self, x, y, validation_data=None, **config):
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
        if validation_data is not None and type(validation_data) is not list:
            validation_data = [validation_data]

        self.model.fit(x, y, eval_set=validation_data, eval_metric=self.metric)
        vals = self.model.evals_result_.get("validation_0").get(self.metric)
        res = sum(vals) / len(vals)
        return res

    def predict(self, x):
        """
        Predict horizon time-points ahead the input x in fit_eval
        :param x: We don't support input x currently.
        :param horizon: horizon length to predict
        :param mc:
        :return:
        """
        if x is None:
            raise Exception("Input invalid x of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")
        self.model.n_jobs = self.n_jobs
        out = self.model.predict(x)
        output_df = pd.DataFrame(out)

        return output_df

    def evaluate(self, x, y, metrics=['mse']):
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
        if x is None:
            raise ValueError("Input invalid x of None")
        if y is None:
            raise ValueError("Input invalid y of None")
        if self.model is None:
            raise Exception("Needs to call fit_eval or restore first before calling predict")

        self.model.n_jobs = self.n_jobs
        y_pred = self.predict(x)
        return [Evaluator.evaluate(m, y, y_pred) for m in metrics]

    def save(self, model_file, config_path=None):
        pickle.dump(self.model, open(model_file, "wb"))

    def restore(self, model_file, **config):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        self.model_init = True

    def _get_required_parameters(self):
        return {}

    def _get_optional_parameters(self):
        param = self.model.get_xgb_params
        return param
