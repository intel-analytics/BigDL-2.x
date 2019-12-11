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
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima_model import ARIMAResults
from zoo.automl.common.metrics import Evaluator


class RandomWalk(object):
    def __init__(self):
        self.model = None

    @staticmethod
    def generate_1d_sample(time_steps=1000,
                           b0=0.0,
                           b1=1.0,
                           b2=0.05,
                           with_white_noise=True):
        """
            y(t) = B0 + B1 * X(t-1) + B2 * e(t)
            generate sample that is random walk
            :param time_steps: time series length
            :param b0: is a coefficient that if set to a value other than zero adds a constant drift to the random walk
            :param b1: is a coefficient to weight the previous time step and is set to 1.0
            :param b2: is a coefficient to white noise
            :param with_white_noise: is the white noise or random fluctuation at that time.
            :return: ndarray
        """
        np.random.seed(1)
        if with_white_noise:
            steps = np.full(time_steps, b0) + b1 * np.random.choice([-1, 1], size=time_steps)\
                    + b2 * np.random.randn(time_steps)
        else:
            steps = np.full(time_steps, b0) + b1 * np.random.choice([-1, 1], size=time_steps)
        random_walk = np.cumsum(steps)
        return random_walk

    @staticmethod
    def random_walk_test(x):
        """
            check a time series is a random walk
            :param x: time series data
            :return: boolean
        """
        results = adfuller(diff(x))
        if results[1] < 0.05:
            return True
        else:
            return False

    def rolling_fit_eval(self, train_data, test_data, metric=None, **config):
        """
            fit and eval time series data
            :param train_data: The train data.
            :param test_data: The test data.
            :param metric: a list of metrics in string format
            :param config: hyper parameters
            :return: boolean
        """
        if metric is None:
            metric = ['mse']
        history = [x for x in train_data]
        predictions = list()
        for t in range(len(test_data)):
            # ARIMA(0,1,0) with a constant is a random walk with drift.
            arima = ARIMA(history, order=(0, 1, 0))
            self.model = arima.fit(**config)
            output = self.model.forecast()
            y_hat = output[0]
            predictions.append(y_hat)
            obs = test_data[t]
            history.append(obs)

        return [Evaluator.evaluate(m, test_data, predictions) for m in metric]

    def fit_eval(self, train_data, test_data, metric=None, **config):
        """
            fit and eval time series data
            :param train_data: The train data.
            :param test_data: The test data.
            :param metric: a list of metrics in string format
            :param config: hyper parameters
            :return: boolean
        """
        if metric is None:
            metric = ['mse']
        arima = ARIMA(train_data, order=(0, 1, 0))

        self.model = arima.fit(**config)

        output = self.model.forecast(steps=len(test_data))
        y_hat = output[0]

        return [Evaluator.evaluate(m, test_data, y_hat) for m in metric]

    def evaluate(self, test_data, metric=None):
        """
            evaluate time series data
            :param test_data: The test data.
            :param metric: a list of metrics in string format
            :return: boolean
        """
        y_pred = self.predict(steps=len(test_data))
        return [Evaluator.evaluate(m, test_data, y_pred) for m in metric]

    def predict(self, steps=1):
        """
            predict time series data
            :param steps: The number of out of sample forecasts from the end of the sample.
            :return: boolean
        """
        output = self.model.forecast(steps=steps)
        y_hat = output[0]
        return y_hat

    def save(self, model_path):
        """
        save model to file.
        :param model_path: the model file.
        :return:
        """
        self.model.save(model_path)

    def restore(self, model_path):
        """
            restore model from file
            :param model_path: the model file
            :return: the restored model
        """
        self.model = ARIMAResults.load(model_path)

