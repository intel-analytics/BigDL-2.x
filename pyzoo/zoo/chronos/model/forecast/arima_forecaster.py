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
from zoo.chronos.model.arima import ARIMAModel


class ARIMAForecaster(Forecaster):
    """
    ARIMA Forecaster
    """

    def __init__(self,
                 horizon=1,
                 p=2,
                 q=2,
                 seasonality_mode=True
                 P=1,
                 Q=1,
                 m=7,
                 metric="mse",
                 ):
        """
        Build a ARIMA Forecast Model. 
        User need to set p, q, P, Q, m for the ARIMA model, the differencing term (d) and
        seasonal differencing term (D) are automatically estimated from the data.
        
        :param p: hyperparameter p for the ARIMA model, for details you may refer to
            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        :param q: hyperparameter q for the ARIMA model, for details you may refer to
            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        :param seasonality_mode: hyperparameter q for the ARIMA model, for details you may refer to
            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        :param P: hyperparameter P for the ARIMA model, for details you may refer to
            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        :param Q: hyperparameter Q for the ARIMA model, for details you may refer to
            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        :param m: hyperparameter m for the ARIMA model, for details you may refer to
            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA
        :param metric: the metric for validation and evaluation. For regression, we support
            Mean Squared Error: ("mean_squared_error", "MSE" or "mse"),
            Mean Absolute Error: ("mean_absolute_error","MAE" or "mae"),
            Mean Absolute Percentage Error: ("mean_absolute_percentage_error", "MAPE", "mape")
            Cosine Proximity: ("cosine_proximity", "cosine")
        """
        self.model_config = {
            "p": p,
            "q": q,
            "seasonality_mode": seasonality_mode,
            "P": P,
            "Q": Q,
            "metric": metric,
        }
        self.internal = None

        super().__init__()

        
    def fit(self, x, target):
        """
        Fit(Train) the forecaster.

        :param x: A 1-D numpy array as the training data
        :param target: A 1-D numpy array as the evaluation data
        """
        self._check_data(x, target)
        x = x.reshape(-1, 1)
        target = target.reshape(-1, 1)
        return self.internal.fit_eval(x=x,
                                      target=target,
                                      **self.config)

    def _check_data(self, x, y):
        assert x.ndim == 1, \
            "x should be an 1-D array), \
            Got x dimension of {}."\
            .format(x.ndim)
        assert target.ndim == 1, \
            "The target should be an 1-D array), \
            Got target dimension of {}."\
            .format(target.ndim)
        
    def predict(self, horizon):
        """
        Predict using a trained forecaster.

        :param x: A 1-D numpy array with length horizon.
        """
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict(horizon=horizon)

    def evaluate(self, x, y, metrics=['mse']):
        """
        Evaluate using a trained forecaster.

        :param x: A 1-D numpy array as the training data
        :param target: A 1-D numpy array as the evaluation data
        :param metrics: A list contains metrics for test/valid data.
        """
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate(x, y, metrics=metrics, multioutput=multioutput)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling save!")
        self.internal.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        Restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        self.internal.restore(checkpoint_file)
