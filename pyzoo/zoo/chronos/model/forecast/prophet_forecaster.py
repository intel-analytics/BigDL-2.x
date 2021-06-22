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
from zoo.chronos.model.prophet import ProphetModel


class ProphetForecaster(Forecaster):
    """
    Prophet Forecaster
    Prophet is a procedure for forecasting time series data based on an additive model
    where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
    It works best with time series that have strong seasonal effects and several seasons of
    historical data. Prophet is robust to missing data and shifts in the trend, and
    typically handles outliers well.
    
    Source: https://github.com/facebook/prophet
    """

    def __init__(self,
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0,
                 seasonality_mode='additive',
                 changepoint_range=0.8,
                 metric="mse",
                 ):
        """
        Build a Prophet Forecast Model.

        :param changepoint_prior_scale: hyperparameter changepoint_prior_scale for the
            Prophet model, for details you may refer to
            https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        :param seasonality_prior_scale: hyperparameter seasonality_prior_scale for the
            Prophet model, for details you may refer to
            https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        :param holidays_prior_scale: hyperparameter holidays_prior_scale for the
            Prophet model, for details you may refer to
            https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        :param seasonality_mode: hyperparameter seasonality_mode for the
            Prophet model, for details you may refer to
            https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        :param changepoint_range: hyperparameter changepoint_range for the
            Prophet model, for details you may refer to
            https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        :param metric: the metric for validation and evaluation. For regression, we support
            Mean Squared Error: ("mean_squared_error", "MSE" or "mse"),
            Mean Absolute Error: ("mean_absolute_error","MAE" or "mae"),
            Mean Absolute Percentage Error: ("mean_absolute_percentage_error", "MAPE", "mape")
            Cosine Proximity: ("cosine_proximity", "cosine")
        """
        self.model_config = {
            "changepoint_prior_scale" : changepoint_prior_scale,
            "seasonality_prior_scale" : seasonality_prior_scale,
            "holidays_prior_scale" : holidays_prior_scale,
            "seasonality_mode" : seasonality_mode,
            "changepoint_range" : changepoint_range,
            "metric": metric
        }
        self.internal = ProphetModel()

        super().__init__()

    def fit(self, x, target):
        """
        Fit(Train) the forecaster.

        :param x: training data, a dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating value
            and Td is the time dimension
        :param target: evaluation data, should be the same type as x
        """
        self._check_data(x, target)
        data = {'x': x, 'y': None, 'val_x': None, 'val_y': target}
        return self.internal.fit_eval(data=data,
                                      **self.model_config)

    def _check_data(self, x, target):
        assert 'ds' in x.columns and 'y' in x.columns, \
            "x should be a dataframe that has at least 2 columns 'ds' and 'y'."\
            .format(x.ndim)
        assert 'ds' in target.columns and 'y' in target.columns, \
            "target should be a dataframe that has at least 2 columns 'ds' and 'y'."\
            .format(x.ndim)

    def predict(self, horizon):
        """
        Predict using a trained forecaster.

        :param horizon: the number of steps forward to predict
        :param rolling: whether to use rolling prediction
        """
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.predict(horizon=horizon)

    def evaluate(self, target, x=None, metrics=['mse']):
        """
        Evaluate using a trained forecaster.

        :param target: evaluation data, a dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating value
            and Td is the time dimension
        :param x: We don't support input x currently.
        :param metrics: A list contains metrics for test/valid data.
        """
        if x is not None:
            raise ValueError("We don't support input x currently")
        if target is None:
            raise ValueError("Input invalid target of None")
        if self.internal.model is None:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        return self.internal.evaluate(x, target, metrics=metrics)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        :param checkpoint_file: The location you want to save the forecaster, should be a json file
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
