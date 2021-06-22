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
import pandas as pd
import tempfile
import os

from zoo.chronos.model.forecast.prophet_forecaster import ProphetForecaster
from unittest import TestCase
import pytest


def create_data():
    seq_len = 400
    x = pd.DataFrame(pd.date_range('20130101', periods=seq_len), columns=['ds'])
    x.insert(1, 'y', np.random.rand(seq_len))
    horizon = np.random.randint(2, 50)
    target = pd.DataFrame(pd.date_range('20140426', periods=horizon), columns=['ds'])
    target.insert(1, 'y', np.random.rand(horizon))
    return x, target


class TestChronosModelProphetForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_prophet_forecaster_fit_eval_pred(self):
        x, target = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )
        train_loss = forecaster.fit(x, target)
        test_pred = forecaster.predict(target.shape[0])
        assert test_pred.shape[0] == target.shape[0]
        test_mse = forecaster.evaluate(target)

    def test_prophet_forecaster_save_restore(self):
        x, target = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )
        train_loss = forecaster.fit(x, target)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "json")
            test_pred_save = forecaster.predict(target.shape[0])
            forecaster.save(ckpt_name)
            forecaster.restore(ckpt_name)
            test_pred_restore = forecaster.predict(target.shape[0])
        assert (test_pred_save['yhat']==test_pred_restore['yhat']).all()

    def test_prophet_forecaster_runtime_error(self):
        x, target = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling predict!"):
            forecaster.predict(horizon=target.shape[0])

        with pytest.raises(Exception,
                           match="Input invalid target of None"):
            forecaster.evaluate(target=None)

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling evaluate!"):
            forecaster.evaluate(target=target)

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling save!"):
            model_file = "tmp.json"
            forecaster.save(model_file)

    def test_prophet_forecaster_shape_error(self):
        x, target = create_data()
        forecaster = ProphetForecaster(changepoint_prior_scale=0.05,
                                       seasonality_prior_scale=10.0,
                                       holidays_prior_scale=10.0,
                                       seasonality_mode='additive',
                                       changepoint_range=0.8,
                                       metric="mse",
                                       )

        with pytest.raises(AssertionError):
            forecaster.fit(x[['ds']], target)

        with pytest.raises(AssertionError):
            forecaster.fit(x, target[['ds']])
