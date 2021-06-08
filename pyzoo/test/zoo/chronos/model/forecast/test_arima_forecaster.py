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
import tempfile
import os

from zoo.chronos.model.forecast.arima_forecaster import ARIMAForecaster
from unittest import TestCase
import pytest


def create_data():
    seq_len = 1095
    x = np.random.rand(seq_len)
    horizon = np.random.randint(2, 50)
    target = np.random.rand(horizon)
    data = {'x': x, 'y': None, 'val_x': None, 'val_y': target}
    return data


class TestChronosModelARIMAForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_arima_forecaster_fit_eval_pred(self):
        data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )
        train_loss = forecaster.fit(data['x'], data['val_y'])
        test_pred = forecaster.predict(len(data['val_y']))
        assert len(test_pred) == len(data['val_y'])
        test_mse = forecaster.evaluate(None, data['val_y'])


    def test_tcn_forecaster_save_restore(self):
        data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )
        train_loss = forecaster.fit(data['x'], data['val_y'])
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "pkl")
            test_pred_save = forecaster.predict(len(data['val_y']))
            forecaster.save(ckpt_name)
            forecaster.restore(ckpt_name)
            test_pred_restore = forecaster.predict(len(data['val_y']))
        np.testing.assert_almost_equal(test_pred_save, test_pred_restore)

    def test_tcn_forecaster_runtime_error(self):
        data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling predict!"):
            forecaster.predict(horizon=len(data['val_y']))

        with pytest.raises(Exception,
                           match="We don't support input x currently"):
            forecaster.evaluate(x=1, target=data['val_y'])

        with pytest.raises(Exception,
                           match="Input invalid target of None"):
            forecaster.evaluate(x=None, target=None)

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling evaluate!"):
            forecaster.evaluate(x=None, target=data['val_y'])

        with pytest.raises(Exception,
                           match="You must call fit or restore first before calling save!"):
            model_file = "tmp.pkl"
            forecaster.save(model_file)


    def test_tcn_forecaster_shape_error(self):
        data = create_data()
        forecaster = ARIMAForecaster(p=2,
                                     q=2,
                                     seasonality_mode=True,
                                     P=1,
                                     Q=1,
                                     m=7
                                     )
        
        with pytest.raises(AssertionError):
            forecaster.fit(data['x'].reshape(-1, 1), data['val_y'])
        
        with pytest.raises(AssertionError):
            forecaster.fit(data['x'], data['val_y'].reshape(-1, 1))
