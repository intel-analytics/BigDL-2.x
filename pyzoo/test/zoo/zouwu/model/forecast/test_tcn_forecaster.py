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

from zoo.zouwu.model.forecast.tcn_forecaster import TCNForecaster
from unittest import TestCase
import pytest


def create_data():
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 24
    input_feature_dim = 1
    output_time_steps = 5
    output_feature_dim = 1

    def get_x_y(num_samples):
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim)
        y = x[:, -output_time_steps:, :]*2 + \
            np.random.rand(num_samples, output_time_steps, output_feature_dim)
        return x, y

    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return train_data, val_data, test_data


class TestZouwuModelTCNForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tcn_forecaster_fit_eva_pred(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(kernel_size=3, lr=0.01)
        train_mse = forecaster.fit(train_data[0], train_data[1], epochs=2)
        forecaster.predict(test_data[0])
        test_mse = forecaster.evaluate(test_data[0], test_data[1])

    def test_tcn_forecaster_save_restore(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(kernel_size=3, lr=0.01)
        train_mse = forecaster.fit(train_data[0], train_data[1], epochs=2)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "ckpt")
            forecaster.save(ckpt_name)
            forecaster.restore(ckpt_name)
            test_mse = forecaster.evaluate(test_data[0], test_data[1])

    def test_tcn_forecaster_runtime_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = TCNForecaster(kernel_size=3, lr=0.01)
        with pytest.raises(RuntimeError):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                ckpt_name = os.path.join(tmp_dir_name, "ckpt")
                forecaster.save(ckpt_name)
        with pytest.raises(RuntimeError):
            forecaster.predict(test_data[0])
        with pytest.raises(RuntimeError):
            forecaster.evaluate(test_data[0], test_data[1])
