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

import pytest
import numpy as np
import tempfile
import os
import tensorflow as tf

from zoo.zouwu.model.forecast.seq2seq_forecaster import Seq2SeqForecaster
from unittest import TestCase

def create_data():
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 24
    input_feature_dim = 2
    output_time_steps = 5
    output_feature_dim = 2

    def get_x_y(num_samples):
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim)
        y = x[:, -output_time_steps:, :]*2 + \
            np.random.rand(num_samples, output_time_steps, output_feature_dim)
        return x, y

    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return train_data, val_data, test_data

class TestZouwuModelSeq2SeqForecaster(TestCase):

    def setUp(self):
        tf.keras.backend.clear_session()

    def tearDown(self):
        pass

    def test_s2s_forecaster_fit_eva_pred(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                  future_seq_len=5,
                                  feature_num=2,
                                  target_col_num=2)
        train_mse = forecaster.fit(train_data[0], train_data[1], epochs=10)
        test_pred = forecaster.predict(test_data[0])
        assert test_pred.shape == test_data[1].shape
        test_mse = forecaster.evaluate(test_data[0], test_data[1])
    
    def test_s2s_forecaster_save_restore(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                  future_seq_len=5,
                                  feature_num=2,
                                  target_col_num=2)
        train_mse = forecaster.fit(train_data[0], train_data[1], epochs=10)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            test_pred_save = forecaster.predict(test_data[0])
            forecaster.save(tmp_dir_name)
            forecaster.restore(tmp_dir_name)
            test_pred_restore = forecaster.predict(test_data[0])
        np.testing.assert_almost_equal(test_pred_save, test_pred_restore)
    
    def test_tcn_forecaster_runtime_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                  future_seq_len=5,
                                  feature_num=2,
                                  target_col_num=2)
        with pytest.raises(RuntimeError):
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                ckpt_name = os.path.join(tmp_dir_name, "ckpt")
                forecaster.save(ckpt_name)
        with pytest.raises(RuntimeError):
            forecaster.predict(test_data[0])
        with pytest.raises(RuntimeError):
            forecaster.evaluate(test_data[0], test_data[1])

    def test_tcn_forecaster_shape_error(self):
        train_data, val_data, test_data = create_data()
        forecaster = Seq2SeqForecaster(past_seq_len=24,
                                  future_seq_len=5,
                                  feature_num=2,
                                  target_col_num=1)
        with pytest.raises(AssertionError):
            forecaster.fit(train_data[0], train_data[1], epochs=2)
