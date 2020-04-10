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
import shutil

import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.model.MTNet_keras import MTNetKeras
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
import pandas as pd
import numpy as np
import tensorflow as tf
from zoo.automl.common.util import save, restore
from numpy.testing import assert_array_almost_equal


class TestMTNetKeras(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        self.ft = TimeSequenceFeatureTransformer()
        self.create_data()
        self.model = MTNetKeras()
        self.config = {"long_num": self.long_num,
                       "time_step": self.time_step,
                       "ar_window": np.random.randint(1, 3),
                       "cnn_height": np.random.randint(1, 3),
                       "epochs": 1}

    def teardown_method(self, method):
        pass

    def create_data(self):
        def gen_train_sample(data, past_seq_len, future_seq_len):
            data = pd.DataFrame(data)
            x, y = self.ft._roll_train(data,
                                       past_seq_len=past_seq_len,
                                       future_seq_len=future_seq_len
                                       )
            return x, y

        def gen_test_sample(data, past_seq_len):
            test_data = pd.DataFrame(data)
            x = self.ft._roll_test(test_data, past_seq_len=past_seq_len)
            return x

        self.long_num = 6
        self.time_step = 2
        look_back = (self.long_num + 1) * self.time_step
        look_forward = 1
        self.x_train, self.y_train = gen_train_sample(data=np.random.randn(
            64, 4), past_seq_len=look_back, future_seq_len=look_forward)
        self.x_val, self.y_val = gen_train_sample(data=np.random.randn(16, 4),
                                                  past_seq_len=look_back,
                                                  future_seq_len=look_forward)
        self.x_test = gen_test_sample(data=np.random.randn(16, 4),
                                      past_seq_len=look_back)

    def test_fit_evaluate(self):
        self.model.fit_eval(self.x_train, self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            **self.config)
        self.model.evaluate(self.x_val, self.y_val)

    def test_save_restore(self):
        self.model.fit_eval(self.x_train, self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            **self.config)
        y_pred = self.model.predict(self.x_test)
        assert y_pred.shape == (self.x_test.shape[0], self.y_train.shape[1])
        dirname = "tmp"
        restored_model = MTNetKeras()
        try:
            save(dirname, model=self.model)
            restore(dirname, model=restored_model, config=self.config)
            predict_after = restored_model.predict(self.x_test)
            assert_array_almost_equal(y_pred, predict_after, decimal=2), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(y_pred, predict_after)
            restored_model.fit_eval(self.x_train, self.y_train, epochs=1)
            restored_model.evaluate(self.x_val, self.y_val)
        finally:
            shutil.rmtree("tmp")

    def test_predict_with_uncertainty(self):
        self.model.fit_eval(self.x_train, self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            mc=True,
                            **self.config)
        pred, uncertainty = self.model.predict_with_uncertainty(self.x_test, n_iter=2)
        assert pred.shape == (self.x_test.shape[0], self.y_train.shape[1])
        assert uncertainty.shape == pred.shape
        assert np.any(uncertainty)


if __name__ == '__main__':
    pytest.main([__file__])
