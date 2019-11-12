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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.model.VanillaLSTM import *
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer


class TestVanillaLSTM(ZooTestCase):

    def setup_method(self, method):
        super().setup_method(method)
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        val_data = pd.DataFrame(data=np.random.randn(16, 4))
        test_data = pd.DataFrame(data=np.random.randn(16, 4))

        future_seq_len = 1
        past_seq_len = 6

        # use roll method in time_sequence
        tsft = TimeSequenceFeatureTransformer()
        self.x_train, self.y_train = tsft._roll_train(train_data,
                                                      past_seq_len=past_seq_len,
                                                      future_seq_len=future_seq_len)
        self.x_val, self.y_val = tsft._roll_train(val_data,
                                                  past_seq_len=past_seq_len,
                                                  future_seq_len=future_seq_len)
        self.x_test = tsft._roll_test(test_data, past_seq_len=past_seq_len)
        self.config = {
            'epochs': 1,
            "lr": 0.001,
            "lstm_1_units": 16,
            "dropout_1": 0.2,
            "lstm_2_units": 10,
            "dropout_2": 0.2,
            "batch_size": 32,
        }
        self.model = VanillaLSTM(check_optional_config=False, future_seq_len=future_seq_len)

    def test_fit_eval(self):
        print("fit_eval:", self.model.fit_eval(self.x_train,
                                               self.y_train,
                                               **self.config))

    def test_fit_eval_mc(self):
        print("fit_eval:", self.model.fit_eval(self.x_train,
                                               self.y_train,
                                               mc=True,
                                               **self.config))

    def test_evaluate(self):
        self.model.fit_eval(self.x_train, self.y_train, **self.config)
        mse, rs = self.model.evaluate(self.x_val,
                                      self.y_val,
                                      metric=['mse', 'r2'])
        print("Mean squared error is:", mse)
        print("R square is:", rs)

    def test_predict(self):
        self.model.fit_eval(self.x_train, self.y_train, **self.config)
        self.y_pred = self.model.predict(self.x_test)
        assert self.y_pred.shape == (self.x_test.shape[0], 1)

    def test_save_restore(self):
        new_model = VanillaLSTM(check_optional_config=False)
        self.model.fit_eval(self.x_train, self.y_train, **self.config)
        predict_before = self.model.predict(self.x_test)

        dirname = tempfile.mkdtemp(prefix="automl_test_vanilla")
        try:
            save(dirname, model=self.model)
            restore(dirname, model=new_model, config=self.config)
            predict_after = new_model.predict(self.x_test)
            assert np.allclose(predict_before, predict_after)
            new_config = {'epochs': 2}
            new_model.fit_eval(self.x_train, self.y_train, **new_config)

        finally:
            shutil.rmtree(dirname)

    def test_predict_with_uncertainty(self,):
        self.model.fit_eval(self.x_train, self.y_train, mc=True, **self.config)
        prediction, uncertainty = self.model.predict_with_uncertainty(self.x_test, n_iter=10)
        assert prediction.shape == (self.x_test.shape[0], 1)
        assert uncertainty.shape == (self.x_test.shape[0], 1)
        assert np.any(uncertainty)

        new_model = VanillaLSTM(check_optional_config=False)
        dirname = tempfile.mkdtemp(prefix="automl_test_feature")
        try:
            save(dirname, model=self.model)
            restore(dirname, model=new_model, config=self.config)
            prediction, uncertainty = new_model.predict_with_uncertainty(self.x_test, n_iter=2)
            assert prediction.shape == (self.x_test.shape[0], 1)
            assert uncertainty.shape == (self.x_test.shape[0], 1)
            assert np.any(uncertainty)
        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
