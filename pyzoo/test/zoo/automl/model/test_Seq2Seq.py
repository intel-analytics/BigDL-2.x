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
import tempfile

import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.model.Seq2Seq import *
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from numpy.testing import assert_array_almost_equal


class TestSeq2Seq(ZooTestCase):

    def setup_method(self, method):
        # super().setup_method(method)
        self.train_data = pd.DataFrame(data=np.random.randn(64, 4))
        self.val_data = pd.DataFrame(data=np.random.randn(16, 4))
        self.test_data = pd.DataFrame(data=np.random.randn(16, 4))

        self.past_seq_len = 6
        self.future_seq_len_1 = 1
        self.future_seq_len_2 = 2

        # use roll method in time_sequence
        self.feat = TimeSequenceFeatureTransformer()

        self.config = {
            'batch_size': 32,
            'epochs': 1
        }

        self.model_1 = LSTMSeq2Seq(check_optional_config=False,
                                   future_seq_len=self.future_seq_len_1)
        self.model_2 = LSTMSeq2Seq(check_optional_config=False,
                                   future_seq_len=self.future_seq_len_2)

        self.fitted = False
        self.predict_1 = None
        self.predict_2 = None

    def teardown_method(self, method):
        pass

    def test_fit_eval_1(self):
        x_train_1, y_train_1 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_1)
        print("fit_eval_future_seq_len_1:",
              self.model_1.fit_eval(x_train_1, y_train_1, **self.config))
        assert self.model_1.past_seq_len == 6
        assert self.model_1.feature_num == 4
        assert self.model_1.future_seq_len == 1
        assert self.model_1.target_col_num == 1

    def test_fit_eval_2(self):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        print("fit_eval_future_seq_len_2:",
              self.model_2.fit_eval(x_train_2, y_train_2, **self.config))
        assert self.model_2.future_seq_len == 2

        self.fitted = True

    def test_evaluate_1(self):
        x_train_1, y_train_1 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_1)
        x_val_1, y_val_1 = self.feat._roll_train(self.val_data,
                                                 past_seq_len=self.past_seq_len,
                                                 future_seq_len=self.future_seq_len_1)

        self.model_1.fit_eval(x_train_1, y_train_1, **self.config)

        print("evaluate_future_seq_len_1:", self.model_1.evaluate(x_val_1,
                                                                  y_val_1,
                                                                  metric=['mse',
                                                                          'r2']))

    def test_evaluate_2(self):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        x_val_2, y_val_2 = self.feat._roll_train(self.val_data,
                                                 past_seq_len=self.past_seq_len,
                                                 future_seq_len=self.future_seq_len_2)

        self.model_2.fit_eval(x_train_2, y_train_2, **self.config)

        print("evaluate_future_seq_len_2:", self.model_2.evaluate(x_val_2,
                                                                  y_val_2,
                                                                  metric=['mse',
                                                                          'r2']))

    def test_predict_1(self):
        x_train_1, y_train_1 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_1)
        x_test_1 = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        self.model_1.fit_eval(x_train_1, y_train_1, **self.config)

        predict_1 = self.model_1.predict(x_test_1)
        assert predict_1.shape == (x_test_1.shape[0], self.future_seq_len_1)

    def test_predict_2(self):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        x_test_2 = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        self.model_2.fit_eval(x_train_2, y_train_2, **self.config)

        predict_2 = self.model_2.predict(x_test_2)
        assert predict_2.shape == (x_test_2.shape[0], self.future_seq_len_2)

    def test_save_restore_1(self):
        x_train_1, y_train_1 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_1)
        x_test_1 = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        self.model_1.fit_eval(x_train_1, y_train_1, **self.config)

        predict_1_before = self.model_1.predict(x_test_1)
        new_model_1 = LSTMSeq2Seq(check_optional_config=False)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature")
        try:
            save(dirname, model=self.model_1)
            restore(dirname, model=new_model_1, config=self.config)
            predict_1_after = new_model_1.predict(x_test_1)
            assert_array_almost_equal(predict_1_before, predict_1_after, decimal=2), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_1_before,
                                                                       predict_1_after)
            new_config = {'epochs': 1}
            new_model_1.fit_eval(x_train_1, y_train_1, **new_config)
        finally:
            shutil.rmtree(dirname)

    def test_save_restore_2(self):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        x_test_2 = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        self.model_2.fit_eval(x_train_2, y_train_2, **self.config)

        predict_2_before = self.model_2.predict(x_test_2)
        new_model_2 = LSTMSeq2Seq(check_optional_config=False)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature")
        try:
            save(dirname, model=self.model_2)
            restore(dirname, model=new_model_2, config=self.config)
            predict_2_after = new_model_2.predict(x_test_2)
            assert_array_almost_equal(predict_2_before, predict_2_after, decimal=2), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_2_before,
                                                                       predict_2_after)
            new_config = {'epochs': 2}
            new_model_2.fit_eval(x_train_2, y_train_2, **new_config)
        finally:
            shutil.rmtree(dirname)

    def test_predict_with_uncertainty(self,):
        x_train_2, y_train_2 = self.feat._roll_train(self.train_data,
                                                     past_seq_len=self.past_seq_len,
                                                     future_seq_len=self.future_seq_len_2)
        x_test_2 = self.feat._roll_test(self.test_data, past_seq_len=self.past_seq_len)
        self.model_2.fit_eval(x_train_2, y_train_2, mc=True, **self.config)
        prediction, uncertainty = self.model_2.predict_with_uncertainty(x_test_2, n_iter=2)
        assert prediction.shape == (x_test_2.shape[0], self.future_seq_len_2)
        assert uncertainty.shape == (x_test_2.shape[0], self.future_seq_len_2)
        assert np.any(uncertainty)

        new_model_2 = LSTMSeq2Seq(check_optional_config=False)
        dirname = tempfile.mkdtemp(prefix="automl_test_feature")
        try:
            save(dirname, model=self.model_2)
            restore(dirname, model=new_model_2, config=self.config)
            prediction, uncertainty = new_model_2.predict_with_uncertainty(x_test_2, n_iter=2)
            assert prediction.shape == (x_test_2.shape[0], self.future_seq_len_2)
            assert uncertainty.shape == (x_test_2.shape[0], self.future_seq_len_2)
            assert np.any(uncertainty)
        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
