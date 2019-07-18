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


class TestSeq2Seq(ZooTestCase):

    def setup_method(self, method):
        super().setup_method(method)
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        val_data = pd.DataFrame(data=np.random.randn(16, 4))
        test_data = pd.DataFrame(data=np.random.randn(16, 4))

        past_seq_len = 6
        future_seq_len_1 = 1
        future_seq_len_2 = 2

        # use roll method in time_sequence
        feat = TimeSequenceFeatureTransformer()
        self.x_train_1, self.y_train_1 = feat._roll_train(train_data, past_seq_len=past_seq_len,
                                                          future_seq_len=future_seq_len_1)
        self.x_train_2, self.y_train_2 = feat._roll_train(train_data, past_seq_len=past_seq_len,
                                                          future_seq_len=future_seq_len_2)
        self.x_val_1, self.y_val_1 = feat._roll_train(val_data, past_seq_len=past_seq_len,
                                                      future_seq_len=future_seq_len_1)
        self.x_val_2, self.y_val_2 = feat._roll_train(val_data, past_seq_len=past_seq_len,
                                                      future_seq_len=future_seq_len_2)

        self.x_test_1 = feat._roll_test(test_data, past_seq_len=past_seq_len)
        self.x_test_2 = feat._roll_test(test_data, past_seq_len=past_seq_len)

        self.config = {
            'batch_size': 32,
            'epochs': 1
        }

        self.model_1 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=future_seq_len_1)
        self.model_2 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=future_seq_len_2)

        self.fitted = False
        self.predict_1 = None
        self.predict_2 = None

    def test_fit_eval(self):
        print("fit_eval_future_seq_len_1:",
              self.model_1.fit_eval(self.x_train_1, self.y_train_1, **self.config))
        assert self.model_1.past_seq_len == 6
        assert self.model_1.feature_num == 4
        assert self.model_1.future_seq_len == 1
        assert self.model_1.target_col_num == 1

        print("fit_eval_future_seq_len_2:",
              self.model_2.fit_eval(self.x_train_2, self.y_train_2, **self.config))
        assert self.model_2.future_seq_len == 2

        self.fitted = True

    def test_evaluate(self):
        if not self.fitted:
            self.model_1.fit_eval(self.x_train_1, self.y_train_1, **self.config)
            self.model_2.fit_eval(self.x_train_2, self.y_train_2, **self.config)

        print("evaluate_future_seq_len_1:", self.model_1.evaluate(self.x_val_1,
                                                                  self.y_val_1,
                                                                  metric=['mean_squared_error',
                                                                          'r_square']))
        print("evaluate_future_seq_len_2:", self.model_2.evaluate(self.x_val_2,
                                                                  self.y_val_2,
                                                                  metric=['mean_squared_error',
                                                                          'r_square']))

    def test_predict(self):
        if not self.fitted:
            self.model_1.fit_eval(self.x_train_1, self.y_train_1, **self.config)
            self.model_2.fit_eval(self.x_train_2, self.y_train_2, **self.config)
        self.predict_1 = self.model_1.predict(self.x_test_1)
        assert self.predict_1.shape == (self.x_test_1.shape[0], 1)

        self.predict_2 = self.model_2.predict(self.x_test_2)
        assert self.predict_2.shape == (self.x_test_2.shape[0], 2)

    def test_save_restore(self):
        """
        test save and restore function in the cases of different future_seq_lens (1 and 2)
        1. if predict values are the same before and after restore.
        """

        if self.predict_1:
            predict_1_before = self.predict_1
            predict_2_before = self.predict_2
        else:
            self.model_1.fit_eval(self.x_train_1, self.y_train_1, **self.config)
            predict_1_before = self.model_1.predict(self.x_test_1)
            self.model_2.fit_eval(self.x_train_2, self.y_train_2, **self.config)
            predict_2_before = self.model_2.predict(self.x_test_2)

        new_model_1 = LSTMSeq2Seq(check_optional_config=False)
        new_model_2 = LSTMSeq2Seq(check_optional_config=False)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature")

        try:
            save(dirname, model=self.model_1)
            restore(dirname, model=new_model_1, config=self.config)
            predict_1_after = new_model_1.predict(self.x_test_1)
            assert np.allclose(predict_1_before, predict_1_after), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_1_before,
                                                                       predict_1_after)
            new_config = {'epochs': 1}
            new_model_1.fit_eval(self.x_train_1, self.y_train_1, **new_config)

            save(dirname, model=self.model_2)
            restore(dirname, model=new_model_2, config=self.config)
            predict_2_after = new_model_2.predict(self.x_test_2)
            assert np.allclose(predict_2_before, predict_2_after), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_2_before,
                                                                       predict_2_after)
            new_model_2.fit_eval(self.x_train_2, self.y_train_2, **new_config)
        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
