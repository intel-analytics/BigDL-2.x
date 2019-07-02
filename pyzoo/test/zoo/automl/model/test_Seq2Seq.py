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
from zoo.automl.model.Seq2Seq import *
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer


class TestSeq2Seq:

    def test_fit_eval(self):
        train_data = pd.DataFrame(data=np.random.randn(64, 4))

        # use roll method in time_sequence
        feat = TimeSequenceFeatureTransformer()
        x_train_1, y_train_1 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=1)
        x_train_2, y_train_2 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=2)

        config = {
            'batch_size': 32,
            'epochs': 1
        }
        model_1 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=1)
        print("fit_eval_future_seq_len_1:", model_1.fit_eval(x_train_1, y_train_1, **config))
        assert model_1.past_seq_len == 6
        assert model_1.feature_num == 4
        assert model_1.future_seq_len == 1
        assert model_1.target_col_num == 1

        model_2 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=2)
        print("fit_eval_future_seq_len_2:", model_2.fit_eval(x_train_2, y_train_2, **config))
        assert model_2.future_seq_len == 2

    def test_evaluate(self):
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        val_data = pd.DataFrame(data=np.random.randn(16, 4))

        # use roll method in time_sequence
        feat = TimeSequenceFeatureTransformer()
        x_train_1, y_train_1 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=1)
        x_train_2, y_train_2 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=2)

        x_val_1, y_val_1 = feat._roll_train(val_data, past_seq_len=6, future_seq_len=1)
        x_val_2, y_val_2 = feat._roll_train(val_data, past_seq_len=6, future_seq_len=2)

        config = {
            'batch_size': 32,
            'epochs': 1
        }

        model_1 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=1)
        model_1.fit_eval(x_train_1, y_train_1, **config)
        print("evaluate_future_seq_len_1:", model_1.evaluate(x_val_1, y_val_1, metric=['mean_squared_error', 'r_square']))

        model_2 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=2)
        model_2.fit_eval(x_train_2, y_train_2, **config)
        print("evaluate_future_seq_len_2:", model_2.evaluate(x_val_2, y_val_2, metric=['mean_squared_error', 'r_square']))

    def test_predict(self):
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        test_data = pd.DataFrame(data=np.random.randn(16, 4))

        # use roll method in time_sequence
        feat = TimeSequenceFeatureTransformer()
        x_train_1, y_train_1 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=1)
        x_train_2, y_train_2 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=2)

        x_test_1 = feat._roll_test(test_data, past_seq_len=6)
        x_test_2 = feat._roll_test(test_data, past_seq_len=6)

        config = {
            'batch_size': 32,
            'epochs': 1
        }

        model_1 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=1)
        model_1.fit_eval(x_train_1, y_train_1, **config)
        predict_1 = model_1.predict(x_test_1)
        assert predict_1.shape == (x_test_1.shape[0], 1)

        model_2 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=2)
        model_2.fit_eval(x_train_2, y_train_2, **config)
        predict_2 = model_2.predict(x_test_2)
        assert predict_2.shape == (x_test_2.shape[0], 2)

    def test_save_restore(self):
        """
        test save and restore function in the cases of different future_seq_lens (1 and 2)
        1. if predict values are the same before and after restore.
        """
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        test_data = pd.DataFrame(data=np.random.randn(16, 4))

        # use roll method in time_sequence
        feat = TimeSequenceFeatureTransformer()
        x_train_1, y_train_1 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=1)
        x_train_2, y_train_2 = feat._roll_train(train_data, past_seq_len=6, future_seq_len=2)

        x_test_1 = feat._roll_test(test_data, past_seq_len=6)
        x_test_2 = feat._roll_test(test_data, past_seq_len=6)

        config = {
            'batch_size': 32,
            'epochs': 2,
            'dropout': 0
        }

        model_1 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=1)
        model_1.fit_eval(x_train_1, y_train_1, **config)
        predict_1_before = model_1.predict(x_test_1)
        new_model_1 = LSTMSeq2Seq(check_optional_config=False)

        model_2 = LSTMSeq2Seq(check_optional_config=False, future_seq_len=2)
        model_2.fit_eval(x_train_2, y_train_2, **config)
        predict_2_before = model_2.predict(x_test_2)
        new_model_2 = LSTMSeq2Seq(check_optional_config=False)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature")

        try:
            save(dirname, model=model_1)
            restore(dirname, model=new_model_1, config=config)
            predict_1_after = new_model_1.predict(x_test_1)
            assert np.allclose(predict_1_before, predict_1_after), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_1_before, predict_1_after)
            new_config = {'epochs': 2}
            new_model_1.fit_eval(x_train_1, y_train_1, **new_config)

            save(dirname, model=model_2)
            restore(dirname, model=new_model_2, config=config)
            predict_2_after = new_model_2.predict(x_test_2)
            assert np.allclose(predict_2_before, predict_2_after), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_2_before, predict_2_after)
            new_model_2.fit_eval(x_train_2, y_train_2, **new_config)
        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
