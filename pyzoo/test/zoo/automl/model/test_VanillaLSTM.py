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
from zoo.automl.model.VanillaLSTM import *
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer


class TestVanillaLSTM:

    def test_fit_eval(self):
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        future_seq_len = 1
        past_seq_len = 6

        # use roll method in time_sequence
        tsft = TimeSequenceFeatureTransformer()
        x_train, y_train = tsft._roll_train(train_data,
                                            past_seq_len=past_seq_len,
                                            future_seq_len=future_seq_len)
        config = {
            'epochs': 2,
            "lr": 0.001,
            "lstm_1_units": 16,
            "dropout_1": 0.2,
            "lstm_2_units": 10,
            "dropout_2": 0.2,
            "batch_size": 32,
        }
        model = VanillaLSTM(check_optional_config=False, future_seq_len=future_seq_len)
        print("fit_eval:", model.fit_eval(x_train, y_train, **config))

    def test_evaluate(self):
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        val_data = pd.DataFrame(data=np.random.randn(16, 4))
        future_seq_len = 1
        past_seq_len = 6
        # use roll method in time_sequence
        tsft = TimeSequenceFeatureTransformer()
        x_train, y_train = tsft._roll_train(train_data,
                                            past_seq_len=past_seq_len,
                                            future_seq_len=future_seq_len)
        x_val, y_val = tsft._roll_train(val_data,
                                        past_seq_len=past_seq_len,
                                        future_seq_len=future_seq_len)
        config = {
            'epochs': 1,
            "lr": 0.001,
            "lstm_1_units": 16,
            "dropout_1": 0.2,
            "lstm_2_units": 10,
            "dropout_2": 0.2,
            "batch_size": 32,
        }
        model = VanillaLSTM(check_optional_config=False, future_seq_len=future_seq_len)
        model.fit_eval(x_train, y_train, **config)
        mse, rs = model.evaluate(x_val, y_val, metric=['mean_squared_error', 'r_square'])
        print("Mean squared error is:", mse)
        print("R square is:", rs)

    def test_predict(self):
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        test_data = pd.DataFrame(data=np.random.randn(16, 4))
        future_seq_len = 1
        past_seq_len = 6

        # use roll method in time_sequence
        tsft = TimeSequenceFeatureTransformer()
        x_train, y_train = tsft._roll_train(train_data,
                                            past_seq_len=past_seq_len,
                                            future_seq_len=future_seq_len)
        x_test = tsft._roll_test(test_data, past_seq_len=past_seq_len)

        config = {
            'epochs': 2,
            "lr": 0.001,
            "lstm_1_units": 16,
            "dropout_1": 0.2,
            "lstm_2_units": 10,
            "dropout_2": 0.2,
            "batch_size": 32,
        }
        model = VanillaLSTM(check_optional_config=False, future_seq_len=future_seq_len)
        model.fit_eval(x_train, y_train, **config)
        y_pred = model.predict(x_test)
        assert y_pred.shape == (x_test.shape[0], 1)

    def test_save_restore(self):
        train_data = pd.DataFrame(data=np.random.randn(64, 4))
        test_data = pd.DataFrame(data=np.random.randn(16, 4))
        future_seq_len = 1
        past_seq_len = 6

        # use roll method in time_sequence
        tsft = TimeSequenceFeatureTransformer()
        x_train, y_train = tsft._roll_train(train_data,
                                            past_seq_len=past_seq_len,
                                            future_seq_len=future_seq_len)
        x_test = tsft._roll_test(test_data, past_seq_len=past_seq_len)

        config = {
            'epochs': 2,
            "lr": 0.001,
            "lstm_1_units": 16,
            "dropout_1": 0,
            "lstm_2_units": 10,
            "dropout_2": 0,
            "batch_size": 8,
        }

        model = VanillaLSTM(check_optional_config=False, future_seq_len=future_seq_len)
        new_model = VanillaLSTM(check_optional_config=False)
        model.fit_eval(x_train, y_train, **config)
        predict_before = model.predict(x_test)

        dirname = tempfile.mkdtemp(prefix="automl_test_vanilla")
        try:
            save(dirname, model=model)
            restore(dirname, model=new_model, config=config)
            predict_after = new_model.predict(x_test)
            assert np.allclose(predict_before, predict_after)
            new_config = {'epochs': 2}
            new_model.fit_eval(x_train, y_train, **new_config)

        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
