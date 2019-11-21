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
from zoo.automl.model.MTNet import *
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer


class TestMTNet(ZooTestCase):

    def setup_method(self, method):
        super().setup_method(method)
        self.train_data = pd.DataFrame(data=np.random.randn(128, 4))
        self.val_data = pd.DataFrame(data=np.random.randn(64, 4))
        self.test_data = pd.DataFrame(data=np.random.randn(64, 4))

        # use roll method in time_sequence
        self.feat = TimeSequenceFeatureTransformer()

    def test_future_seq_len_1(self):
        # test fit_eval while future_seq_len = 1
        config = {
            'batch_size': 16,
            'epochs': 1,
            "T": 4,
            "n": 3,
            "W": 2,
            'highway_window': 2,
            'metric': 'mse'
        }
        config['past_seq_len'] = (config['n'] + 1) * config['T']
        future_seq_len = 1
        x_train, y_train = self.feat._roll_train(self.train_data,
                                                 past_seq_len=config['past_seq_len'],
                                                 future_seq_len=future_seq_len)
        x_val, y_val = self.feat._roll_train(self.train_data,
                                             past_seq_len=config['past_seq_len'],
                                             future_seq_len=future_seq_len)
        x_test = self.feat._roll_test(self.test_data, past_seq_len=config['past_seq_len'])

        model = MTNet(future_seq_len=future_seq_len)
        # fit_eval without validation data
        evaluation_result = model.fit_eval(x_train, y_train, **config)
        assert isinstance(evaluation_result, np.float)
        r2, smape = model.evaluate(x_val, y_val, metrics=["r2", "smape"])
        assert isinstance(r2, np.float)
        assert isinstance(smape, np.float)
        predict_result = model.predict(x_test)
        assert predict_result.shape == (x_test.shape[0], future_seq_len)
        new_model = MTNet()
        dirname = tempfile.mkdtemp(prefix="test_mtnet")
        try:
            save(dirname, model=model)
            restore_configs = restore(dirname, model=new_model)
            assert all(item in restore_configs.items() for item in config.items())
            predict_after = new_model.predict(x_test)
            assert np.allclose(predict_result, predict_after), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_result,
                                                                       predict_after)
            new_model.fit_eval(x_train, y_train, validation_data=(x_val, y_val), verbose=1,
                               epochs=10, decay_epochs=8)
        finally:
            shutil.rmtree(dirname)

    def test_future_seq_len_more_than_1(self):
        # test fit_eval while future_seq_len > 1
        config = {
            'batch_size': 16,
            'epochs': 1,
            "T": 4,
            "n": 3,
            "W": 2,
            'highway_window': 2,
            'metric': 'mse'
        }
        config['past_seq_len'] = (config['n'] + 1) * config['T']
        future_seq_len = np.random.randint(1, 10)
        x_train, y_train = self.feat._roll_train(self.train_data,
                                                 past_seq_len=config['past_seq_len'],
                                                 future_seq_len=future_seq_len)
        x_val, y_val = self.feat._roll_train(self.train_data,
                                             past_seq_len=config['past_seq_len'],
                                             future_seq_len=future_seq_len)
        x_test = self.feat._roll_test(self.test_data, past_seq_len=config['past_seq_len'])

        model = MTNet(future_seq_len=future_seq_len)
        # fit_eval without validation data
        evaluation_result = model.fit_eval(x_train, y_train, **config)
        assert isinstance(evaluation_result, np.float)
        r2, smape = model.evaluate(x_val, y_val, metrics=["r2", "smape"])
        assert len(r2) == future_seq_len
        assert len(smape) == future_seq_len
        predict_result = model.predict(x_test)
        assert predict_result.shape == (x_test.shape[0], future_seq_len)
        new_model = MTNet()
        dirname = tempfile.mkdtemp(prefix="test_mtnet")
        try:
            save(dirname, model=model)
            restore_configs = restore(dirname, model=new_model)
            assert all(item in restore_configs.items() for item in config.items())
            predict_after = new_model.predict(x_test)
            assert np.allclose(predict_result, predict_after), \
                "Prediction values are not the same after restore: " \
                "predict before is {}, and predict after is {}".format(predict_result,
                                                                       predict_after)
            print(new_model.lr_value)
            new_model.fit_eval(x_train, y_train, validation_data=(x_val, y_val), verbose=1,
                               epochs=10, decay_epochs=8)
            print(new_model.lr_value)
        finally:
            shutil.rmtree(dirname)

    def test_default_configs(self):
        # test fit_eval while future_seq_len = 1
        config = {
            'batch_size': 16,
            'epochs': 1,
            'T': 1,
            'n': 7,
        }

        config['past_seq_len'] = (config['n'] + 1) * config['T']
        future_seq_len = 1
        x_train, y_train = self.feat._roll_train(self.train_data,
                                                 past_seq_len=config['past_seq_len'],
                                                 future_seq_len=future_seq_len)
        x_val, y_val = self.feat._roll_train(self.train_data,
                                             past_seq_len=config['past_seq_len'],
                                             future_seq_len=future_seq_len)
        x_test = self.feat._roll_test(self.test_data, past_seq_len=config['past_seq_len'])

        model = MTNet(future_seq_len=future_seq_len)
        # fit_eval without validation data
        evaluation_result = model.fit_eval(x_train, y_train, **config)
        assert isinstance(evaluation_result, np.float)
        r2, smape = model.evaluate(x_val, y_val, metrics=["r2", "smape"])
        assert isinstance(r2, np.float)
        assert isinstance(smape, np.float)
        predict_result = model.predict(x_test)
        assert predict_result.shape == (x_test.shape[0], future_seq_len)

    def test_mc_dropout(self):
        # test fit_eval while future_seq_len > 1
        config = {
            'batch_size': 16,
            'epochs': 1,
            "T": 4,
            "n": 3,
            "W": 2,
            'highway_window': 2,
            'metric': 'mse'
        }
        config['past_seq_len'] = (config['n'] + 1) * config['T']
        future_seq_len = np.random.randint(1, 10)
        x_train, y_train = self.feat._roll_train(self.train_data,
                                                 past_seq_len=config['past_seq_len'],
                                                 future_seq_len=future_seq_len)
        x_test = self.feat._roll_test(self.test_data, past_seq_len=config['past_seq_len'])

        model = MTNet(future_seq_len=future_seq_len)
        # fit_eval without validation data
        evaluation_result = model.fit_eval(x_train, y_train, mc=True, **config)
        assert isinstance(evaluation_result, np.float)
        predict_result, pred_uncertainty = model.predict_with_uncertainty(x_test, n_iter=2)
        assert predict_result.shape == (x_test.shape[0], future_seq_len)
        assert np.any(pred_uncertainty)


if __name__ == '__main__':
    pytest.main([__file__])
