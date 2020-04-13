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
from zoo.automl.model import BaseModel
from zoo.automl.pipeline.time_sequence import *
from zoo.automl.regression.time_sequence_predictor import *
import numpy as np
import pandas as pd
import ray
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal


default_past_seq_len = 2


class TestTimeSequencePipeline(ZooTestCase):

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        """
        Teardown any state that was previously setup with a setup_method call.
        """
        pass

    def get_input_tsp(self, future_seq_len, target_col):
        sample_num = np.random.randint(100, 200)
        train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019',
                                                           periods=sample_num),
                                 target_col: np.random.randn(sample_num)})
        test_sample_num = np.random.randint(20, 30)
        test_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019',
                                                          periods=test_sample_num),
                                target_col: np.random.randn(test_sample_num)})
        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col=target_col,
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        return train_df, test_df, tsp, test_sample_num

    # def test_evaluate_predict_future_equal_1(self):
    #     target_col = "values"
    #     metrics = ["mse", "r2"]
    #     future_seq_len = 1
    #     train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
    #     pipeline = tsp.fit(train_df, test_df)
    #     mse, rs = pipeline.evaluate(test_df, metrics=metrics)
    #     assert isinstance(mse, np.float)
    #     assert isinstance(rs, np.float)
    #     y_pred = pipeline.predict(test_df)
    #     assert y_pred.shape == (test_sample_num - default_past_seq_len + 1,
    #                             future_seq_len + 1)
    #
    #     y_pred_df = pipeline.predict(test_df[:-future_seq_len])
    #     y_df = test_df[default_past_seq_len:]
    #
    #     mse_pred_eval, rs_pred_eval = [Evaluator.evaluate(m,
    #                                                       y_df[target_col].values,
    #                                                       y_pred_df[target_col].values)
    #                                    for m in metrics]
    #     mse_eval, rs_eval = pipeline.evaluate(test_df, metrics)
    #     assert mse_pred_eval == mse_eval
    #     assert rs_pred_eval == rs_eval

    def test_save_restore_future_equal_1(self):
        target_col = "values"
        metrics = ["mse", "r2"]
        future_seq_len = 1
        train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
        pipeline = tsp.fit(train_df, test_df)
        y_pred = pipeline.predict(test_df)
        mse, rs = pipeline.evaluate(test_df, metrics=metrics)
        print("Evaluation result: Mean square error is: {}, R square is: {}.".format(mse, rs))

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            pipeline.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)
            assert new_pipeline.config is not None
            assert isinstance(new_pipeline.feature_transformers, TimeSequenceFeatureTransformer)
            assert isinstance(new_pipeline.model, TimeSequenceModel)
            new_pipeline.describe()
            new_pred = new_pipeline.predict(test_df)
            assert_array_almost_equal(y_pred[target_col].values, new_pred[target_col].values,
                                      decimal=2)
        finally:
            shutil.rmtree(dirname)

        new_pipeline.fit(train_df, epoch_num=1)
        new_mse, new_rs = new_pipeline.evaluate(test_df,
                                                metrics=["mse", "r2"])
        print("Evaluation result after restore and fit: "
              "Mean square error is: {}, R square is: {}.".format(new_mse, new_rs))

    def test_evaluate_predict_future_more_1(self):
        target_col = "values"
        metrics = ["mse", "r2"]
        future_seq_len = np.random.randint(2, 6)
        train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
        pipeline = tsp.fit(train_df, test_df)
        mse, rs = pipeline.evaluate(test_df, metrics=metrics)
        assert len(mse) == future_seq_len
        assert len(rs) == future_seq_len
        y_pred = pipeline.predict(test_df)
        assert y_pred.shape == (test_sample_num - default_past_seq_len + 1,
                                future_seq_len + 1)

        y_pred_df = pipeline.predict(test_df[:-future_seq_len])
        columns = ["{}_{}".format(target_col, i) for i in range(future_seq_len)]
        y_pred_value = y_pred_df[columns].values

        y_df = test_df[default_past_seq_len:]
        y_value = TimeSequenceFeatureTransformer()._roll_test(y_df[target_col], future_seq_len)

        mse_pred_eval, rs_pred_eval = [Evaluator.evaluate(m, y_value, y_pred_value)
                                       for m in metrics]
        mse_eval, rs_eval = pipeline.evaluate(test_df, metrics)
        assert_array_almost_equal(mse_pred_eval, mse_eval, decimal=2)
        assert_array_almost_equal(rs_pred_eval, rs_eval, decimal=2)

    # def test_save_restore_future_more_1(self):
    #     target_col = "values"
    #     metrics = ["mse", "r2"]
    #     future_seq_len = np.random.randint(2, 6)
    #     train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
    #     pipeline = tsp.fit(train_df, test_df)
    #     y_pred = pipeline.predict(test_df)
    #     mse, rs = pipeline.evaluate(test_df, metrics=metrics)
    #     print("Evaluation result: Mean square error is: {}, R square is: {}.".format(mse, rs))
    #
    #     dirname = tempfile.mkdtemp(prefix="saved_pipeline")
    #     try:
    #         save_pipeline_file = os.path.join(dirname, "my.ppl")
    #         pipeline.save(save_pipeline_file)
    #         assert os.path.isfile(save_pipeline_file)
    #         new_pipeline = load_ts_pipeline(save_pipeline_file)
    #
    #         new_pred = new_pipeline.predict(test_df)
    #         columns = ["{}_{}".format(target_col, i) for i in range(future_seq_len)]
    #         assert_array_almost_equal(y_pred[columns].values, new_pred[columns].values, decimal=2)
    #
    #     finally:
    #         shutil.rmtree(dirname)
    #
    #     new_pipeline.fit(train_df, epoch_num=1)
    #     new_mse, new_rs = new_pipeline.evaluate(test_df,
    #                                             metrics=["mse", "r2"])
    #     print("Evaluation result after restore and fit: "
    #           "Mean square error is: {}, R square is: {}.".format(new_mse, new_rs))

    def test_predict_df_list(self):
        target_col = "values"
        future_seq_len = np.random.randint(2, 6)
        train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
        df_num = 2
        train_df_list = [train_df] * df_num
        test_df_list = [test_df] * df_num
        pipeline = tsp.fit(train_df_list, test_df_list)
        y_pred = pipeline.predict(test_df_list)
        assert len(y_pred) == df_num
        assert_frame_equal(y_pred[0], y_pred[1], check_exact=False, check_less_precise=2)
        assert y_pred[0].shape == (test_sample_num - default_past_seq_len + 1,
                                   future_seq_len + 1)

    # def test_look_back_future_1(self):
    #     target_col = "values"
    #     min_past_seq_len = np.random.randint(3, 5)
    #     max_past_seq_len = np.random.randint(5, 8)
    #     future_seq_len = 1
    #     train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
    #
    #     random_pipeline = tsp.fit(train_df, test_df, recipe=RandomRecipe(
    #         look_back=(min_past_seq_len, max_past_seq_len)))
    #     y_pred_random = random_pipeline.predict(test_df)
    #     assert y_pred_random.shape[0] >= test_sample_num - max_past_seq_len + 1
    #     assert y_pred_random.shape[0] <= test_sample_num - min_past_seq_len + 1
    #     assert y_pred_random.shape[1] == future_seq_len + 1
    #     mse, rs = random_pipeline.evaluate(test_df, metrics=["mse", "r2"])
    #     assert isinstance(mse, np.float)
    #     assert isinstance(rs, np.float)
    #
    # def test_look_back_future_more_1(self):
    #     target_col = "values"
    #     min_past_seq_len = np.random.randint(3, 5)
    #     max_past_seq_len = np.random.randint(5, 8)
    #     future_seq_len = np.random.randint(2, 6)
    #     train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
    #
    #     random_pipeline = tsp.fit(train_df, test_df, recipe=RandomRecipe(
    #         look_back=(min_past_seq_len, max_past_seq_len)))
    #     y_pred_random = random_pipeline.predict(test_df)
    #     assert y_pred_random.shape[0] >= test_sample_num - max_past_seq_len + 1
    #     assert y_pred_random.shape[0] <= test_sample_num - min_past_seq_len + 1
    #     assert y_pred_random.shape[1] == future_seq_len + 1
    #     mse, rs = random_pipeline.evaluate(test_df, metrics=["mse", "r2"])
    #     assert len(mse) == future_seq_len
    #     assert len(rs) == future_seq_len

    def test_look_back_value(self):
        target_col = "values"
        future_seq_len = np.random.randint(2, 6)
        train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
        # test min_past_seq_len < 2
        tsp.fit(train_df, test_df, recipe=RandomRecipe(look_back=(1, 2)))
        # test max_past_seq_len < 2
        with pytest.raises(ValueError, match=r".*max look back value*."):
            tsp.fit(train_df, test_df, recipe=RandomRecipe(look_back=(0, 1)))
        # test look_back value < 2
        with pytest.raises(ValueError, match=r".*look back value should not be smaller than 2*."):
            tsp.fit(train_df, test_df, recipe=RandomRecipe(look_back=1))

        # test look back is None
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            tsp.fit(train_df, test_df, recipe=RandomRecipe(look_back=None))
        # test look back is str
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            tsp.fit(train_df, test_df, recipe=RandomRecipe(look_back="a"))
        # test look back is float
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            tsp.fit(train_df, test_df, recipe=RandomRecipe(look_back=2.5))
        # test look back range is float
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            tsp.fit(train_df, test_df, recipe=RandomRecipe(look_back=(2.5, 3)))

    def test_get_default_configs(self):
        ppl = TimeSequencePipeline(name='test')
        ppl.get_default_configs()

    # def test_fit_with_fixed_configs_save(self):
    #     target_col = "values"
    #     future_seq_len = np.random.randint(2, 6)
    #     train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
    #     ppl = TimeSequencePipeline(name='test')
    #     ppl.fit_with_fixed_configs(train_df, test_df, target_col=target_col,
    #                                batch_size=64, epochs=1, future_seq_len=future_seq_len)
    #     mse, rs = ppl.evaluate(test_df, metrics=["mse", "r2"])
    #     assert len(mse) == future_seq_len
    #     assert len(rs) == future_seq_len
    #     y_pred = ppl.predict(test_df)
    #     ppl_file = ppl.save()
    #     config_file = ppl.config_save()
    #     assert os.path.isfile(config_file)
    #     reload_configs = load_config(config_file)
    #     reload_ppl = load_ts_pipeline(ppl_file)
    #     os.remove(ppl_file)
    #     os.remove(config_file)
    #     os.rmdir(os.path.dirname(os.path.abspath(ppl_file)))
    #     reload_y_pred = reload_ppl.predict(test_df)
    #     assert reload_configs["batch_size"] == 64
    #     assert_frame_equal(y_pred, reload_y_pred, check_exact=False, check_less_precise=5)

    # def test_predict_with_uncertainty(self):
    #     target_col = "values"
    #     future_seq_len = np.random.randint(2, 6)
    #     train_df, test_df, tsp, test_sample_num = self.get_input_tsp(future_seq_len, target_col)
    #
    #     # test future_seq_len = 3
    #     pipeline = tsp.fit(train_df, mc=True, validation_df=test_df)
    #     y_out, y_pred_uncertainty = pipeline.predict_with_uncertainty(test_df, n_iter=2)
    #     assert y_out.shape == (test_sample_num - default_past_seq_len + 1,
    #                            future_seq_len + 1)
    #     assert y_pred_uncertainty.shape == (test_sample_num - default_past_seq_len + 1,
    #                                         future_seq_len)
    #     assert np.any(y_pred_uncertainty)
    #
    # def test_fit_predict_with_uncertainty(self):
    #     # test future_seq_len = 1
    #     self.pipeline_1 = self.tsp_1.fit(self.train_df, mc=True, validation_df=self.validation_df)
    #     self.pipeline_1.fit(self.validation_df, mc=True, epoch_num=1)
    #     y_out, y_pred_uncertainty = self.pipeline_1.predict_with_uncertainty(self.test_df,
    #                                                                          n_iter=2)
    #     assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                            self.future_seq_len_1 + 1)
    #     assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                                         self.future_seq_len_1)
    #     assert np.any(y_pred_uncertainty)
    #
    #     # test future_seq_len = 3
    #     self.pipeline_3 = self.tsp_3.fit(self.train_df, mc=True, validation_df=self.validation_df)
    #     self.pipeline_3.fit(self.validation_df, mc=True, epoch_num=1)
    #     y_out, y_pred_uncertainty = self.pipeline_3.predict_with_uncertainty(self.test_df,
    #                                                                          n_iter=2)
    #     assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                            self.future_seq_len_3 + 1)
    #     assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                                         self.future_seq_len_3)
    #     assert np.any(y_pred_uncertainty)
    #
    # def test_fit_fixed_configs_predict_with_uncertainty(self):
    #     # test future_seq_len = 1
    #     self.pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df)
    #     config_file = self.pipeline_1.config_save()
    #     assert os.path.isfile(config_file)
    #     configs = load_config(config_file)
    #     os.remove(config_file)
    #     os.rmdir(os.path.dirname(os.path.abspath(config_file)))
    #     ppl = TimeSequencePipeline(name='test', config=configs)
    #     ppl.fit_with_fixed_configs(self.train_df, self.validation_df, mc=True)
    #     y_out, y_pred_uncertainty = ppl.predict_with_uncertainty(self.test_df, n_iter=2)
    #     assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                            self.future_seq_len_1 + 1)
    #     assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                                         self.future_seq_len_1)
    #     assert np.any(y_pred_uncertainty)
    #
    #     # test future_seq_len = 3
    #     self.pipeline_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df)
    #     config_file = self.pipeline_3.config_save()
    #     assert os.path.isfile(config_file)
    #     configs = load_config(config_file)
    #     os.remove(config_file)
    #     os.rmdir(os.path.dirname(os.path.abspath(config_file)))
    #     ppl = TimeSequencePipeline(name='test', config=configs)
    #     ppl.fit_with_fixed_configs(self.train_df, self.validation_df, mc=True)
    #     y_out, y_pred_uncertainty = ppl.predict_with_uncertainty(self.test_df, n_iter=2)
    #     assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                            self.future_seq_len_3 + 1)
    #     assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
    #                                         self.future_seq_len_3)
    #     assert np.any(y_pred_uncertainty)


if __name__ == '__main__':
    pytest.main([__file__])
