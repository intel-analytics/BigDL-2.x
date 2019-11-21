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


class TestTimeSequencePipeline(ZooTestCase):

    def setup_method(self, method):
        super().setup_method(method)
        ray.init()

        # sample_num should > past_seq_len, the default value of which is 1
        sample_num = 500
        self.train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2000',
                                                                periods=sample_num),
                                     "value": np.random.randn(sample_num)})
        val_sample_num = 64
        self.validation_df = pd.DataFrame({"datetime": pd.date_range('1/2/2000',
                                                                     periods=val_sample_num),
                                           "value": np.random.randn(val_sample_num)})
        self.test_sample_num = 64
        self.test_df = pd.DataFrame({"datetime": pd.date_range('1/3/2000',
                                                               periods=self.test_sample_num),
                                     "value": np.random.randn(self.test_sample_num)})

        self.future_seq_len_1 = 1
        self.tsp_1 = TimeSequencePredictor(dt_col="datetime",
                                           target_col="value",
                                           future_seq_len=self.future_seq_len_1,
                                           extra_features_col=None, )

        self.future_seq_len_3 = 3
        self.tsp_3 = TimeSequencePredictor(dt_col="datetime",
                                           target_col="value",
                                           future_seq_len=self.future_seq_len_3,
                                           extra_features_col=None, )
        self.default_past_seq_len = 2

    def teardown_method(self, method):
        """
        Teardown any state that was previously setup with a setup_method call.
        """
        super().teardown_method(method)
        ray.shutdown()

    def test_evaluate_1(self):
        self.pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df)
        mse, rs = self.pipeline_1.evaluate(self.test_df, metrics=["mse", "r2"])
        assert isinstance(mse, np.float)
        assert isinstance(rs, np.float)
        print("Mean square error (future_seq_len=1) is:", mse)
        print("R square (future_seq_len=1) is:", rs)

    def test_evaluate_1_df_list(self):
        train_df_list = [self.train_df] * 3
        val_df_list = [self.validation_df] * 3
        test_df_list = [self.test_df] * 3
        self.pipeline_1 = self.tsp_1.fit(train_df_list, validation_df=val_df_list)
        mse, rs = self.pipeline_1.evaluate(test_df_list, metrics=["mse", "r2"])
        assert isinstance(mse, np.float)
        assert isinstance(rs, np.float)
        print("Mean square error (future_seq_len=1) is:", mse)
        print("R square (future_seq_len=1) is:", rs)

    def test_evaluate_3(self):
        self.pipeline_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df)
        mse, rs = self.pipeline_3.evaluate(self.test_df, metrics=["mse", "r2"])
        assert len(mse) == self.future_seq_len_3
        assert len(rs) == self.future_seq_len_3
        print("Mean square error (future_seq_len=3) is:", mse)
        print("R square (future_seq_len=3) is:", rs)

    def test_evaluate_3_df_list(self):
        train_df_list = [self.train_df] * 3
        val_df_list = [self.validation_df] * 3
        test_df_list = [self.test_df] * 3
        self.pipeline_3 = self.tsp_3.fit(train_df_list, validation_df=val_df_list)
        mse, rs = self.pipeline_3.evaluate(test_df_list, metrics=["mse", "r2"])
        assert len(mse) == self.future_seq_len_3
        assert len(rs) == self.future_seq_len_3
        print("Mean square error (future_seq_len=3) is:", mse)
        print("R square (future_seq_len=3) is:", rs)

    def test_evaluate_RandomRecipe_1(self):
        self.random_pipeline_1 = self.tsp_1.fit(self.train_df,
                                                validation_df=self.validation_df,
                                                recipe=RandomRecipe(1))
        mse, rs, smape = self.random_pipeline_1.evaluate(self.test_df,
                                                         metrics=["mse",
                                                                  "r2", "smape"])
        assert all(100 > i > 0 for i in smape)
        assert isinstance(mse, np.float)
        assert isinstance(rs, np.float)
        assert isinstance(smape, np.float)
        assert 100 > smape > 0
        
        print("Mean square error (future_seq_len=1) is:", mse)
        print("R square (future_seq_len=1) is:", rs)
        print("sMAPE (future_seq_len=1) is:", smape)

        mse, rs, smape = self.random_pipeline_1.evaluate(self.test_df,
                                                         metrics=["mse",
                                                                  "r2", "smape"],
                                                         multioutput='uniform_average')
        assert isinstance(mse, np.float)
        assert isinstance(rs, np.float)
        assert isinstance(smape, np.float)
        assert 0 < smape < 100

    def test_evaluate_RandomRecipe_3(self):
        self.random_pipeline_3 = self.tsp_3.fit(self.train_df,
                                                validation_df=self.validation_df,
                                                recipe=RandomRecipe(1))
        mse, rs, smape = self.random_pipeline_3.evaluate(self.test_df,
                                                         metrics=["mse",
                                                                  "r2", "smape"])
        assert len(mse) == self.future_seq_len_3
        assert len(rs) == self.future_seq_len_3
        assert len(smape) == self.future_seq_len_3
        assert all(100 > i > 0 for i in smape)

        print("Mean square error (future_seq_len=3) is:", mse)
        print("R square (future_seq_len=3) is:", rs)
        print("sMAPE (future_seq_len=1) is:", smape)

        mse, rs, smape = self.random_pipeline_3.evaluate(self.test_df,
                                                         metrics=["mse",
                                                                  "r2", "smape"],
                                                         multioutput='uniform_average')
        assert isinstance(mse, np.float)
        assert isinstance(rs, np.float)
        assert isinstance(smape, np.float)
        assert 0 < smape < 100

    def test_predict_1(self):
        self.pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df,)
        # sample_num should > past_seq_len, the default value of which is 50
        y_pred_1 = self.pipeline_1.predict(self.test_df)
        assert y_pred_1.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                  self.future_seq_len_1 + 1)

    def test_predict_1_df_list(self):
        train_df_list = [self.train_df] * 3
        val_df_list = [self.validation_df] * 3
        test_df_list = [self.test_df] * 3
        self.pipeline_1 = self.tsp_1.fit(train_df_list, validation_df=val_df_list,)
        # sample_num should > past_seq_len, the default value of which is 50
        y_pred_1 = self.pipeline_1.predict(test_df_list)
        assert len(y_pred_1) == 3
        assert_frame_equal(y_pred_1[0], y_pred_1[1])
        assert_frame_equal(y_pred_1[1], y_pred_1[2])
        assert y_pred_1[0].shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                     self.future_seq_len_1 + 1)

    def test_predict_3(self):
        self.pipeline_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df)
        y_pred_3 = self.pipeline_3.predict(self.test_df)
        assert y_pred_3.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                  self.future_seq_len_3 + 1)

    def test_predict_3_df_list(self):
        train_df_list = [self.train_df] * 3
        val_df_list = [self.validation_df] * 3
        test_df_list = [self.test_df] * 3
        self.pipeline_3 = self.tsp_3.fit(train_df_list, validation_df=val_df_list)
        y_pred_3 = self.pipeline_3.predict(test_df_list)
        assert len(y_pred_3) == 3
        assert_frame_equal(y_pred_3[0], y_pred_3[1])
        assert_frame_equal(y_pred_3[1], y_pred_3[2])
        assert y_pred_3[0].shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                     self.future_seq_len_3 + 1)

    def test_predict_RandomRecipe(self):
        self.random_pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                                                recipe=RandomRecipe(1))
        y_pred_random = self.random_pipeline_1.predict(self.test_df)
        assert y_pred_random.shape == (self.test_sample_num - self.default_past_seq_len + 1, 2)

    def test_predict_BayesRecipe(self):
        self.bayes_pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                                               recipe=BayesRecipe(1))
        y_pred_bayes = self.bayes_pipeline_1.predict(self.test_df)
        assert y_pred_bayes.shape == (self.test_sample_num - self.default_past_seq_len + 1, 2)

    def test_evaluate_predict_1(self):
        metric = ["mse", "r2"]
        target_col = "value"
        self.pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df)
        y_pred_df_1 = self.pipeline_1.predict(self.test_df[:-self.future_seq_len_1])
        y_df_1 = self.test_df[self.default_past_seq_len:]

        mse_pred_eval_1, rs_pred_eval_1 = [Evaluator.evaluate(m,
                                                              y_df_1[target_col].values,
                                                              y_pred_df_1[target_col].values)
                                           for m in metric]
        mse_eval_1, rs_eval_1 = self.pipeline_1.evaluate(self.test_df, metric)
        assert mse_pred_eval_1 == mse_eval_1
        assert rs_pred_eval_1 == rs_eval_1

    def test_evaluate_predict_df_list(self):
        metric = ["mse", "r2"]
        target_col = "value"
        train_df_list = [self.train_df] * 3
        val_df_list = [self.validation_df] * 3
        test_df_list = [self.test_df] * 3
        pred_test_df_list = [self.test_df[:-self.future_seq_len_1]] * 3
        target_test_value = self.test_df[self.default_past_seq_len:][target_col].values
        self.pipeline_1 = self.tsp_1.fit(train_df_list, validation_df=val_df_list)
        y_pred_df_list = self.pipeline_1.predict(pred_test_df_list)
        y_pred = np.concatenate([df[target_col].values for df in y_pred_df_list])
        y_target = np.concatenate([target_test_value] * 3)

        print(y_pred.shape)
        print(y_target.shape)
        mse_pred_eval_1, rs_pred_eval_1 = [Evaluator.evaluate(m,
                                                              y_target,
                                                              y_pred)
                                           for m in metric]
        mse_eval_1, rs_eval_1 = self.pipeline_1.evaluate(test_df_list, metric)
        assert mse_pred_eval_1 == mse_eval_1
        assert rs_pred_eval_1 == rs_eval_1

    def test_evaluate_predict_3(self):
        target_col = "value"
        metric = ["mse", "r2"]
        self.pipeline_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df)
        y_pred_df = self.pipeline_3.predict(self.test_df[:-self.future_seq_len_3])
        columns = ["{}_{}".format(target_col, i) for i in range(self.future_seq_len_3)]
        y_pred_value = y_pred_df[columns].values

        y_df_3 = self.test_df[self.default_past_seq_len:]
        y_value = TimeSequenceFeatureTransformer()._roll_test(y_df_3[target_col],
                                                              self.future_seq_len_3)

        mse_pred_eval_3, rs_pred_eval_3 = [Evaluator.evaluate(m, y_value, y_pred_value)
                                           for m in metric]
        mse_eval_3, rs_eval_3 = self.pipeline_3.evaluate(self.test_df, metric)
        assert np.array_equal(mse_pred_eval_3, mse_eval_3)
        assert np.array_equal(rs_pred_eval_3, rs_eval_3)

    def test_save_restore_1(self):
        self.pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df)
        y_pred_1 = self.pipeline_1.predict(self.test_df)
        mse, rs = self.pipeline_1.evaluate(self.test_df,
                                           metrics=["mse", "r2"])
        print("Evaluation result: Mean square error is: {}, R square is: {}.".format(mse, rs))

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            self.pipeline_1.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(self.test_df)
            np.testing.assert_allclose(y_pred_1["value"].values, new_pred["value"].values)
        finally:
            shutil.rmtree(dirname)

        new_pipeline.fit(self.train_df, epoch_num=1)
        new_mse, new_rs = new_pipeline.evaluate(self.test_df,
                                                metrics=["mse", "r2"])
        print("Evaluation result after restore and fit: "
              "Mean square error is: {}, R square is: {}.".format(new_mse, new_rs))

    def test_save_restore_3(self):
        target_col = "value"
        self.pipeline_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df)
        y_pred_3 = self.pipeline_3.predict(self.test_df)
        mse, rs = self.pipeline_3.evaluate(self.test_df,
                                           metrics=["mse", "r2"])
        print("Evaluation result: Mean square error is: {}, R square is: {}.".format(mse, rs))

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            self.pipeline_3.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(self.test_df)
            print(y_pred_3)
            print(new_pred)
            columns = ["{}_{}".format(target_col, i) for i in range(self.future_seq_len_3)]
            np.testing.assert_allclose(y_pred_3[columns].values, new_pred[columns].values)
        finally:
            shutil.rmtree(dirname)

        new_pipeline.fit(self.train_df, epoch_num=10)
        new_mse, new_rs = new_pipeline.evaluate(self.test_df,
                                                metrics=["mse", "r2"])
        print("Evaluation result after restore and fit: "
              "Mean square error is: {}, R square is: {}.".format(new_mse, new_rs))

    def test_save_restore_1_RandomRecipe(self):
        self.random_pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                                                recipe=RandomRecipe(1))
        y_pred_random = self.random_pipeline_1.predict(self.test_df)

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            self.random_pipeline_1.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(self.test_df)
            np.testing.assert_allclose(y_pred_random["value"].values, new_pred["value"].values)
        finally:
            shutil.rmtree(dirname)

    def test_save_restore_1_bayesRecipe(self):
        self.bayes_pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                                               recipe=BayesRecipe(1))
        y_pred_bayes = self.bayes_pipeline_1.predict(self.test_df)

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            self.bayes_pipeline_1.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(self.test_df)
            np.testing.assert_allclose(y_pred_bayes["value"].values, new_pred["value"].values)
        finally:
            shutil.rmtree(dirname)

    def test_look_back_1(self):
        min_past_seq_len = 5
        max_past_seq_len = 7
        print("=" * 10, "fit in test_look_back_1", "=" * 10)

        random_pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                                           recipe=RandomRecipe(
                                               look_back=(min_past_seq_len, max_past_seq_len)))
        y_pred_random_1 = random_pipeline_1.predict(self.test_df)
        assert y_pred_random_1.shape[0] >= self.test_sample_num - max_past_seq_len + 1
        assert y_pred_random_1.shape[0] <= self.test_sample_num - min_past_seq_len + 1
        assert y_pred_random_1.shape[1] == self.future_seq_len_1 + 1
        mse, rs = random_pipeline_1.evaluate(self.test_df, metrics=["mse", "r2"])
        assert isinstance(mse, np.float)
        assert isinstance(rs, np.float)

    def test_look_back_3(self):
        min_past_seq_len = 5
        max_past_seq_len = 7

        random_pipeline_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                                           recipe=RandomRecipe(
                                               look_back=(min_past_seq_len, max_past_seq_len)))
        y_pred_random_3 = random_pipeline_3.predict(self.test_df)
        assert y_pred_random_3.shape[0] >= self.test_sample_num - max_past_seq_len + 1
        assert y_pred_random_3.shape[0] <= self.test_sample_num - min_past_seq_len + 1
        assert y_pred_random_3.shape[1] == self.future_seq_len_3 + 1
        mse, rs = random_pipeline_3.evaluate(self.test_df,
                                             metrics=["mse", "r2"])
        assert len(mse) == self.future_seq_len_3
        assert len(rs) == self.future_seq_len_3

    def test_look_back_value(self):
        # test min_past_seq_len < 2
        self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                       recipe=RandomRecipe(look_back=(1, 2)))
        # test max_past_seq_len < 2
        with pytest.raises(ValueError, match=r".*max look back value*."):
            self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                           recipe=RandomRecipe(look_back=(0, 1)))
        # test look_back value < 2
        with pytest.raises(ValueError, match=r".*look back value should not be smaller than 2*."):
            self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                           recipe=RandomRecipe(look_back=1))

        # test look back is None
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                           recipe=RandomRecipe(look_back=None))
        # test look back is str
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                           recipe=RandomRecipe(look_back="a"))
        # test look back is float
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                           recipe=RandomRecipe(look_back=2.5))
        # test look back range is float
        with pytest.raises(ValueError, match=r".*look_back should be either*."):
            self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                           recipe=RandomRecipe(look_back=(2.5, 3)))

    def test_look_back_value_bayes_recipe(self):
        # test min_past_seq_len < 2
        self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                       recipe=BayesRecipe(look_back=(1, 2)))
        # test max_past_seq_len < 2
        with pytest.raises(ValueError, match=r".*max look back value*."):
            self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                           recipe=BayesRecipe(look_back=(0, 1)))
        # test look_back value < 2
        with pytest.raises(ValueError, match=r".*look back value should not be smaller than 2*."):
            self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                           recipe=BayesRecipe(look_back=1))

    def test_get_default_configs(self):
        ppl = TimeSequencePipeline(name='test')
        ppl.get_default_configs()

    def test_fit_with_fixed_configs(self):
        future_seq_len = 2
        ppl = TimeSequencePipeline(name='test')
        ppl.fit_with_fixed_configs(self.train_df, self.validation_df,
                                   batch_size=16, epochs=1, future_seq_len=future_seq_len)
        mse, rs = ppl.evaluate(self.test_df, metrics=["mse", "r2"])
        assert len(mse) == future_seq_len
        assert len(rs) == future_seq_len
        y_pred = ppl.predict(self.test_df)
        ppl_file = ppl.save()
        reload_ppl = load_ts_pipeline(ppl_file)
        os.remove(ppl_file)
        os.rmdir(os.path.dirname(os.path.abspath(ppl_file)))
        reload_y_pred = reload_ppl.predict(self.test_df)
        assert_frame_equal(y_pred, reload_y_pred)

    def test_save_configs(self):
        self.pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df)
        mse, rs = self.pipeline_1.evaluate(self.test_df, metrics=["mse", "r2"])
        print(mse, rs)
        config_file = self.pipeline_1.config_save()
        assert os.path.isfile(config_file)
        configs = load_config(config_file)
        os.remove(config_file)
        os.rmdir(os.path.dirname(os.path.abspath(config_file)))
        ppl = TimeSequencePipeline(name='test', config=configs)
        ppl.fit_with_fixed_configs(self.train_df, self.validation_df)
        mse, rs = ppl.evaluate(self.test_df, metrics=["mse", "r2"])
        print(mse, rs)

    def test_load_ts_pipeline_describe(self):
        pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df)
        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            pipeline_1.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)
            assert new_pipeline.config is not None
            assert isinstance(new_pipeline.feature_transformers, TimeSequenceFeatureTransformer)
            assert isinstance(new_pipeline.model, TimeSequenceModel)
            new_pipeline.describe()
        finally:
            shutil.rmtree(dirname)

    def test_predict_with_uncertainty(self):
        # test future_seq_len = 1
        self.pipeline_1 = self.tsp_1.fit(self.train_df, mc=True, validation_df=self.validation_df)
        y_out, y_pred_uncertainty = self.pipeline_1.predict_with_uncertainty(self.test_df,
                                                                             n_iter=2)
        assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                               self.future_seq_len_1 + 1)
        assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                            self.future_seq_len_1)
        assert np.any(y_pred_uncertainty)

        # test future_seq_len = 3
        self.pipeline_3 = self.tsp_3.fit(self.train_df, mc=True, validation_df=self.validation_df)
        y_out, y_pred_uncertainty = self.pipeline_3.predict_with_uncertainty(self.test_df,
                                                                             n_iter=2)
        assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                               self.future_seq_len_3 + 1)
        assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                            self.future_seq_len_3)
        assert np.any(y_pred_uncertainty)

    def test_fit_predict_with_uncertainty(self):
        # test future_seq_len = 1
        self.pipeline_1 = self.tsp_1.fit(self.train_df, mc=True, validation_df=self.validation_df)
        self.pipeline_1.fit(self.validation_df, mc=True, epoch_num=1)
        y_out, y_pred_uncertainty = self.pipeline_1.predict_with_uncertainty(self.test_df,
                                                                             n_iter=2)
        assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                               self.future_seq_len_1 + 1)
        assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                            self.future_seq_len_1)
        assert np.any(y_pred_uncertainty)

        # test future_seq_len = 3
        self.pipeline_3 = self.tsp_3.fit(self.train_df, mc=True, validation_df=self.validation_df)
        self.pipeline_3.fit(self.validation_df, mc=True, epoch_num=1)
        y_out, y_pred_uncertainty = self.pipeline_3.predict_with_uncertainty(self.test_df,
                                                                             n_iter=2)
        assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                               self.future_seq_len_3 + 1)
        assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                            self.future_seq_len_3)
        assert np.any(y_pred_uncertainty)

    def test_fit_fixed_configs_predict_with_uncertainty(self):
        # test future_seq_len = 1
        self.pipeline_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df)
        config_file = self.pipeline_1.config_save()
        assert os.path.isfile(config_file)
        configs = load_config(config_file)
        os.remove(config_file)
        os.rmdir(os.path.dirname(os.path.abspath(config_file)))
        ppl = TimeSequencePipeline(name='test', config=configs)
        ppl.fit_with_fixed_configs(self.train_df, self.validation_df, mc=True)
        y_out, y_pred_uncertainty = ppl.predict_with_uncertainty(self.test_df, n_iter=2)
        assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                               self.future_seq_len_1 + 1)
        assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                            self.future_seq_len_1)
        assert np.any(y_pred_uncertainty)

        # test future_seq_len = 3
        self.pipeline_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df)
        config_file = self.pipeline_3.config_save()
        assert os.path.isfile(config_file)
        configs = load_config(config_file)
        os.remove(config_file)
        os.rmdir(os.path.dirname(os.path.abspath(config_file)))
        ppl = TimeSequencePipeline(name='test', config=configs)
        ppl.fit_with_fixed_configs(self.train_df, self.validation_df, mc=True)
        y_out, y_pred_uncertainty = ppl.predict_with_uncertainty(self.test_df, n_iter=2)
        assert y_out.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                               self.future_seq_len_3 + 1)
        assert y_pred_uncertainty.shape == (self.test_sample_num - self.default_past_seq_len + 1,
                                            self.future_seq_len_3)
        assert np.any(y_pred_uncertainty)

    def test_mtnet_1(self):
        mtnet_ppl_1 = self.tsp_1.fit(self.train_df, validation_df=self.validation_df,
                                     recipe=MTNetSmokeRecipe())
        mse, rs = mtnet_ppl_1.evaluate(self.test_df, metrics=["mse", "r2"])
        assert isinstance(mse, np.float)
        assert isinstance(rs, np.float)
        y_pred_1 = mtnet_ppl_1.predict(self.test_df)
        assert y_pred_1.shape == (self.test_sample_num - mtnet_ppl_1.config["past_seq_len"] + 1,
                                  self.future_seq_len_1 + 1)

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            mtnet_ppl_1.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(self.test_df)
            np.testing.assert_allclose(y_pred_1["value"].values, new_pred["value"].values)
        finally:
            shutil.rmtree(dirname)

    def test_mtnet_3(self):
        mtnet_ppl_3 = self.tsp_3.fit(self.train_df, validation_df=self.validation_df,
                                     recipe=MTNetSmokeRecipe())
        mse, rs = mtnet_ppl_3.evaluate(self.test_df, metrics=["mse", "r2"])
        assert len(mse) == self.future_seq_len_3
        assert len(rs) == self.future_seq_len_3
        y_pred_3 = mtnet_ppl_3.predict(self.test_df)
        assert y_pred_3.shape == (self.test_sample_num - mtnet_ppl_3.config["past_seq_len"] + 1,
                                  self.future_seq_len_3 + 1)

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            mtnet_ppl_3.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(self.test_df)
            columns = ["{}_{}".format("value", i) for i in range(self.future_seq_len_3)]
            np.testing.assert_allclose(y_pred_3[columns].values, new_pred[columns].values)
        finally:
            shutil.rmtree(dirname)


if __name__ == '__main__':
    pytest.main([__file__])
