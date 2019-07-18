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


class TestTimeSequencePipeline(ZooTestCase):

    def setup_method(self, method):
        super().setup_method(method)
        ray.init()

        # sample_num should > past_seq_len, the default value of which is 50
        sample_num = 100
        self.train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019',
                                                                periods=sample_num),
                                     "value": np.random.randn(sample_num)})
        self.test_sample_num = 64
        self.test_df = pd.DataFrame({"datetime": pd.date_range('1/10/2019',
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
        self.default_past_seq_len = 50

        self.fitted = False
        self.fitted_random = False

    def teardown_method(self, method):
        """
        Teardown any state that was previously setup with a setup_method call.
        """
        super().teardown_method(method)
        ray.shutdown()

    def test_fit(self):
        self.pipeline_1 = self.tsp_1.fit(train_df)
        self.pipeline_3 = self.tsp_3.fit(train_df)  

        assert isinstance(self.pipeline_1, TimeSequencePipeline)
        assert isinstance(self.pipeline_1.feature_transformers, TimeSequenceFeatureTransformer)
        assert isinstance(self.pipeline_1.model, BaseModel)
        assert self.pipeline_1.config is not None
        self.fitted = True

    def test_fit_RandomRecipe(self):
        self.random_pipeline_1 = self.tsp_1.fit(train_df, recipe=RandomRecipe(1))
        self.random_pipeline_3 = self.tsp_3.fit(train_df, recipe=RandomRecipe(1))

        assert isinstance(self.random_pipeline_1, TimeSequencePipeline)
        assert isinstance(self.random_pipeline_1.feature_transformers,
                          TimeSequenceFeatureTransformer)
        assert isinstance(self.random_pipeline_1.model, BaseModel)
        assert self.random_pipeline_1.config is not None   
        self.fitted_random = True

    def test_evaluate(self):
        if not self.fitted:
            self.pipeline_1 = self.tsp_1.fit(train_df)
            self.pipeline_3 = self.tsp_3.fit(train_df)

        mse, rs = self.pipeline_1.evaluate(test_df, metric=["mean_squared_error", "r_square"])
        assert len(mse) == self.future_seq_len_1
        assert len(rs) == self.future_seq_len_1
        print("Mean square error (future_seq_len=1) is:", mse)
        print("R square (future_seq_len=1) is:", rs)

        mse, rs = self.pipeline_3.evaluate(test_df, metric=["mean_squared_error", "r_square"])
        assert len(mse) == self.future_seq_len_3
        assert len(rs) == self.future_seq_len_3
        print("Mean square error (future_seq_len=3) is:", mse)
        print("R square (future_seq_len=3) is:", rs)

    def test_evaluate_RandomRecipe(self):
        if not self.fitted_random:
            self.random_pipeline_1 = self.tsp_1.fit(train_df, recipe=RandomRecipe(1))
            self.random_pipeline_3 = self.tsp_3.fit(train_df, recipe=RandomRecipe(1))

        mse, rs = self.random_pipeline_1.evaluate(test_df,
                                                  metric=["mean_squared_error", "r_square"])
        assert len(mse) == self.future_seq_len_1
        assert len(rs) == self.future_seq_len_1
        print("Mean square error (future_seq_len=1) is:", mse)
        print("R square (future_seq_len=1) is:", rs)

        mse, rs = self.random_pipeline_3.evaluate(test_df,
                                                  metric=["mean_squared_error", "r_square"])
        assert len(mse) == self.future_seq_len_3
        assert len(rs) == self.future_seq_len_3
        print("Mean square error (future_seq_len=3) is:", mse)
        print("R square (future_seq_len=3) is:", rs)

    def test_predict(self):
        # sample_num should > past_seq_len, the default value of which is 50
        y_pred = self.pipeline_1.predict(test_df)


        assert y_pred.shape == (self.test_sample_num - default_past_seq_len + 1, 2)

    def test_predict_RandomRecipe(self):
        y_pred = self.random_pipeline_1.predict(test_df)

        default_past_seq_len = 50
        assert y_pred.shape == (self.test_sample_num - default_past_seq_len + 1, 2)

    def test_evaluate_predict_1(self):
        future_seq_len = 1
        sample_num = 100
        dt_col = "datetime"
        target_col = "value"
        train_df = pd.DataFrame({dt_col: pd.date_range('1/1/2019', periods=sample_num),
                                 target_col: np.random.randn(sample_num)})
        test_sample_num = 64
        test_df = pd.DataFrame({dt_col: pd.date_range('1/10/2019', periods=test_sample_num),
                                target_col: np.random.randn(test_sample_num)})

        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        y_pred_df = pipeline.predict(test_df[:-future_seq_len])
        default_past_seq_len = 50
        y_df = test_df[default_past_seq_len:]

        metric = ["mean_squared_error", "r_square"]
        mse1, rs1 = [Evaluator.evaluate(m, y_df[target_col].values,
                                        y_pred_df[target_col].values) for m in metric]
        mse2, rs2 = pipeline.evaluate(test_df, metric)
        assert mse1 == mse2
        assert rs1 == rs2

    def test_evaluate_predict_2(self):
        future_seq_len = 2
        sample_num = 100
        dt_col = "datetime"
        target_col = "value"
        train_df = pd.DataFrame({dt_col: pd.date_range('1/1/2019', periods=sample_num),
                                 target_col: np.random.randn(sample_num)})
        test_sample_num = 64
        test_df = pd.DataFrame({dt_col: pd.date_range('1/10/2019', periods=test_sample_num),
                                target_col: np.random.randn(test_sample_num)})

        tsp = TimeSequencePredictor(dt_col=dt_col,
                                    target_col=target_col,
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        y_pred_df = pipeline.predict(test_df[:-future_seq_len])
        columns = ["{}_{}".format(target_col, i) for i in range(future_seq_len)]
        y_pred_value = y_pred_df[columns].values

        default_past_seq_len = 50
        y_df = test_df[default_past_seq_len:]
        y_value = TimeSequenceFeatureTransformer()._roll_test(y_df[target_col], future_seq_len)

        metric = ["mean_squared_error", "r_square"]
        mse1, rs1 = [Evaluator.evaluate(m, y_value, y_pred_value) for m in metric]
        mse2, rs2 = pipeline.evaluate(test_df, metric)
        assert np.array_equal(mse1, mse2)
        assert np.array_equal(rs1, rs2)

    def test_save_restore_1(self):
        future_seq_len = 1
        sample_num = 100
        train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                                 "value": np.random.randn(sample_num)})
        sample_num = 64
        test_df = pd.DataFrame({"datetime": pd.date_range('1/10/2019', periods=sample_num),
                                "value": np.random.randn(sample_num)})

        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        pred = pipeline.predict(test_df)

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            pipeline.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(test_df)
            np.testing.assert_allclose(pred["value"].values, new_pred["value"].values)
        finally:
            shutil.rmtree(dirname)

    def test_save_restore_2(self):
        future_seq_len = 2
        sample_num = 100
        dt_col = "dt"
        target_col = "v"
        train_df = pd.DataFrame({dt_col: pd.date_range('1/1/2019', periods=sample_num),
                                 target_col: np.random.randn(sample_num)})
        sample_num = 64
        test_df = pd.DataFrame({dt_col: pd.date_range('1/10/2019', periods=sample_num),
                                target_col: np.random.randn(sample_num)})

        tsp = TimeSequencePredictor(dt_col=dt_col,
                                    target_col=target_col,
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        pred = pipeline.predict(test_df)

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            pipeline.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(test_df)
            print(pred)
            print(new_pred)
            columns = ["{}_{}".format(target_col, i) for i in range(future_seq_len)]
            np.testing.assert_allclose(pred[columns].values, new_pred[columns].values)
        finally:
            shutil.rmtree(dirname)

    def test_save_restore_1_RandomRecipe(self):
        future_seq_len = 1
        sample_num = 100
        train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                                 "value": np.random.randn(sample_num)})
        sample_num = 64
        test_df = pd.DataFrame({"datetime": pd.date_range('1/10/2019', periods=sample_num),
                                "value": np.random.randn(sample_num)})

        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df, recipe=RandomRecipe(1))
        pred = pipeline.predict(test_df)

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            pipeline.save(save_pipeline_file)
            assert os.path.isfile(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)

            new_pred = new_pipeline.predict(test_df)
            np.testing.assert_allclose(pred["value"].values, new_pred["value"].values)
        finally:
            shutil.rmtree(dirname)

    def test_fit(self):
        # sample_num should > past_seq_len, the default value of which is 50
        sample_num = 100
        train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                                 "value": np.random.randn(sample_num)})
        sample_num = 64
        test_df = pd.DataFrame({"datetime": pd.date_range('1/10/2019', periods=sample_num),
                                "value": np.random.randn(sample_num)})

        future_seq_len = 1
        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        mse, rs = pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"])
        print("Evaluation result: Mean square error is: {}, R square is: {}.".format(mse, rs))

        dirname = tempfile.mkdtemp(prefix="saved_pipeline")
        try:
            save_pipeline_file = os.path.join(dirname, "my.ppl")
            pipeline.save(save_pipeline_file)
            new_pipeline = load_ts_pipeline(save_pipeline_file)
        finally:
            shutil.rmtree(dirname)

        new_pipeline.fit(train_df, epoch_num=10)
        new_mse, new_rs = new_pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"])
        print("Evaluation result after restore and fit: "
              "Mean square error is: {}, R square is: {}.".format(new_mse, new_rs))


if __name__ == '__main__':
    pytest.main([__file__])
