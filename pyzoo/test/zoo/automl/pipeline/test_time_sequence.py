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

from zoo.automl.pipeline.time_sequence import *
from zoo.automl.regression.time_sequence_predictor import TimeSequencePredictor
import numpy as np
import pandas as pd


class TestTimeSequencePipeline:

    def test_evaluate(self):
        # sample_num should > past_seq_len, the default value of which is 50
        sample_num = 100
        train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                                 "value": np.random.randn(sample_num)})
        sample_num = 64
        test_df = pd.DataFrame({"datetime": pd.date_range('1/10/2019', periods=sample_num),
                                 "value": np.random.randn(sample_num)})

        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=1,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        evaluate_value = pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"])
        print("evaluate:", evaluate_value)

    def test_predict(self):
        # sample_num should > past_seq_len, the default value of which is 50
        sample_num = 100
        train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                                 "value": np.random.randn(sample_num)})
        test_sample_num = 64
        test_df = pd.DataFrame({"datetime": pd.date_range('1/10/2019', periods=test_sample_num),
                                "value": np.random.randn(test_sample_num)})

        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=1,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        y_pred = pipeline.predict(test_df)

        default_past_seq_len = 50
        assert y_pred.shape == (test_sample_num - default_past_seq_len + 1, 2)

    def test_evaluate_predict(self):
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
        e = Evaluator()
        metric = ["mean_squared_error", "r_square"]
        mse1, rs1 = [e.evaluate(m, y_df[target_col].values, y_pred_df[target_col].values) for m in metric]
        mse2, rs2 = pipeline.evaluate(test_df, metric)
        assert mse1 == mse2
        assert rs1 == rs2

    def test_save_restore(self):
        sample_num = 100
        train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                                 "value": np.random.randn(sample_num)})
        sample_num = 64
        test_df = pd.DataFrame({"datetime": pd.date_range('1/10/2019', periods=sample_num),
                                "value": np.random.randn(sample_num)})

        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
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


if __name__ == '__main__':
    pytest.main([__file__])