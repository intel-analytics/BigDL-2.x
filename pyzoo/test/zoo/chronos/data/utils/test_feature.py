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
import pandas as pd
import numpy as np

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.chronos.data.utils.feature import generate_dt_features, generate_global_features
import tsfresh


class TestFeature(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_generate_dt_features(self):
        dates = pd.date_range('1/1/2019', periods=8)
        data = np.random.randn(8, 3)
        df = pd.DataFrame({"datetime": dates, "values": data[:, 0],
                           "A": data[:, 1], "B": data[:, 2]})
        df = generate_dt_features(df, dt_col="datetime")
        assert set(df.columns) == {'IS_AWAKE(datetime)',
                                   'IS_BUSY_HOURS(datetime)',
                                   'HOUR(datetime)',
                                   'DAY(datetime)',
                                   'IS_WEEKEND(datetime)',
                                   'WEEKDAY(datetime)',
                                   'MONTH(datetime)',
                                   'DAYOFYEAR(datetime)',
                                   'WEEKOFYEAR(datetime)',
                                   'MINUTE(datetime)',
                                   'A',
                                   'B',
                                   'values',
                                   'datetime'}

    def test_gen_global_feature_single_id(self):
        dates = pd.date_range('1/1/2019', periods=8)
        data = np.random.randn(8, 3)
        df = pd.DataFrame({"datetime": dates, "values": data[:, 0],
                           "A": data[:, 1], "B": data[:, 2],
                           "id": ["00"]*8})
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        from tsfresh.feature_extraction import MinimalFCParameters
        from tsfresh.feature_extraction import EfficientFCParameters
        for params in [ComprehensiveFCParameters(),
                       MinimalFCParameters(),
                       EfficientFCParameters()]:
            output_df, _ = generate_global_features(input_df=df,
                                                    column_id="id",
                                                    column_sort="datetime",
                                                    default_fc_parameters=params)

            assert "datetime" in output_df.columns
            assert "values" in output_df.columns
            assert "A" in output_df.columns
            assert "B" in output_df.columns
            assert "id" in output_df.columns

            for col in output_df.columns:
                if col in ["datetime", "values", "A", "B", "id"]:
                    continue
                assert len(set(output_df[col])) == 1
                assert output_df[col].isna().sum() == 0

    def test_gen_global_feature_multi_id(self):
        dates = pd.date_range('1/1/2019', periods=8)
        data = np.random.randn(8, 3)
        df = pd.DataFrame({"datetime": dates, "values": data[:, 0],
                           "A": data[:, 1], "B": data[:, 2],
                           "id": ["00"]*4+["01"]*4})
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        from tsfresh.feature_extraction import MinimalFCParameters
        from tsfresh.feature_extraction import EfficientFCParameters
        for params in [ComprehensiveFCParameters(),
                       MinimalFCParameters(),
                       EfficientFCParameters()]:
            output_df, _ = generate_global_features(input_df=df,
                                                    column_id="id",
                                                    column_sort="datetime",
                                                    default_fc_parameters=params)

            assert "datetime" in output_df.columns
            assert "values" in output_df.columns
            assert "A" in output_df.columns
            assert "B" in output_df.columns
            assert "id" in output_df.columns

            for col in output_df.columns:
                if col in ["datetime", "values", "A", "B", "id"]:
                    continue
                assert len(set(output_df[output_df["id"] == "00"][col])) == 1
                assert len(set(output_df[output_df["id"] == "01"][col])) == 1
                assert output_df[output_df["id"] == "00"][col].isna().sum() == 0
                assert output_df[output_df["id"] == "01"][col].isna().sum() == 0
