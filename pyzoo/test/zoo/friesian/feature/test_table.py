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

import os.path
import shutil

import pytest
from unittest import TestCase

from zoo.common.nncontext import *
from zoo.friesian.feature import FeatureTable


class TestTable(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

    def test_fillna_int(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_df = FeatureTable.read_parquet(file_path)
        filled_df = feature_df.fillna(0, ["col_1", "col_2"])
        assert filled_df.df.filter("col_1 is null").count() == 0, "col_1 null values should be " \
                                                                  "filled"
        assert filled_df.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                  "filled"

    def test_fillna_double(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_df = FeatureTable.read_parquet(file_path)
        filled_df = feature_df.fillna(3.2, ["col_1", "col_2"])
        assert filled_df.df.filter("col_1 is null").count() == 0, "col_1 null values should be " \
                                                                  "filled"
        assert filled_df.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                  "filled"

    def test_fillna_string(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_df = FeatureTable.read_parquet(file_path)
        with self.assertRaises(Exception) as context:
            feature_df.fillna(3.2, ["col_3", "col_4"])
        self.assertTrue('numeric does not match the type of column col_3' in str(context.exception))

        filled_df = feature_df.fillna("bb", ["col_3", "col_4"])
        assert filled_df.df.filter("col_3 is null").count() == 0, "col_3 null values should be " \
                                                                  "filled"
        assert filled_df.df.filter("col_4 is null").count() == 0, "col_4 null values should be " \
                                                                  "filled"

    def test_gen_string_idx(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_df = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_df.gen_string_idx(["col_3", "col_4"], freq_limit="1")
        assert string_idx_list[0].df.count() == 3, "col_3 should have 3 indices"
        assert string_idx_list[1].df.count() == 2, "col_4 should have 2 indices"
