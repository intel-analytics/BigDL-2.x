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

    def test_fillna(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_df = FeatureTable.read_parquet(file_path)
        feature_df.fillna(0, ["col_1"])
        feature_df.show()
