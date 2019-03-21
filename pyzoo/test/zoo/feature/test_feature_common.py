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
from bigdl.transform.vision.image import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.feature.common import *


class TestFeatureCommon(ZooTestCase):

    def test_BigDL_adapter(self):
        new_preprocessing = BigDLAdapter(Resize(1, 1))
        assert isinstance(new_preprocessing, Preprocessing)

    def test_relations(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        path = os.path.join(resource_path, "qa")
        relations = Relations.read(path + "/relations.txt")
        assert isinstance(relations, list)
        relations2 = Relations.read(path + "/relations.csv", self.sc, 2)
        assert isinstance(relations2, RDD)
        relations3 = Relations.read_parquet(path + "/relations.parquet", self.sc)
        assert isinstance(relations3, RDD)


if __name__ == "__main__":
    pytest.main([__file__])
