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

from zoo.models.recommendation.utils import *
from zoo.common.nncontext import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestRecommenderUtils:

    def test_get_boundaries(self):
        index = get_boundaries(42, [18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
        assert index == 5

    def test_categorical_from_vocab_list(self):
        MARITAL_STATUS_VOCAB = ["Married-civ-spouse", "Divorced", "Married-spouse-absent",
                                "Never-married", "Separated", "Married-AF-spouse", "Widowed"]
        index = categorical_from_vocab_list("Never-married", MARITAL_STATUS_VOCAB)
        assert index == 3

    def test_hash_bucket(self):
        np.random.seed(1337)
        res = hash_bucket("Prof-specialty", 1000)
        assert res < 1000


if __name__ == "__main__":
    pytest.main([__file__])
