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

import numpy as np
from zoo.models.textclassification import TextClassifier
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestTextClassifier(ZooTestCase):

    def test_forward_backward(self):
        model = TextClassifier(10, 30, 100)
        model.summary()
        input_data = np.random.random([3, 100, 30])
        self.assert_forward_backward(model, input_data)

    def test_save_load(self):
        model = TextClassifier(20, 200)
        input_data = np.random.random([2, 500, 200])
        self.assert_zoo_model_save_load(model, input_data)


if __name__ == "__main__":
    pytest.main([__file__])
