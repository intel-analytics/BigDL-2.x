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
import os

import numpy as np
from zoo.models.textclassification import TextClassifier
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility

resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
glove_path = os.path.join(resource_path, "glove.6B/glove.6B.50d.txt")


class TestTextClassifier(ZooTestCase):

    def test_forward_backward(self):
        model = TextClassifier(10, glove_path)
        model.summary()
        input_data = np.random.randint(20, size=(4, 500))
        self.assert_forward_backward(model, input_data)
        model.set_evaluate_status()
        # Forward twice will get the same output
        output1 = model.forward(input_data)
        output2 = model.forward(input_data)
        assert np.allclose(output1, output2)

    def test_save_load(self):
        model = TextClassifier(20, glove_path, sequence_length=100)
        input_data = np.random.randint(20, size=(3, 100))
        self.assert_zoo_model_save_load(model, input_data)


if __name__ == "__main__":
    pytest.main([__file__])
