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
from zoo.models.anomalydetection import AnomalyDetector
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestAnomalyDetector(ZooTestCase):

    def test_forward_backward(self):
        model = AnomalyDetector(feature_shape=(10, 3), hidden_layers=[8, 32, 15],
                                dropouts=[0.2, 0.2, 0.2])
        model.summary()
        input_data = np.random.rand(100, 10, 3)
        self.assert_forward_backward(model, input_data)
        model.set_evaluate_status()
        # Forward twice will get the same output
        output1 = model.forward(input_data)
        output2 = model.forward(input_data)
        assert np.allclose(output1, output2)

    def test_save_load(self):
        model = AnomalyDetector(feature_shape=(10, 3), hidden_layers=[8, 32, 15],
                                dropouts=[0.2, 0.2, 0.2])
        input_data = np.random.rand(100, 10, 3)
        self.assert_zoo_model_save_load(model, input_data)

if __name__ == "__main__":
    pytest.main([__file__])
