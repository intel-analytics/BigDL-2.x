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

import os
import pytest
import numpy as np

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.inference import InferenceModel

np.random.seed(1337)  # for reproducibility

resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
base_dir = "/home/kai/Documents/openvino/"


class TestInferenceModel(ZooTestCase):
    def test_load_model(self):
        model = InferenceModel(3)
        model.load(os.path.join(resource_path, "models/bigdl/bigdl_lenet.model"))
        input_data = np.random.random([4, 28, 28, 1])
        output_data = model.predict(input_data)

    def test_load_openvino_ir(self):
        model = InferenceModel()
        model.load_openvino_ir(base_dir + "frozen_inference_graph.xml", base_dir + "frozen_inference_graph.bin", "cpu")
        input_data = np.random.random([1, 1, 3, 600, 600])
        output_data = model.predict(input_data)

    def test_load_tf_as_openvino(self):
        model = InferenceModel(3)
        model.load_tf_as_openvino(base_dir + "frozen_inference_graph.pb", base_dir + "pipeline.config",
                                  base_dir + "faster_rcnn_support.json", "cpu")
        input_data = np.random.random([1, 1, 3, 600, 600])
        output_data = model.predict(input_data)


if __name__ == "__main__":
    pytest.main([__file__])
