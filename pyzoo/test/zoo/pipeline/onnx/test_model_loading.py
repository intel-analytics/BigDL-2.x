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


from test.zoo.pipeline.utils.test_utils_onnx import OnnxTestCase
from zoo.pipeline.api.keras.layers import *

np.random.seed(1337)  # for reproducibility
import torch


class TestModelLoading(OnnxTestCase):
    def test_onnx_conv2d(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_conv2d_2(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            torch.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3)

        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)
