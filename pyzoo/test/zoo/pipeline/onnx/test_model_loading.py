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
import numpy as np

np.random.seed(1337)  # for reproducibility
import torch
import onnx.helper as helper
import onnx
from zoo.pipeline.api.onnx.onnx_loader import OnnxLoader


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

    def test_conv_with_padding(self):
        x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)
        W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        # Convolution with padding
        node_with_padding = helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )
        y_with_padding = np.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                                     [33., 54., 63., 72., 51.],
                                     [63., 99., 108., 117., 81.],
                                     [93., 144., 153., 162., 111.],
                                     [72., 111., 117., 123., 84.]]]]).astype(np.float32)
        output = OnnxLoader.run_node(node_with_padding, [x, W])
        np.testing.assert_almost_equal(output["y"], y_with_padding, decimal=5)

    def test_conv_without_padding(self):
        x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)
        W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)
        # Convolution without padding
        node_without_padding = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[0, 0, 0, 0],
        )
        y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                        [99., 108., 117.],
                                        [144., 153., 162.]]]]).astype(np.float32)
        output = OnnxLoader.run_node(node_without_padding, [x, W])
        np.testing.assert_almost_equal(output["y"], y_without_padding, decimal=5)

    def test_onnx_gemm(self):
        # TODO: Linear(bias = Flase) is mapped to Transpose + MatMul, not GEMM
        pytorch_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=4, bias=True)
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_add(self):
        class Add(torch.nn.Module):
            def forward(self, x):
                return x[0] + x[1]

        pytorch_model = Add()
        input_shape_with_batch = [(1, 3), (1, 3)]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_averagepool2d(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3, count_include_pad=False)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_relu(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.ReLU()
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_softmax(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.Softmax()
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_maxpool2d(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)
