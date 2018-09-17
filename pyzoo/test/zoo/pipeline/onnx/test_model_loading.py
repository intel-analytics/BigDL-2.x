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

    def test_relu(self):
        node = helper.make_node(
            'Relu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_softmax(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.Softmax()
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_softmax(self):
        node = helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output [[0.09003058, 0.24472848, 0.66524094]]
        y = np.exp(x) / np.sum(np.exp(x), axis=1)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_maxpool2d(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_reshape(self):
        original_shape = [2, 3, 4]
        test_cases = {
            'reordered_dims': np.array([4, 2, 3], dtype=np.int64),
            'reduced_dims': np.array([3, 8], dtype=np.int64),
            'extended_dims': np.array([3, 2, 2, 2], dtype=np.int64),
            'one_dim': np.array([24], dtype=np.int64)
            # 'negative_dim': np.array([6, -1, 2], dtype=np.int64),
        }
        data = np.random.random_sample(original_shape).astype(np.float32)

        for test_name, shape in test_cases.items():
            node = onnx.helper.make_node(
                'Reshape',
                inputs=['data', 'shape'],
                outputs=['reshaped'],
            )

            output = OnnxLoader.run_node(node, [data, shape])
            reshaped = np.reshape(data, shape)
            np.testing.assert_almost_equal(output["reshaped"], reshaped, decimal=5)

    def test_reshape_pytorch(self):
        class View(torch.nn.Module):
            def __init__(self, *shape):
                super(View, self).__init__()
                self.shape = shape

            def forward(self, input):
                return input.view(self.shape)

        pytorch_model = torch.nn.Sequential(
            torch.nn.Linear(20, 20),
            View(2, 5, 4))
        input_shape_with_batch = (2, 20)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_constant(self):
        values = np.random.randn(5, 5).astype(np.float32)
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )
        output = OnnxLoader.run_node(node, [])
        np.testing.assert_almost_equal(output["values"], values, decimal=5)

    def test_maxpool2d_pads(self):
        node = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2]

        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[
            [13, 14, 15, 15, 15],
            [18, 19, 20, 20, 20],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25]]]]).astype(np.float32)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_maxpool2d_same_upper(self):
        node = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad="SAME_UPPER"
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9, 10],
                        [17, 19, 20],
                        [22, 24, 25]]]]).astype(np.float32)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_maxpool2d_strides(self):
        node = helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            strides=[2, 2]
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9],
                        [17, 19]]]]).astype(np.float32)

    def test_onnx_logsoftmax(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.LogSoftmax()
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_tanh(self):
        node = onnx.helper.make_node(
            'Tanh',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.tanh(x)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y)

    def test_onnx_exp(self):
        node = onnx.helper.make_node(
            'Exp',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.exp(x)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_flatten(self):
        node = onnx.helper.make_node(
            'Flatten',
            inputs=['a'],
            outputs=['b'],
        )
        shape = (5, 4, 3, 2)
        a = np.random.random_sample(shape).astype(np.float32)
        new_shape = (5, 24)
        b = np.reshape(a, new_shape)
        output = OnnxLoader.run_node(node, [a])
        np.testing.assert_almost_equal(output["b"], b, decimal=5)

    def test_onnx_sqrt(self):
        node = onnx.helper.make_node(
            'Sqrt',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.sqrt(x)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_log(self):
        node = onnx.helper.make_node(
            'Log',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.exp(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.log(x)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_hardsigmoid(self):
        default_alpha = 0.2
        default_beta = 0.5
        node = onnx.helper.make_node(
            'HardSigmoid',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x * default_alpha + default_beta, 0, 1)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)
