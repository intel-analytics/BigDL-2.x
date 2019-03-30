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
import pytest
from zoo.pipeline.api.onnx.onnx_loader import OnnxLoader
from onnx import backend
from onnx.backend import test
from onnx.backend.test.case import node
from onnx.backend.test.case.node import pool_op_common


class Squeeze(torch.nn.Module):
    def __init__(self, *dim):
        super(Squeeze, self).__init__()
        if dim:
            self.dim = dim[0]
        else:
            self.dim = -1

    def forward(self, x):
        if (self.dim >= 0):
            return torch.squeeze(x, dim=self.dim)
        else:
            return torch.squeeze(x)


class Transpose(torch.nn.Module):
    def __init__(self, *parameter):
        super(Transpose, self).__init__()
        self.dim0 = parameter[0]
        self.dim1 = parameter[1]

    def forward(self, x):
        return torch.transpose(x, dim0=self.dim0, dim1=self.dim1)


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

    def _batchnorm_test_mode(self, x, s, bias, mean, var, epsilon=1e-5):
        dims_x = len(x.shape)
        dim_ones = (1,) * (dims_x - 2)
        s = s.reshape(-1, *dim_ones)
        bias = bias.reshape(-1, *dim_ones)
        mean = mean.reshape(-1, *dim_ones)
        var = var.reshape(-1, *dim_ones)
        return s * (x - mean) / np.sqrt(var + epsilon) + bias

    # Momentum is always equal to 1 no matter what value we set
    def test_onnx_batch_norm1(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=3, momentum=1, affine=False)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch, rtol=1e-3, atol=1e-3)

    # Momentum is always equal to 1 no matter what value we set
    def test_onnx_batch_norm2(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=3, momentum=1, affine=True)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch, rtol=1e-3, atol=1e-3)

    def test_batch_norm(self):
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32).reshape((3, 2, 1, 1))
        s = np.array([1.0, 1.0]).astype(np.float32).reshape((2, 1))
        bias = np.array([0, 0]).astype(np.float32).reshape((2, 1))
        mean = np.array([0, 3]).astype(np.float32).reshape((2, 1))
        var = np.array([1, 1.5]).astype(np.float32).reshape((2, 1))
        y = self._batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
        )
        output = OnnxLoader.run_node(node, [x, s, bias, mean, var])
        np.testing.assert_almost_equal(output["y"], y, decimal=3)

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

    def test_onnx_abs(self):
        class Abs(torch.nn.Module):
            def forward(self, x):
                return abs(x)

        pytorch_model = Abs()
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_abs(self):
        node = onnx.helper.make_node(
            'Abs',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.abs(x)

    def test_onnx_neg(self):
        class Neg(torch.nn.Module):
            def forward(self, x):
                return -x

        pytorch_model = Neg()
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_neg(self):
        node = onnx.helper.make_node(
            'Neg',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-4, 2]).astype(np.float32).reshape([2, 1])
        y = np.negative(x)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.negative(x)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_averagepool2d(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3, count_include_pad=False)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_averagepool2d_padding(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=10, padding=4, count_include_pad=False)
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
            outputs=['y']
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
            outputs=['y']
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output [[0.09003058, 0.24472848, 0.66524094]]
        y = np.exp(x) / np.sum(np.exp(x), axis=1)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

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

    def test_onnx_maxpool2d(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3)
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_maxpool2d_pads(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, padding=(0, 1))
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

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

    def test_maxpool2d_pads01(self):
        import pytest
        with pytest.raises(Exception) as e_info:
            node = onnx.helper.make_node(
                'MaxPool',
                inputs=['x'],
                outputs=['y'],
                kernel_shape=[3, 3],
                pads=[0, 0, 1, 1]
            )
            x = np.random.randn(1, 3, 28, 28).astype(np.float32)
            x_shape = np.shape(x)
            kernel_shape = (3, 3)
            strides = (1, 1)
            pad_top = pad_left = 0
            pad_bottom = pad_right = 1
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            out_shape = pool_op_common.get_output_shape('VALID', np.add(x_shape[2:], pad_shape),
                                                        kernel_shape, strides)
            padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=np.nan)
            y = pool_op_common.pool(padded, x_shape, kernel_shape, strides,
                                    out_shape, pad_shape, 'MAX')
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
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

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

    def test_matmul_2d(self):
        node = onnx.helper.make_node(
            'MatMul',
            inputs=['a', 'b'],
            outputs=['c'],
        )

        # 2d
        a = np.random.randn(3, 4).astype(np.float32).reshape((3, 4))
        b = np.random.randn(4, 3).astype(np.float32).reshape((4, 3))
        c = np.matmul(a, b)
        output = OnnxLoader.run_node(node, [a, b])
        np.testing.assert_almost_equal(output["c"], c, decimal=5)

    def test_matmul_3d(self):
        node = onnx.helper.make_node(
            'MatMul',
            inputs=['a', 'b'],
            outputs=['c'],
        )
        # 3d
        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(2, 4, 3).astype(np.float32)
        c = np.matmul(a, b)
        output = OnnxLoader.run_node(node, [a, b])
        np.testing.assert_almost_equal(output["c"], c, decimal=5)

    def test_minit(self):
        import torch.nn as nn
        import torch.nn.functional as F

        class MnistNet(nn.Module):
            def __init__(self):
                super(MnistNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = F.dropout2d(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        pytorch_model = MnistNet()
        pytorch_model.train(mode=False)
        self.compare_with_pytorch(pytorch_model, [(1, 1, 28, 28)])

    def test_onnx_sub(self):
        class Sub(torch.nn.Module):
            def forward(self, x):
                return x[0] - x[1]

        pytorch_model = Sub()
        input_shape_with_batch = [(1, 3), (1, 3)]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_sub(self):
        node = onnx.helper.make_node(
            'Sub',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32).reshape([3, 1])
        y = np.array([3, 2, 1]).astype(np.float32).reshape([3, 1])
        z = x - y
        output = OnnxLoader.run_node(node, [x, y])
        np.testing.assert_almost_equal(output["z"], z, decimal=5)

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x - y
        output = OnnxLoader.run_node(node, [x, y])
        np.testing.assert_almost_equal(output["z"], z, decimal=5)

    def test_onnx_squeeze(self):
        pytorch_model = Squeeze()
        input_shape_with_batch = (2, 1, 2, 1, 2)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_squeeze_dim0(self):
        pytorch_model = Squeeze(0)
        input_shape_with_batch = (1, 2, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_squeeze_dim1(self):
        pytorch_model = Squeeze(1)
        input_shape_with_batch = (2, 1, 3, 1, 2)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_squeeze(self):
        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['x'],
            outputs=['y'],
            axes=[0],
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        y = np.squeeze(x, axis=0)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_squeeze_none(self):
        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(1, 1, 4, 5).astype(np.float32)
        y = np.squeeze(x)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_squeeze_list(self):
        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['x'],
            outputs=['y'],
            axes=[0, 1],
        )
        x = np.random.randn(1, 1, 4, 5).astype(np.float32)
        y = np.squeeze(x)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_squeeze_axis(self):
        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['x'],
            outputs=['y'],
            axes=[1],
        )
        x = np.random.randn(3, 1, 4, 5).astype(np.float32)
        y = np.squeeze(x, axis=1)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_sigmoid(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.Sigmoid()
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_sigmoid(self):
        node = helper.make_node(
            'Sigmoid',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        y = 1.0 / (1.0 + np.exp(np.negative(x)))  # expected output [0.26894143, 0.5, 0.7310586]
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_index_select(self):
        class IndexSelect(torch.nn.Module):
            def __init__(self, *parameter):
                super(IndexSelect, self).__init__()
                self.dim = parameter[0]
                self.index = parameter[1]

            def forward(self, x):
                return torch.index_select(x, dim=self.dim, index=torch.tensor(self.index))

        pytorch_model = IndexSelect(3, 2)
        input_shape_with_batch = (3, 4, 5, 6)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_index_select_axis0(self):
        import pytest
        with pytest.raises(Exception) as e_info:
            class IndexSelect(torch.nn.Module):
                def __init__(self, *parameter):
                    super(IndexSelect, self).__init__()
                    self.dim = parameter[0]
                    self.index = parameter[1]

                def forward(self, x):
                    return torch.index_select(x, dim=self.dim, index=torch.tensor(self.index))

            pytorch_model = IndexSelect(0, 2)
            input_shape_with_batch = (3, 4, 5, 6)
            self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_concat(self):
        class Concat(torch.nn.Module):
            def forward(self, x):
                return torch.cat([v for v in x], 1)

        pytorch_model = Concat()
        input_shape_with_batch = [(1, 3), (1, 3)]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_concat(self):
        test_cases = {
            '1d': ([1, 2],
                   [3, 4]),
            '2d': ([[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]),
            '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                   [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        }  # type: Dict[Text, Sequence[Any]]

        for test_case, values_ in test_cases.items():
            values = [np.asarray(v, dtype=np.float32) for v in values_]
            for i in range(1, len(values[0].shape)):
                in_args = ['value' + str(k) for k in range(len(values))]
                node = onnx.helper.make_node(
                    'Concat',
                    inputs=[s for s in in_args],
                    outputs=['output'],
                    axis=i
                )
                y = np.concatenate(values, i)
                output = OnnxLoader.run_node(node, [v for v in values])
                np.testing.assert_almost_equal(output["output"], y, decimal=5)

    def test_concat_axis(self):
        test_cases = {
            '1d': ([1, 2],
                   [3, 4]),
            '2d': ([[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]),
            '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                   [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        }  # type: Dict[Text, Sequence[Any]]

        for test_case, values_ in test_cases.items():
            values = [np.asarray(v, dtype=np.float32) for v in values_]
            for i in range(1, len(values[0].shape)):
                in_args = ['value' + str(k) for k in range(len(values))]
                node = onnx.helper.make_node(
                    'Concat',
                    inputs=[s for s in in_args],
                    outputs=['output'],
                    axis=0
                )
                y = np.concatenate(values, 0)
                output = OnnxLoader.run_node(node, [v for v in values])
                np.testing.assert_almost_equal(output["output"], y, decimal=5)

    def test_torch_add(self):
        class Add(torch.nn.Module):
            def forward(self, x):
                return torch.add(x[0], 1, x[1])

        pytorch_model = Add()
        input_shape_with_batch = [(1, 3), (1, 3)]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_leakyrelu(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.LeakyReLU()
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_leakyrelu(self):
        node = helper.make_node(
            'LeakyRelu',
            inputs=['x'],
            outputs=['y'],
            alpha=0.1
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-0.1, 0., 1.]
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_gt(self):
        class gt(torch.nn.Module):
            def forward(self, x):
                return torch.gt(x[0], x[1])

        pytorch_model = gt()
        input_shape_with_batch = [(1, 3), (1, 3)]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_gt(self):
        node = helper.make_node(
            'Greater',
            inputs=['x', 'y'],
            outputs=['greater'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater(x, y)

        output = OnnxLoader.run_node(node, [x, y])
        np.testing.assert_almost_equal(output['greater'], z, decimal=5)

    def test_maxpool1d(self):
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2],
        )
        x = np.random.randn(1, 3, 32).astype(np.float32)
        x_shape = np.array(np.shape(x))
        kernel_shape = np.array([2])
        strides = [1]
        out_shape = pool_op_common.get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool_op_common.pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'MAX')
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_maxpool1d_strides(self):
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2],
            strides=[2]
        )
        x = np.random.randn(1, 3, 32).astype(np.float32)
        x_shape = np.array(np.shape(x))
        kernel_shape = np.array([2])
        strides = [2]
        out_shape = pool_op_common.get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool_op_common.pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'MAX')
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_maxpool1d(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.MaxPool1d(2)
        )
        input_shape_with_batch = (1, 3, 32)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_maxpool1d_pads(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.MaxPool1d(2, padding=1)
        )
        input_shape_with_batch = (1, 3, 32)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_threshold(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.Threshold(0, 0))
        input_shape_with_batch = (2, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_mul(self):
        class Mul(torch.nn.Module):
            def forward(self, x):
                return x[0] * x[1]

        pytorch_model = Mul()
        input_shape_with_batch = [(1, 3), (1, 3)]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_mul1(self):
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32).reshape([3, 1])
        y = np.array([4, 5, 6]).astype(np.float32).reshape([3, 1])
        z = x * y  # expected output [4., 10., 18.]
        output = OnnxLoader.run_node(node, [x, y])
        np.testing.assert_almost_equal(output['z'], z, decimal=5)

    def test_mul2(self):
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x * y
        output = OnnxLoader.run_node(node, [x, y])
        np.testing.assert_almost_equal(output['z'], z, decimal=5)

    def test_onnx_div(self):
        class Div(torch.nn.Module):
            def forward(self, x):
                return x[0] / x[1]

        pytorch_model = Div()
        input_shape_with_batch = [(1, 3), (1, 3)]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_div1(self):
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([3, 4]).astype(np.float32).reshape([2, 1])
        y = np.array([1, 2]).astype(np.float32).reshape([2, 1])
        z = x / y
        output = OnnxLoader.run_node(node, [x, y])
        np.testing.assert_almost_equal(output["z"], z, decimal=5)

    def test_div2(self):
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
        z = x / y
        output = OnnxLoader.run_node(node, [x, y])
        np.testing.assert_almost_equal(output["z"], z, decimal=5)

    def test_pow(self):
        class Power(torch.nn.Module):
            def forward(self, x):
                return torch.pow(x, 2)

        pytorch_model = Power()
        input_shape_with_batch = (1, 2, 2)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_elu(self):
        node = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y'],
            alpha=2.0
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_elu_default(self):
        node = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y']
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 1.0
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_elu_default(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.ELU()
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_elu(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.ELU(alpha=2)
        )
        input_shape_with_batch = (1, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_torch_clip(self):
        class clamp(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, -1, 1)

        pytorch_model = torch.nn.Sequential(
            clamp()
        )
        input_shape_with_batch = (1, 3, 32)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_exception_clip(self):
        import pytest
        with pytest.raises(Exception) as e_info:
            class clamp(torch.nn.Module):
                def forward(self, x):
                    return torch.clamp(x, 1, -1)

            pytorch_model = torch.nn.Sequential(
                clamp()
            )
            input_shape_with_batch = (1, 3, 32)
            self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_embedding(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=10, embedding_dim=3)
        )
        input_shape_with_batch = (2, 4)
        input_data_with_batch = [[[1, 2, 4, 5], [4, 3, 2, 9]]]
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch, input_data_with_batch)

    def test_onnx_slice1(self):
        class Slice(torch.nn.Module):
            def __init__(self, *parameter):
                super(Slice, self).__init__()
                self.axes = parameter[0]
                self.starts = parameter[1]
                self.ends = parameter[2]

            def forward(self, x):
                return x[self.starts:self.ends]

        pytorch_model = Slice(0, 0, 2)
        input_shape_with_batch = (3, 3, 3)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_slice1_start_out_of_bounds(self):
        with pytest.raises(Exception) as e_info:
            node = onnx.helper.make_node(
                'Slice',
                inputs=['x'],
                outputs=['y'],
                axes=[0],
                starts=[1000],
                ends=[1000],
            )

            x = np.random.randn(3, 3, 3).astype(np.float32)
            y = x[1000:1000]
            output = OnnxLoader.run_node(node, [x])
            np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_slice2(self):
        class Slice(torch.nn.Module):
            def __init__(self, *parameter):
                super(Slice, self).__init__()
                self.axes = parameter[0]
                self.starts = parameter[1]
                self.ends = parameter[2]

            def forward(self, x):
                return x[self.starts[0]:self.ends[0], self.starts[1]:self.ends[1]]

        pytorch_model = Slice([0, 1], [0, 0], [2, -2])
        input_shape_with_batch = (20, 10, 5)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_slice2_neg(self):
        node = onnx.helper.make_node(
            'Slice',
            inputs=['x'],
            outputs=['y'],
            axes=[0, 1],
            starts=[0, 0],
            ends=[2, -2],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        y = x[0:2, 0:-2]
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_slice3(self):
        class Slice(torch.nn.Module):
            def __init__(self, *parameter):
                super(Slice, self).__init__()
                self.axes = parameter[0]
                self.starts = parameter[1]
                self.ends = parameter[2]

            def forward(self, x):
                return x[self.starts[0]:self.ends[0], self.starts[1]:self.ends[1],
                         self.starts[2]:self.ends[2]]

        pytorch_model = Slice([0, 1, 2], [0, 0, 3], [20, 10, 4])
        input_shape_with_batch = (20, 10, 5)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_slice3_default_axes(self):
        node = onnx.helper.make_node(
            'Slice',
            inputs=['x'],
            outputs=['y'],
            starts=[0, 0, 3],
            ends=[20, 10, 4],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        y = x[:, :, 3:4]
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_reducemean_keepdims(self):
        class ReduceMean(torch.nn.Module):
            def __init__(self, *parameter):
                super(ReduceMean, self).__init__()
                self.dim = parameter[0]
                self.keepdim = parameter[1]

            def forward(self, x):
                return torch.mean(x, dim=self.dim, keepdim=self.keepdim)

        pytorch_model = ReduceMean(1, True)
        input_shape_with_batch = (1, 2, 2)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_reducemean(self):
        class ReduceMean(torch.nn.Module):
            def __init__(self, *parameter):
                super(ReduceMean, self).__init__()
                self.dim = parameter[0]
                self.keepdim = parameter[1]

            def forward(self, x):
                return torch.mean(x, dim=self.dim, keepdim=self.keepdim)

        pytorch_model = ReduceMean(1, False)
        input_shape_with_batch = (1, 2, 2)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_reducemean_do_not_keepdims(self):
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceMean',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)

        data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
                        dtype=np.float32)
        reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
        output = OnnxLoader.run_node(node, [data])
        np.testing.assert_almost_equal(output["reduced"], reduced, decimal=5)

    def test_reducemean_keepdims(self):
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceMean',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
        output = OnnxLoader.run_node(node, [data])
        np.testing.assert_almost_equal(output["reduced"], reduced, decimal=5)

    def test_onnx_reducesum_keepdims(self):
        class ReduceSum(torch.nn.Module):
            def __init__(self, *parameter):
                super(ReduceSum, self).__init__()
                self.dim = parameter[0]
                self.keepdim = parameter[1]

            def forward(self, x):
                return torch.sum(x, dim=self.dim, keepdim=self.keepdim)

        pytorch_model = ReduceSum(1, True)
        input_shape_with_batch = (20, 10, 5)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_reducesum(self):
        class ReduceSum(torch.nn.Module):
            def __init__(self, *parameter):
                super(ReduceSum, self).__init__()
                self.dim = parameter[0]
                self.keepdim = parameter[1]

            def forward(self, x):
                return torch.sum(x, dim=self.dim, keepdim=self.keepdim)

        pytorch_model = ReduceSum(1, False)
        input_shape_with_batch = (20, 10, 5)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_reducesum_do_not_keepdims(self):
        axes = [1]
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
                        dtype=np.float32)
        reduced = np.sum(data, axis=tuple(axes), keepdims=keepdims == 1)
        output = OnnxLoader.run_node(node, [data])
        np.testing.assert_almost_equal(output["reduced"], reduced, decimal=5)

    def test_reducesum_keepdims(self):
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)
        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=tuple(axes), keepdims=keepdims == 1)
        output = OnnxLoader.run_node(node, [data])
        np.testing.assert_almost_equal(output["reduced"], reduced, decimal=5)

    def test_onnx_unsqueeze_axis0(self):
        class Unsqueeze(torch.nn.Module):
            def __init__(self, *parameter):
                super(Unsqueeze, self).__init__()
                self.dim = parameter[0]

            def forward(self, x):
                return torch.unsqueeze(x, dim=self.dim)

        pytorch_model = Unsqueeze(0)
        input_shape_with_batch = (1, 2, 2)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_unsqueeze_axis0(self):
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[0],
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        y = np.expand_dims(x, axis=0)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_unsqueeze_axis1(self):
        class Unsqueeze(torch.nn.Module):
            def __init__(self, *parameter):
                super(Unsqueeze, self).__init__()
                self.dim = parameter[0]

            def forward(self, x):
                return torch.unsqueeze(x, dim=self.dim)

        pytorch_model = Unsqueeze(1)
        input_shape_with_batch = (1, 2, 2)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_unsqueeze_axis1(self):
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[1],
        )
        x = np.random.randn(3, 1, 4, 5).astype(np.float32)
        y = np.expand_dims(x, axis=1)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_unsqueeze_list(self):
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[0, 4],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=4)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_transpose(self):
        pytorch_model = Transpose(2, 3)
        input_shape_with_batch = (3, 7, 8, 9)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_onnx_transpose0(self):
        import pytest
        with pytest.raises(Exception) as e_info:
            pytorch_model = Transpose(0, 3)
            input_shape_with_batch = (3, 7, 8, 9)
            self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_transpose(self):
        shape = (2, 3, 4)
        data = np.random.random_sample(shape).astype(np.float32)
        permutation = (0, 2, 1)

        node = onnx.helper.make_node(
            'Transpose',
            inputs=['data'],
            outputs=['transposed'],
            perm=permutation
        )
        transposed = np.transpose(data, permutation)
        output = OnnxLoader.run_node(node, [data])
        np.testing.assert_almost_equal(output["transposed"], transposed, decimal=5)

    def test_transpose_default(self):
        import pytest
        with pytest.raises(Exception) as e_info:
            shape = (2, 3, 4)
            data = np.random.random_sample(shape).astype(np.float32)

            node = onnx.helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed']
            )

            transposed = np.transpose(data)
            output = OnnxLoader.run_node(node, [data])
            np.testing.assert_almost_equal(output["transposed"], transposed, decimal=5)

    def test_shape(self):
        node = onnx.helper.make_node(
            'Shape',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).astype(np.float32)
        y = np.array([
            2, 3,
        ]).astype(np.int64)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.array(x.shape).astype(np.int64)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_globalaveragepool(self):
        node = onnx.helper.make_node(
            'GlobalAveragePool',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(2, 3, 7, 5).astype(np.float32)
        spatial_shape = np.ndim(x) - 2
        y = np.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
        for _ in range(spatial_shape):
            y = np.expand_dims(y, -1)
        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_onnx_globalaveragepool2(self):
        pytorch_model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        input_shape_with_batch = (1, 3, 224, 224)
        self.compare_with_pytorch(pytorch_model, input_shape_with_batch)

    def test_lrn_default(self):
        import math
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        nsize = 3
        node = onnx.helper.make_node(
            'LRN',
            inputs=['x'],
            outputs=['y'],
            size=3
        )
        x = np.random.randn(5, 5, 5, 5).astype(np.float32)
        square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = sum(
                x[n, max(0, c - int(math.floor((nsize - 1) / 2))):
                  min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                  h, w] ** 2)
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)

    def test_lrn(self):
        import math
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        nsize = 3
        node = onnx.helper.make_node(
            'LRN',
            inputs=['x'],
            outputs=['y'],
            alpha=alpha,
            beta=beta,
            bias=bias,
            size=nsize
        )
        x = np.random.randn(5, 5, 5, 5).astype(np.float32)
        square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = sum(
                x[n, max(0, c - int(math.floor((nsize - 1) / 2))):
                  min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                  h, w] ** 2)
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)

        output = OnnxLoader.run_node(node, [x])
        np.testing.assert_almost_equal(output["y"], y, decimal=5)
