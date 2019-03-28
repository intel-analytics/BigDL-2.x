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

import keras.backend as KK
import keras.layers as klayers
import numpy as np
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.autograd import *
import zoo.pipeline.api.autograd as A
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import Sequential, Model
from zoo.pipeline.api.utils import remove_batch

np.random.seed(1337)  # for reproducibility


class TestOperator(ZooTestCase):
    # shape including batch
    def compare_binary_op(self, kk_func, z_layer, shape, rtol=1e-5, atol=1e-5):
        x = klayers.Input(shape=shape[0][1:])
        y = klayers.Input(shape=shape[1][1:])

        batch = shape[0][0]

        kkresult = kk_func(x, y)
        x_value = np.random.uniform(0, 1, shape[0])
        y_value = np.random.uniform(0, 1, shape[1])

        k_grads = KK.get_session().run(KK.gradients(kkresult, [x, y]),
                                       feed_dict={x: x_value, y: y_value})
        k_output = KK.get_session().run(kkresult,
                                        feed_dict={x: x_value, y: y_value})
        inputs = [Input(s) for s in remove_batch(shape)]
        model = Model(inputs, z_layer(inputs))
        z_output = model.forward([x_value, y_value])
        grad_output = np.array(z_output)
        grad_output.fill(1.0)
        z_grads = model.backward([x_value, y_value], grad_output)

        # Check if the model can be forward/backward multiple times or not
        z_output2 = model.forward([x_value, y_value])
        z_grads2 = model.backward([x_value, y_value], grad_output)
        self.assert_allclose(z_output, z_output2, rtol, atol)
        [self.assert_allclose(z, k, rtol, atol) for (z, k) in zip(z_grads, z_grads2)]

        self.assert_allclose(z_output, k_output, rtol, atol)
        [self.assert_allclose(z, k, rtol, atol) for (z, k) in zip(z_grads, k_grads)]

    # shape including batch
    def compare_unary_op(self, kk_func, z_layer, shape, rtol=1e-5, atol=1e-5):
        x = klayers.Input(shape=shape[1:])

        batch = shape[0]

        kkresult = kk_func(x)
        x_value = np.random.uniform(0, 1, shape)

        k_grads = KK.get_session().run(KK.gradients(kkresult, x),
                                       feed_dict={x: x_value})
        k_output = KK.get_session().run(kkresult,
                                        feed_dict={x: x_value})
        model = Sequential()
        model.add(InputLayer(shape[1:]))
        model.add(z_layer)
        z_output = model.forward(x_value)
        grad_output = np.array(z_output)
        grad_output.fill(1.0)
        z_grad = model.backward(x_value, grad_output)

        z_output2 = model.forward(x_value)
        z_grad2 = model.backward(x_value, grad_output)
        self.assert_allclose(z_output, z_output2, rtol, atol)
        self.assert_allclose(z_grad, z_grad2, rtol, atol)

        self.assert_allclose(z_output, k_output, rtol, atol)
        self.assert_allclose(z_grad, k_grads[0], rtol, atol)

    def test_add(self):
        def z_add_func(x, y):
            return x + y

        def k_add_func(x, y):
            return x + y

        self.compare_binary_op(k_add_func,
                               Lambda(function=z_add_func), [[2, 3], [2, 3]])

    def test_add_constant(self):
        def z_add_func(x):
            return x + 3.0

        def k_add_func(x):
            return x + 3.0

        self.compare_unary_op(k_add_func,
                              Lambda(function=z_add_func), [2, 3])

    def test_radd_constant(self):
        def z_add_func(x):
            return 3.0 + x

        def k_add_func(x):
            return 3.0 + x

        self.compare_unary_op(k_add_func,
                              Lambda(function=z_add_func), [2, 3])

    def test_sub(self):
        def z_func(x, y):
            return x - y

        def k_func(x, y):
            return x - y

        self.compare_binary_op(k_func,
                               Lambda(function=z_func), [[2, 3], [2, 3]])

    def test_sub_constant(self):
        def z_func(x):
            return x - 3.0

        def k_func(x):
            return x - 3.0

        self.compare_unary_op(k_func,
                              Lambda(function=z_func), [2, 3])

    def test_rsub_constant(self):
        def z_func(x):
            return 3.0 - x

        def k_func(x):
            return 3.0 - x

        self.compare_unary_op(k_func,
                              Lambda(function=z_func), [2, 3])

    def test_div(self):
        def z_func(x, y):
            return x / y

        def k_func(x, y):
            return x / y

        self.compare_binary_op(k_func,
                               Lambda(function=z_func), [[2, 3], [2, 3]])

    def test_div_constant(self):
        def z_func(x):
            return x / 3.0

        def k_func(x):
            return x / 3.0

        self.compare_unary_op(k_func,
                              Lambda(function=z_func), [2, 3])

    def test_rdiv_constant(self):
        def z_func(x):
            return 3.0 / x

        def k_func(x):
            return 3.0 / x

        self.compare_unary_op(k_func,
                              Lambda(function=z_func), [2, 3])

    def test_mul(self):
        def z_func(x, y):
            return x * y

        def k_func(x, y):
            return x * y

        self.compare_binary_op(k_func,
                               Lambda(function=z_func), [[2, 3], [2, 3]])

    def test_mul_constant(self):
        def z_func(x):
            return x * 3.0

        def k_func(x):
            return x * 3.0

        self.compare_unary_op(k_func,
                              Lambda(function=z_func), [2, 3])

    def test_rmul_constant(self):
        def z_func(x):
            return 3.0 * x

        def k_func(x):
            return 3.0 * x

        self.compare_unary_op(k_func,
                              Lambda(function=z_func), [2, 3])

    def test_neg(self):
        def z_func(x):
            return - x

        def k_func(x):
            return - x

        self.compare_unary_op(k_func,
                              Lambda(function=z_func), [2, 3])

    def test_clip(self):
        def z_func(x):
            return clip(x, 0.5, 0.8)

        def k_func(x):
            return KK.clip(x, 0.5, 0.8)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_pow(self):
        def z_func(x):
            return pow(x, 3.0)

        def k_func(x):
            return KK.pow(x, 3.0)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_square(self):
        def z_func(x):
            return square(x)

        def k_func(x):
            return KK.square(x)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_square_as_first_layer(self):
        def z_func(x):
            return square(x)

        ll = Lambda(function=z_func, input_shape=[2, 3])
        seq = Sequential()
        seq.add(ll)
        result = seq.forward(np.ones([2, 3]))
        assert (result == np.ones([2, 3])).all()

    def test_expose_node(self):
        image_shape = [3, 16, 16]
        input_shape = [2] + image_shape
        input = Input(shape=input_shape, name="input1")

        def l1(x):
            x1 = x.index_select(1, 0)  # input is [B, 2, 3, 16, 16]
            x2 = x.index_select(1, 0)
            return abs(x1 - x2)

        output = Lambda(function=l1)(input)
        model = Model(input, output)

        mock_data = np.random.uniform(0, 1, [10] + input_shape)
        out_data = model.forward(mock_data)
        assert out_data.shape == (10, 3, 16, 16)

    def test_softsign(self):
        def z_func(x):
            return softsign(x)

        def k_func(x):
            return KK.softsign(x)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_softplus(self):
        def z_func(x):
            return softplus(x)

        def k_func(x):
            return KK.softplus(x)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_exp(self):
        def z_func(x):
            return exp(x)

        def k_func(x):
            return KK.exp(x)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_abs(self):
        def z_func(x):
            return abs(x)

        def k_func(x):
            return KK.abs(x)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_log(self):
        def z_func(x):
            return log(x)

        def k_func(x):
            return KK.log(x)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_sqrt(self):
        def z_func(x):
            return sqrt(x)

        def k_func(x):
            return KK.sqrt(x)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_mean(self):
        def z_func(x):
            return mean(x, 0, False)

        def k_func(x):
            return KK.mean(x, 0, False)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_sum(self):
        def z_func(x):
            return sum(x, 0, False)

        def k_func(x):
            return KK.sum(x, 0, False)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3])

    def test_maximum(self):
        def z_func(x, y):
            return maximum(x, y)

        def k_func(x, y):
            return KK.maximum(x, y)

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[2, 3], [2, 3]])

    def test_expand_dim1(self):
        def z_func(x):
            return expand_dims(x, 1)

        def k_func(x):
            return KK.expand_dims(x, 1)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3, 4])

    def test_expand_dim2(self):
        def z_func(x):
            return expand_dims(x, 2)

        def k_func(x):
            return KK.expand_dims(x, 2)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [2, 3, 4, 5])

    def test_stack(self):
        def z_func(x, y):
            return stack([x, y], axis=1)

        def k_func(x, y):
            return KK.stack([x, y], axis=1)

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2, 4], [3, 2, 4]])

    def test_stack2(self):
        def z_func(x, y):
            return stack([x, y], axis=2)

        def k_func(x, y):
            return KK.stack([x, y], axis=2)

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2, 4], [3, 2, 4]])

    def test_slice(self):
        def z_func(x):
            return x.slice(1, 1, 2)

        def k_func(x):
            return x[:, 1:3, :]

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [3, 5, 4])

    def test_slice2(self):
        def z_func(x):
            return x.slice(2, 1, 2)

        def k_func(x):
            return x[:, :, 1:3]

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [3, 2, 4])

    def test_index_select(self):
        def z_func(x):
            return x.index_select(2, 2)

        def k_func(x):
            return x[:, :, 2]

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [3, 2, 4])

    def test_dot_2D(self):
        def z_func(x, y):
            return A.batch_dot(x, y, axes=1, normalize=False)

        def k_func(x, y):
            return klayers.Dot(axes=[1, 1], normalize=False)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2], [3, 2]])

    def test_dot_3D_2(self):
        def z_func(x, y):
            return A.batch_dot(x, y, axes=2, normalize=False)

        def k_func(x, y):
            return klayers.Dot(axes=2, normalize=False)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2, 4], [3, 2, 4]])

    def test_dot_3D_1(self):
        def z_func(x, y):
            return A.batch_dot(x, y, axes=1, normalize=False)

        def k_func(x, y):
            return klayers.Dot(axes=1, normalize=False)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2, 4], [3, 2, 4]])

    def test_dot_3D_1_2(self):
        def z_func(x, y):
            return A.batch_dot(x, y, axes=[1, 2], normalize=False)

        def k_func(x, y):
            return klayers.Dot(axes=[1, 2], normalize=False)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2, 4], [3, 4, 2]])

    def test_dot_3D_2_2_norm(self):
        def z_func(x, y):
            return A.batch_dot(x, y, axes=[2, 2], normalize=True)

        def k_func(x, y):
            return klayers.Dot(axes=[2, 2], normalize=True)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2, 4], [3, 2, 4]])

    def test_dot_3D_1_2_norm(self):
        def z_func(x, y):
            return A.batch_dot(x, y, axes=[1, 2], normalize=True)

        def k_func(x, y):
            return klayers.Dot(axes=[1, 2], normalize=True)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[3, 2, 4], [3, 4, 2]])

    def test_l2_normalize(self):
        def z_func(x):
            return A.l2_normalize(x, axis=1)

        def k_func(x):
            return KK.l2_normalize(x, axis=1)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [3, 2])

    def test_l2_normalize_2(self):
        def z_func(x):
            return A.l2_normalize(x, axis=2)

        def k_func(x):
            return KK.l2_normalize(x, axis=2)

        self.compare_unary_op(k_func,
                              Lambda(function=z_func, ), [3, 2, 4])

    def test_mm(self):
        def z_func(x, y):
            return A.mm(x, y, axes=[2, 2])

        def k_func(x, y):
            return klayers.Dot(axes=[2, 2], normalize=False)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[2, 3, 4], [2, 3, 4]])

    def test_mm2(self):
        def z_func(x, y):
            return A.mm(x, y, axes=[1, 1])

        def k_func(x, y):
            return klayers.Dot(axes=[1, 1], normalize=False)([x, y])

        self.compare_binary_op(k_func,
                               Lambda(function=z_func, ), [[2, 3, 4], [2, 3, 4]])

    def test_ExpandDim(self):
        inputdata = np.array([2, 1, 6])
        input = Parameter(shape=(3,), init_weight=inputdata)
        expand = ExpandDim(dim=0)(input)
        model = Model(input, expand)
        assert model.get_output_shape() == (1, 3)
        desired = inputdata.reshape(1, 3)
        outputdata = model.forward(inputdata)
        np.testing.assert_almost_equal(outputdata, desired, decimal=4)


if __name__ == "__main__":
    pytest.main([__file__])
