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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
import zoo.pipeline.api.autograd as auto

np.random.seed(1337)  # for reproducibility


class TestAutoGradModel(ZooTestCase):
    def test_var(self):
        input = Input(shape=[2, 20])
        time = TimeDistributed(layer=Dense(30))(input)
        t1 = time.index_select(1, 0)
        t2 = time.index_select(1, 1)
        diff = auto.abs(t1 - t2)
        assert diff.get_output_shape() == (None, 30)
        assert diff.get_input_shape() == (None, 30)
        model = Model(input, diff)
        data = np.random.uniform(0, 1, [10, 2, 20])
        output = model.forward(data)
        print(output.shape)

    def test_var_model(self):
        input = Input(shape=[2, 3, 16, 16])

        vgg_mock = Sequential()
        vgg_mock.add(Conv2D(2, 4, 4, input_shape=[3, 16, 16]))  # output: 2, 13, 13
        vgg_mock.add(Reshape([2 * 13 * 13]))
        vgg_mock.add(Dense(100))
        vgg_mock.add(Reshape([100, 1, 1]))

        time = TimeDistributed(layer=vgg_mock)(input)
        t1 = time.index_select(1, 0)
        t2 = time.index_select(1, 1)
        diff = t1 - t2
        model = Model(input, diff)
        data = np.random.uniform(0, 1, [10, 2, 3, 16, 16])
        output = model.forward(data)
        print(output.shape)

    def test_zoo_keras_layer_of(self):
        input = Input(shape=[2, 3])
        dense = Dense(3)
        ZooKerasLayer.of(dense.value)(input)

    def test_parameter_create(self):
        w = auto.Parameter(input_shape=(3, 2))
        value = w.get_weight()
        w.set_weight(value)
        x = auto.Variable(input_shape=(3,))
        b = auto.Parameter(input_shape=(2,))
        out = auto.mm(x, w, axes=(1, 0)) + b
        model = Model(input=x, output=out)
        input_data = np.random.uniform(0, 1, (4, 3))
        model.forward(input_data)
