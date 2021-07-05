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
from zoo.pipeline.api.autograd import Parameter, Constant


class TestOperatorSpecial(ZooTestCase):
    def test_mean_1D(self):
        data = np.random.randn(2, )
        parameter = Parameter(shape=(2,), init_weight=data)
        out = autograd.mean(parameter, axis=0)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, data.mean(axis=0), decimal=5)

    def test_mean_2D(self):
        data = np.random.randn(2, 3)
        parameter = Parameter(shape=(2, 3), init_weight=data)
        out = autograd.mean(parameter, axis=0)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, data.mean(axis=0), decimal=5)

    def test_slice_1D(self):
        data = np.random.randn(4, )
        parameter = Parameter(shape=(4,), init_weight=data)
        out = parameter.slice(0, 0, 2)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, data[0:2], decimal=5)

    def test_slice_2D(self):
        data = np.random.randn(2, 3)
        parameter = Parameter(shape=(2, 3), init_weight=data)
        out = parameter.slice(0, 0, 2)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, data[0:2], decimal=5)

    def test_unsqueeze_1D(self):
        data = np.random.randn(4, )
        parameter = Parameter(shape=(4,), init_weight=data)
        out = autograd.expand_dims(parameter, axis=0)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, np.expand_dims(data, axis=0), decimal=5)

    def test_unsqueeze_2D(self):
        data = np.random.randn(2, 3)
        parameter = Parameter(shape=(2, 3), init_weight=data)
        out = autograd.expand_dims(parameter, axis=0)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, np.expand_dims(data, axis=0), decimal=5)

    def test_sum_1D(self):
        data = np.random.randn(2, )
        parameter = Parameter(shape=(2,), init_weight=data)
        out = autograd.sum(parameter, axis=0)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, data.sum(axis=0), decimal=5)

    def test_sum_2D(self):
        data = np.random.randn(2, 3)
        parameter = Parameter(shape=(2, 3), init_weight=data)
        out = autograd.sum(parameter, axis=0)
        model = Model(input=parameter, output=out)
        result = model.forward(data)
        np.testing.assert_almost_equal(result, data.sum(axis=0), decimal=5)
