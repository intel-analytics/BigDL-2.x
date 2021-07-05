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
import sys
from zoo.pipeline.api.keras2.base import ZooKeras2Layer

if sys.version >= '3':
    long = int
    unicode = str


class Maximum(ZooKeras2Layer):
    """
    Layer that computes the maximum (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    >>> from zoo.pipeline.api.keras.layers import Input
    >>> max = Maximum()([Input(shape=(4, 5)), Input(shape=(4, 5))])
    creating: createZooKeras2Maximum
    creating: createZooKerasInput
    creating: createZooKerasInput
    """
    def __init__(self,
                 input_shape=None, **kwargs):
        super(Maximum, self).__init__(None,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


def maximum(inputs, **kwargs):
    """Functional interface to the `Maximum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise maximum of the inputs.
    >>> from zoo.pipeline.api.keras.layers import Input
    >>> max = maximum([Input(shape=(4, 5)), Input(shape=(4, 5))])
    creating: createZooKerasInput
    creating: createZooKerasInput
    creating: createZooKeras2Maximum
    """
    return Maximum(**kwargs)(inputs)


class Minimum(ZooKeras2Layer):
    """
    Layer that computes the minimum (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    >>> from zoo.pipeline.api.keras.layers import Input
    >>> max = Minimum()([Input(shape=(4, 5)), Input(shape=(4, 5))])
    creating: createZooKeras2Minimum
    creating: createZooKerasInput
    creating: createZooKerasInput
    """
    def __init__(self,
                 input_shape=None, **kwargs):
        super(Minimum, self).__init__(None,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


def minimum(inputs, **kwargs):
    """Functional interface to the `Minimum` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise minimum of the inputs.
    >>> from zoo.pipeline.api.keras.layers import Input
    >>> min = minimum([Input(shape=(4, 5)), Input(shape=(4, 5))])
    creating: createZooKerasInput
    creating: createZooKerasInput
    creating: createZooKeras2Minimum
    """
    return Minimum(**kwargs)(inputs)


class Average(ZooKeras2Layer):
    """
    Layer that computes the average (element-wise) a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    >>> from zoo.pipeline.api.keras.layers import Input
    >>> ave = Average()([Input(shape=(4, 5)), Input(shape=(4, 5)), Input(shape=(4, 5))])
    creating: createZooKeras2Average
    creating: createZooKerasInput
    creating: createZooKerasInput
    creating: createZooKerasInput
    """
    def __init__(self,
                 input_shape=None, **kwargs):
        super(Average, self).__init__(None,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


def average(inputs, **kwargs):
    """Functional interface to the `Average` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise minimum of the inputs.
    >>> from zoo.pipeline.api.keras.layers import Input
    >>> ave = average([Input(shape=(4, 5)), Input(shape=(4, 5)), Input(shape=(4, 5))])
    creating: createZooKerasInput
    creating: createZooKerasInput
    creating: createZooKerasInput
    creating: createZooKeras2Average
    """
    return Average(**kwargs)(inputs)
