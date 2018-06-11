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
