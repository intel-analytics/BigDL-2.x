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


class MaxPooling1D(ZooKeras2Layer):
    """
    Max pooling operation for temporal data.

    # Arguments
        pool_size: Integer, size of the max pooling windows.
        strides: Integer, or None. Factor by which to downscale.
            E.g. 2 will halve the input.
            If None, it will be set to -1, which will be default to pool_size.
        padding: One of `"valid"` or `"same"` (case-insensitive).

    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.

    # Output shape
        3D tensor with shape: `(batch_size, downsampled_steps, features)`.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    >>> maxpooling1d = MaxPooling1D(3, input_shape=(3, 24))
    creating: createZooKeras2MaxPooling1D
    """
    def __init__(self,
                 pool_size=2,
                 strides=None,
                 padding="valid",
                 input_shape=None, **kwargs):
        if not strides:
            strides = -1
        super(MaxPooling1D, self).__init__(None,
                                           pool_size,
                                           strides,
                                           padding,
                                           list(input_shape) if input_shape else None,
                                           **kwargs)


class AveragePooling1D(ZooKeras2Layer):
    """
    Average pooling operation for temporal data.

    # Arguments
        pool_size: Integer, size of the average pooling windows.
        strides: Integer, or None. Factor by which to downscale.
            E.g. 2 will halve the input.
            If None, it will be set to -1, which will be default to pool_size.
        padding: One of `"valid"` or `"same"` (case-insensitive).

    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.

    # Output shape
        3D tensor with shape: `(batch_size, downsampled_steps, features)`.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    >>> averagepooling1d = AveragePooling1D(input_shape=(3, 24))
    creating: createZooKeras2AveragePooling1D
    """
    def __init__(self,
                 pool_size=2,
                 strides=None,
                 padding="valid",
                 input_shape=None, **kwargs):
        if not strides:
            strides = -1
        super(AveragePooling1D, self).__init__(None,
                                               pool_size,
                                               strides,
                                               padding,
                                               list(input_shape) if input_shape else None,
                                               **kwargs)


class GlobalAveragePooling1D(ZooKeras2Layer):

    """
    Applies global average pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalaveragepooling1d = GlobalAveragePooling1D(input_shape=(12, 12))
    creating: createZooKeras2GlobalAveragePooling1D
    """
    def __init__(self,
                 input_shape=None,
                 **kwargs):

        super(GlobalAveragePooling1D, self).__init__(None,
                                                     list(input_shape) if input_shape else None,
                                                     **kwargs)


class GlobalMaxPooling1D(ZooKeras2Layer):
    """
    Applies global max pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension) .

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalmaxpooling1d = GlobalMaxPooling1D(input_shape=(4, 8))
    creating: createZooKeras2GlobalMaxPooling1D
    """

    def __init__(self, input_shape=None, **kwargs):
        super(GlobalMaxPooling1D, self).__init__(None,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)


class GlobalAveragePooling2D(ZooKeras2Layer):
    """
    Applies global average pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    data_format: Format of input data. Either channels_first  or channels_last.

    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalaveragepooling2d = GlobalAveragePooling2D(input_shape=(4, 32, 32))
    creating: createZooKeras2GlobalAveragePooling2D
    """
    def __init__(self,
                 data_format="channels_first",
                 input_shape=None,
                 **kwargs):
        super(GlobalAveragePooling2D, self).__init__(None,
                                                     data_format,
                                                     list(input_shape) if input_shape else None,
                                                     **kwargs)
