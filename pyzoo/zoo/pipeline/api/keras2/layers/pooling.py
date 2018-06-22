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


class GlobalAveragePooling1D(ZooKeras2Layer):

    """Global average pooling operation for spatial data.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`

    >>> globalaveragepooling1d = GlobalAveragePooling1D(input_shape=(12, 12))
    creating: createZooKeras2GlobalAveragePooling1D
    """
    def __init__(self,
                 input_shape=None,
                 **kwargs):

        super(GlobalAveragePooling1D, self).__init__(None,
                                                     list(input_shape) if input_shape else None,
                                                     **kwargs)
