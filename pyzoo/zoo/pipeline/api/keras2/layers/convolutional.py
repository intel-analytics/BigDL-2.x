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


class Conv1D(ZooKeras2Layer):
    """1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
            `"valid"` means "no padding".
            `"same"` results in padding the input such that
            the output has the same length as the original input.
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.

    >>> conv1d = Conv1D(12, 4, input_shape=(3, 16))
    creating: createZooKeras2Conv1D
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding="valid",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zero",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 input_shape=None, **kwargs):
        super(Conv1D, self).__init__(None,
                                     filters,
                                     kernel_size,
                                     strides,
                                     padding,
                                     activation,
                                     use_bias,
                                     kernel_initializer,
                                     bias_initializer,
                                     kernel_regularizer,
                                     bias_regularizer,
                                     list(input_shape) if input_shape else None,
                                     **kwargs)


class Conv2D(ZooKeras2Layer):
    """2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
            Note that `"same"` is slightly inconsistent across backends with
            `strides` != 1.
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
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.

    >>> conv2d = Conv2D(12, kernel_size=(2, 5), input_shape=(3, 16, 16))
    creating: createZooKeras2Conv2D
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 data_format="channels_first",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zero",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 input_shape=None, **kwargs):
        super(Conv2D, self).__init__(None,
                                     filters,
                                     kernel_size,
                                     strides,
                                     padding,
                                     data_format,
                                     activation,
                                     use_bias,
                                     kernel_initializer,
                                     bias_initializer,
                                     kernel_regularizer,
                                     bias_regularizer,
                                     list(input_shape) if input_shape else None,
                                     **kwargs)


class Cropping1D(ZooKeras2Layer):
    """
    Cropping layer for 1D input (e.g. temporal sequence).
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    cropping: Int tuple of length 2. How many units should be trimmed off at the beginning and
              end of the cropping dimension. Default is (1, 1).
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> cropping1d = Cropping1D(cropping=(1, 2), input_shape=(8, 8))
    creating: createZooKeras2Cropping1D
    """
    def __init__(self, cropping=(1, 1), input_shape=None, **kwargs):
        super(Cropping1D, self).__init__(None,
                                         cropping,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)
