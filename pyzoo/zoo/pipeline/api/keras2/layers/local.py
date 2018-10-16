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


class LocallyConnected1D(ZooKeras2Layer):
    """
    Locally-connected layer for 1D inputs which works similarly to the TemporalConvolution
    layer, except that weights are unshared, that is, a different set of filters is applied
    at each different patch of the input..
    Padding currently supported for this layer is 'valid'.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    filters: Dimensionality of the output.
    kernel_size: The extension (spatial or temporal) of each filter.
    strides: Factor by which to subsample output. Int. Default is 1.
    padding: Only 'valid' is supported for now.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    kernel_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    bias_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    use_bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> locallyconnected1d = LocallyConnected1D(6, 3, input_shape=(8, 12))
    creating: createZooKeras2LocallyConnected1D
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding="valid",
                 activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 input_shape=None,
                 **kwargs):
        if padding != "valid":
            raise ValueError("For LocallyConnected1D, "
                             "only padding='valid' is supported for now")
        super(LocallyConnected1D, self).__init__(None,
                                                 filters,
                                                 kernel_size,
                                                 strides,
                                                 padding,
                                                 activation,
                                                 kernel_regularizer,
                                                 bias_regularizer,
                                                 use_bias,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)
