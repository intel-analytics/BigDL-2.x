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

import zoo.pipeline.api.keras.layers as klayers1
from zoo.pipeline.api.keras2.base import ZooKeras2Layer

if sys.version >= '3':
    long = int
    unicode = str


class Dense(ZooKeras2Layer):
    """
    A densely-connected NN layer.
    The most common input is 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    units: The size of output dimension.
    kernel_initializer: String representation of the initialization method \
          for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    kernel_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias_regularizer: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_dim: Dimensionality of the input for 2D input. For nD input, you can alternatively
               specify 'input_shape' when using this layer as the first layer.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> dense = Dense(10, kernel_initializer="glorot_uniform", input_dim=8, name="dense1")
    creating: createZooKeras2Dense
    """
    def __init__(self, units, kernel_initializer="glorot_uniform",
                 bias_initializer="zero", activation=None,
                 kernel_regularizer=None, bias_regularizer=None,
                 use_bias=True, input_dim=None, input_shape=None, **kwargs):
        if input_dim:
            input_shape = (input_dim, )
        super(Dense, self).__init__(None,
                                    units,
                                    kernel_initializer,
                                    bias_initializer,
                                    activation,
                                    kernel_regularizer,
                                    bias_regularizer,
                                    use_bias,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class Activation(ZooKeras2Layer):
    """
    Simple activation function to be applied to the output.
    Available activations: 'tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
                           'hard_sigmoid', 'linear', 'relu6', 'tanh_shrink', 'softmin',
                           'log_sigmoid' and 'log_softmax'.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    activation: Name of the activation function as string.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> activation = Activation("relu", input_shape=(3, 4))
    creating: createZooKeras2Activation
    """
    def __init__(self,
                 activation,
                 input_shape=None,
                 **kwargs):
        super(Activation, self).__init__(None,
                                         activation,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class Dropout(ZooKeras2Layer):
    """
    Applies Dropout to the input by randomly setting a fraction 'rate' of input units to 0 at each
    update during training time in order to prevent overfitting.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    rate: Fraction of the input units to drop. Float between 0 and 1.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> dropout = Dropout(0.25, input_shape=(2, 3))
    creating: createZooKeras2Dropout
    """
    def __init__(self,
                 rate,
                 input_shape=None,
                 **kwargs):
        super(Dropout, self).__init__(None,
                                      float(rate),
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class Flatten(ZooKeras2Layer):
    """
    Flattens the input without affecting the batch size.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> flatten = Flatten(input_shape=(3, 10, 2))
    creating: createZooKeras2Flatten
    """
    def __init__(self,
                 input_shape=None,
                 **kwargs):
        super(Flatten, self).__init__(None,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)
