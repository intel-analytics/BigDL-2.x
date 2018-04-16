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

# Torch layers in Keras style.

import sys

from ..engine.topology import ZooKerasLayer

if sys.version >= '3':
    long = int
    unicode = str


class Select(ZooKerasLayer):
    """
    Select an index of the input in the given dim and return the subset part.
    The batch dimension needs to be unchanged.
    For example, if input is: [[1 2 3], [4 5 6]]
    Select(1, 1) will give output [2 5]
    Select(1, -1) will give output [3 6]

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim: The dimension to select. 0-based index. Cannot select the batch dimension.
         -1 means the last dimension of the input.
    index: The index of the dimension to be selected. 0-based index.
           -1 means the last dimension of the input.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> select = Select(0, -1, input_shape=(3, 4), name="select1")
    creating: createZooKerasSelect
    """
    def __init__(self, dim, index, input_shape=None, **kwargs):
        super(Select, self).__init__(None,
                                     dim,
                                     index,
                                     list(input_shape) if input_shape else None,
                                     **kwargs)


class Narrow(ZooKerasLayer):
    """
    Narrow the input with the number of dimensions not being reduced.
    The batch dimension needs to be unchanged.
    For example, if input is: [[1 2 3], [4 5 6]]
    Narrow(1, 1, 2) will give output [[2 3], [5 6]]
    Narrow(1, 2, -1) will give output [[3], [6]]

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim: The dimension to narrow. 0-based index. Cannot narrow the batch dimension.
         -1 means the last dimension of the input.
    offset: Non-negative integer. The start index on the given dimension. 0-based index.
    length: The length to narrow. Default is 1.
            Can use a negative length such as -1 in the case where input size is unknown.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> narrow = Narrow(1, 3, input_shape=(5, 6, 7), name="narrow1")
    creating: createZooKerasNarrow
    """
    def __init__(self, dim, offset, length=1, input_shape=None, **kwargs):
        super(Narrow, self).__init__(None,
                                     dim,
                                     offset,
                                     length,
                                     list(input_shape) if input_shape else None,
                                     **kwargs)


class Squeeze(ZooKerasLayer):
    """
    Delete the singleton dimension(s).
    The batch dimension needs to be unchanged.
    For example, if input has size (2, 1, 3, 4, 1):
    Squeeze(1) will give output size (2, 3, 4, 1)
    Squeeze() will give output size (2, 3, 4)

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim: The dimension(s) to squeeze. Can be either int or tuple of int.
         0-based index. Cannot squeeze the batch dimension.
         The selected dimensions must be singleton, i.e. having size 1.
         Default is None, and in this case all the non-batch singleton dimensions will be deleted.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> squeeze1 = Squeeze(1, input_shape=(1, 4, 5))
    creating: createZooKerasSqueeze
    >>> squeeze2 = Squeeze(input_shape=(1, 8, 1, 4))
    creating: createZooKerasSqueeze
    >>> squeeze3 = Squeeze((1, 2), input_shape=(1, 1, 1, 32))
    creating: createZooKerasSqueeze
    """
    def __init__(self, dim=None, input_shape=None, **kwargs):
        if isinstance(dim, int):
            dim = (dim, )
        super(Squeeze, self).__init__(None,
                                      dim,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class AddConstant(ZooKerasLayer):
    """
    Add a (non-learnable) scalar constant to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    constant: The scalar constant to be added.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> addconstant = AddConstant(1, input_shape=(1, 4, 5))
    creating: createZooKerasAddConstant
    """
    def __init__(self, constant, input_shape=None, **kwargs):
        super(AddConstant, self).__init__(None,
                                          float(constant),
                                          list(input_shape) if input_shape else None,
                                          **kwargs)


class MulConstant(ZooKerasLayer):
    """
    Multiply the input by a (non-learnable) scalar constant.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    constant: The scalar constant to be multiplied.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> mulconstant = MulConstant(2.2, input_shape=(3, 4))
    creating: createZooKerasMulConstant
    """
    def __init__(self, constant, input_shape=None, **kwargs):
        super(MulConstant, self).__init__(None,
                                          float(constant),
                                          list(input_shape) if input_shape else None,
                                          **kwargs)


class LRN2D(ZooKerasLayer):
    """
    Local Response Normalization between different feature maps.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    alpha: Float. The scaling parameter. Default is 0.0001.
    k: Float. A constant.
    beta: Float. The exponent. Default is 0.75.
    n: The number of channels to sum over.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> lrn2d = LRN2D(1e-3, 1.2, 0.4, 4, dim_ordering="tf", input_shape=(4, 5, 6))
    creating: createZooKerasLRN2D
    """
    def __init__(self, alpha=1e-4, k=1.0, beta=0.75, n=5,
                 dim_ordering="th", input_shape=None, **kwargs):
        super(LRN2D, self).__init__(None,
                                    float(alpha),
                                    float(k),
                                    float(beta),
                                    n,
                                    dim_ordering,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class ShareConvolution2D(ZooKerasLayer):
    """
    Applies a 2D convolution over an input image composed of several input planes.
    You can also use ShareConv2D as an alias of this layer.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    pad_h: The additional zeros added to the height dimension. Default is 0.
    pad_w: The additional zeros added to the width dimension. Default is 0.
    propagate_back: Whether to propagate gradient back. Default is True.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> shareconv2d = ShareConvolution2D(32, 3, 4, activation="tanh", input_shape=(3, 128, 128))
    creating: createZooKerasShareConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, init="glorot_uniform",
                 activation=None, subsample=(1, 1), pad_h=0, pad_w=0, propagate_back=True,
                 dim_ordering="th", W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, **kwargs):
        super(ShareConvolution2D, self).__init__(None,
                                                 nb_filter,
                                                 nb_row,
                                                 nb_col,
                                                 init,
                                                 activation,
                                                 subsample,
                                                 pad_h,
                                                 pad_w,
                                                 propagate_back,
                                                 dim_ordering,
                                                 W_regularizer,
                                                 b_regularizer,
                                                 bias,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)


ShareConv2D = ShareConvolution2D
