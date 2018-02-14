#
# Copyright 2016 The BigDL Authors.
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

from bigdl.nn.layer import Layer, Sequential as TSequential, Model as TModel
from bigdl.util.common import callBigDlFunc, JTensor, JavaValue, to_list

if sys.version >= '3':
    long = int
    unicode = str


class InferShape(JavaValue):
    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type

    @classmethod
    def __to_keras_shape(cls, shape):
        return tuple([None] + shape[1:])

    def __process_shape(self, shape):
        if len(shape) == 1:
            return self.__to_keras_shape(shape[0])
        else:
            return [self.__to_keras_shape(s) for s in shape]

    def get_input_shape(self):
        """
        Return a list of shape tuples if there are multiple inputs.
        Return one shape tuple otherwise.
        """
        input = callBigDlFunc(self.bigdl_type, "getInputShape",
                              self.value)
        return self.__process_shape(input)

    def get_output_shape(self):
        """
        Return a list of shape tuples if there are multiple outputs.
        Return one shape tuple otherwise.
        """
        output = callBigDlFunc(self.bigdl_type, "getOutputShape",
                               self.value)
        return self.__process_shape(output)


class KerasLayer(Layer):
    def jvm_class_constructor(self):
        name = "createKeras" + self.__class__.__name__
        print("creating: " + name)
        return name


class Sequential(TSequential, InferShape):
    """
    Container for a Sequential model.

    >>> sequential = Sequential()
    creating: createSequential
    """
    def __init__(self, bigdl_type="float"):
        super(Sequential, self).__init__(bigdl_type, True)


class Model(TModel, InferShape):
    def __init__(self, input, output, bigdl_type="float"):
        super(Model, self).__init__(to_list(input),
                                    to_list(output),
                                    is_keras=True,
                                    bigdl_type=bigdl_type)


class Input(KerasLayer):
    def __init__(self, name=None, input_shape=None, bigdl_type="float"):
        super(Input, self).__init__(None, bigdl_type,
                                    name,
                                    list(input_shape) if input_shape else None)


class InputLayer(KerasLayer):
    """
    Layer to be used as an entry point into a model.

    # Arguments
    input_shape: A shape tuple, not including the batch axis.

    >>> inputlayer = InputLayer(input_shape=(3, 5))
    creating: createKerasInputLayer
    """
    def __init__(self, input_shape=None, bigdl_type="float"):
        super(InputLayer, self).__init__(None, bigdl_type,
                                         list(input_shape) if input_shape else None)


class Dense(KerasLayer):
    """
    A densely-connected NN layer.
    The most common input is 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: The size of output dimension.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear). Default is True.
    input_shape: A shape tuple, not including batch.

    >>> dense = Dense(10, input_shape=(3, 4))
    creating: createKerasDense
    """
    def __init__(self, output_dim, init="glorot_uniform", activation=None,
                 W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, bigdl_type="float"):
        super(Dense, self).__init__(None, bigdl_type,
                                    output_dim,
                                    init,
                                    activation,
                                    W_regularizer,
                                    b_regularizer,
                                    bias,
                                    list(input_shape) if input_shape else None)


class MaxoutDense(KerasLayer):
    """
    A dense maxout layer that takes the element-wise maximum of nbFeature, Dense(inputDim, outputDim) linear layers.
    This allows the layer to learn a convex, piecewise linear activation function over the inputs.
    The input of this layer should be 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: The size of output dimension.
    nb_feature: Number of Dense layers to use internally. Int. Default is 4.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear). Default is True.
    input_shape: A shape tuple, not including batch.

    >>> maxoutdense = MaxoutDense(6, input_shape=(10, ))
    creating: createKerasMaxoutDense
    """
    def __init__(self, output_dim, nb_feature=4, W_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, bigdl_type="float"):
        super(MaxoutDense, self).__init__(None, bigdl_type,
                                          output_dim,
                                          nb_feature,
                                          W_regularizer,
                                          b_regularizer,
                                          bias,
                                          list(input_shape) if input_shape else None)


class Embedding(KerasLayer):
    """
    Turn positive integers (indexes) into dense vectors of fixed size.
    The input of this layer should be 2D.

    This layer can only be used as the first layer in a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_dim: Size of the vocabulary. Int > 0.
    output_dim: Dimension of the dense embedding. Int >= 0.
    init: String representations of initialization method for the weights of the layer.
          Default is 'uniform'.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the embedding matrix. Default is None.
    input_shape: A shape tuple, not including batch.

    >>> embedding = Embedding(1000, 32, input_shape=(10, ))
    creating: createKerasEmbedding
    """
    def __init__(self, input_dim, output_dim, init="uniform",
                 W_regularizer=None, input_shape=None, bigdl_type="float"):
        super(Embedding, self).__init__(None, bigdl_type,
                                        input_dim,
                                        output_dim,
                                        init,
                                        W_regularizer,
                                        list(input_shape) if input_shape else None)


class BatchNormalization(KerasLayer):
    """
    Batch normalization layer.
    Normalize the activations of the previous layer at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation standard deviation close to 1.
    It is a feature-wise normalization, each feature map in the input will be normalized separately.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    epsilon: Small float > 0. Fuzz parameter. Default is 0.001.
    momentum: Float. Momentum in the computation of the exponential average of the mean and
              standard deviation of the data, for feature-wise normalization. Default is 0.99.
    beta_init: Name of initialization function for shift parameter. Default is 'zero'.
    gamma_init: Name of initialization function for scale parameter. Default is 'one'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
                  For 'th', axis along which to normalize is 1. For 'tf', axis is 3.
    input_shape: A shape tuple, not including batch.

    >>> batchnormalization = BatchNormalization(input_shape=(3, 12, 12))
    creating: createKerasBatchNormalization
    """
    def __init__(self, epsilon=0.001, momentum=0.99, beta_init="zero", gamma_init="one",
                 dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
                                                 float(epsilon),
                                                 float(momentum),
                                                 beta_init,
                                                 gamma_init,
                                                 dim_ordering,
                                                 list(input_shape) if input_shape else None)

    def set_running_mean(self, running_mean):
        callBigDlFunc(self.bigdl_type, "setKerasRunningMean",
                      self.value, JTensor.from_ndarray(running_mean))
        return self

    def set_running_std(self, running_std):
        callBigDlFunc(self.bigdl_type, "setKerasRunningStd",
                      self.value, JTensor.from_ndarray(running_std))
        return self

    def get_running_mean(self):
        return callBigDlFunc(self.bigdl_type, "getKerasRunningMean",
                             self.value).to_ndarray()

    def get_running_std(self):
        return callBigDlFunc(self.bigdl_type, "getKerasRunningStd",
                             self.value).to_ndarray()


class Merge(KerasLayer):
    """
    Used to merge a list of tensors into a single tensor, following some merge mode.
    Merge must have at least two input layers.

    When using this layer as the first layer in a model, you need to provide the argument
    input_shape for input layers (a list of shape tuples, does not include the batch dimension).

    # Arguments
    layers: A list of layer instances. Must be more than one layer.
    mode: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos',
          'dot', 'max'. Default is 'sum'.
    concat_axis: Int, axis to use in mode concat. Only specify this when mode is 'concat'.
                 Default is -1, meaning the last axis of the input.
    input_shape: A list of shape tuples, each not including batch.

    >>> l1 = InputLayer(input_shape=(3, 5))
    creating: createKerasInputLayer
    >>> l2 = InputLayer(input_shape=(3, 5))
    creating: createKerasInputLayer
    >>> merge = Merge(layers=[l1, l2], mode='sum')
    creating: createKerasMerge
    """
    def __init__(self, layers=None, mode="sum", concat_axis=-1,
                 input_shape=None, bigdl_type="float"):
        super(Merge, self).__init__(None, bigdl_type,
                                    list(layers) if layers else None,
                                    mode,
                                    concat_axis,
                                    input_shape)


class Dropout(KerasLayer):
    """
    Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each
    update during training time in order to prevent overfitting.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    input_shape: A shape tuple, not including batch.

    >>> dropout = Dropout(0.25, input_shape=(2, 3))
    creating: createKerasDropout
    """
    def __init__(self, p, input_shape=None, bigdl_type="float"):
        super(Dropout, self).__init__(None, bigdl_type,
                                      float(p),
                                      list(input_shape) if input_shape else None)


class Flatten(KerasLayer):
    """
    Flattens the input without affecting the batch size.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.

    >>> flatten = Flatten(input_shape=(3, 10, 2))
    creating: createKerasFlatten
    """
    def __init__(self, input_shape=None, bigdl_type="float"):
        super(Flatten, self).__init__(None, bigdl_type,
                                      list(input_shape) if input_shape else None)


class Reshape(KerasLayer):
    """
    Reshapes an output to a certain shape.
    Supports shape inference by allowing one -1 in the target shape.
    For example, if input_shape = (2, 3, 4), target_shape = (3, -1),
    then output_shape will be (3, 8).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    target_shape: A shape tuple. The target shape that you desire to have. Batch dimension should be excluded.
    input_shape: A shape tuple, not including batch.

    >>> reshape = Reshape((2, 10), input_shape=(5, 4))
    creating: createKerasReshape
    """
    def __init__(self, target_shape, input_shape=None, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type,
                                      target_shape,
                                      list(input_shape) if input_shape else None)


class Activation(KerasLayer):
    """
    Simple activation function to be applied to the output.
    Available activations: 'tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'hard_sigmoid'.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    activation: Name of the activation function as string.
    input_shape: A shape tuple, not including batch.

    >>> activation = Activation("relu", input_shape=(3, 4))
    creating: createKerasActivation
    """
    def __init__(self, activation, input_shape=None, bigdl_type="float"):
        super(Activation, self).__init__(None, bigdl_type,
                                         activation,
                                         list(input_shape) if input_shape else None)


class RepeatVector(KerasLayer):
    """
    Repeats the input n times.
    The input of this layer should be 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    n: Repetition factor. Int.
    input_shape: A shape tuple, not including batch.

    >>> repeatvector = RepeatVector(5, input_shape=(3, ))
    creating: createKerasRepeatVector
    """
    def __init__(self, n, input_shape=None, bigdl_type="float"):
        super(RepeatVector, self).__init__(None, bigdl_type,
                                           n,
                                           list(input_shape) if input_shape else None)


class Permute(KerasLayer):
    """
    Permutes the dimensions of the input according to a given pattern.
    Useful for connecting RNNs and convnets together.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dims: Tuple of int. Permutation pattern, does not include the samples dimension. Indexing starts at 1.
    input_shape: A shape tuple, not including batch.

    >>> permute = Permute((2, 1, 3), input_shape=(3, 4, 5))
    creating: createKerasPermute
    """
    def __init__(self, dims, input_shape=None, bigdl_type="float"):
        super(Permute, self).__init__(None, bigdl_type,
                                      dims,
                                      list(input_shape) if input_shape else None)


class Highway(KerasLayer):
    """
    Densely connected highway network. Highway layers are a natural extension of LSTMs to feedforward networks.
    The input of this layer should be 2D, i.e. (batch, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> highway = Highway(activation='relu', input_shape=(8, ))
    creating: createKerasHighway
    """
    def __init__(self, activation=None, W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, bigdl_type="float"):
        super(Highway, self).__init__(None, bigdl_type,
                                      activation,
                                      W_regularizer,
                                      b_regularizer,
                                      bias,
                                      list(input_shape) if input_shape else None)


class Convolution1D(KerasLayer):
    """
    Applies convolution operator for filtering neighborhoods of 1D inputs.
    You can also use Conv1D as an alias of this layer.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    filter_length: The extension (spatial or temporal) of each filter.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample_length: Factor by which to subsample output. Int. Default is 1.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> conv1d = Convolution1D(12, 4, input_shape=(3, 16))
    creating: createKerasConvolution1D
    """
    def __init__(self, nb_filter, filter_length, init="glorot_uniform",
                 activation=None, border_mode="valid", subsample_length=1,
                 W_regularizer=None, b_regularizer=None, bias=True,
                 input_shape=None, bigdl_type="float"):
        super(Convolution1D, self).__init__(None, bigdl_type,
                                            nb_filter,
                                            filter_length,
                                            init,
                                            activation,
                                            border_mode,
                                            subsample_length,
                                            W_regularizer,
                                            b_regularizer,
                                            bias,
                                            list(input_shape) if input_shape else None)


class Convolution2D(KerasLayer):
    """
    Applies a 2D convolution over an input image composed of several input planes.
    You can also use Conv2D as an alias of this layer.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> conv2d = Convolution2D(32, 3, 3, input_shape=(3, 128, 128))
    creating: createKerasConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col,
                 init="glorot_uniform", activation=None,
                 border_mode="valid", subsample=(1, 1), dim_ordering="th",
                 W_regularizer=None, b_regularizer=None, bias=True,
                 input_shape=None, bigdl_type="float"):
        super(Convolution2D, self).__init__(None, bigdl_type,
                                            nb_filter,
                                            nb_row,
                                            nb_col,
                                            init,
                                            activation,
                                            border_mode,
                                            subsample,
                                            dim_ordering,
                                            W_regularizer,
                                            b_regularizer,
                                            bias,
                                            list(input_shape) if input_shape else None)


class Convolution3D(KerasLayer):
    """
    Applies convolution operator for filtering windows of three-dimensional inputs.
    You can also use Conv3D as an alias of this layer.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    kernel_dim1: Length of the first dimension in the convolution kernel.
    kernel_dim2: Length of the second dimension in the convolution kernel.
    kernel_dim3: Length of the third dimension in the convolution kernel.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Int tuple of length 3. Factor by which to subsample output.
               Also called strides elsewhere. Default is (1, 1, 1).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> conv3d = Convolution3D(32, 3, 4, 5, input_shape=(3, 64, 64, 64))
    creating: createKerasConvolution3D
    """
    def __init__(self, nb_filter, kernel_dim1, kernel_dim2, kernel_dim3,
                 init="glorot_uniform", activation=None, border_mode="valid",
                 subsample=(1, 1, 1), dim_ordering="th", W_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, bigdl_type="float"):
        super(Convolution3D, self).__init__(None, bigdl_type,
                                            nb_filter,
                                            kernel_dim1,
                                            kernel_dim2,
                                            kernel_dim3,
                                            init,
                                            activation,
                                            border_mode,
                                            subsample,
                                            dim_ordering,
                                            W_regularizer,
                                            b_regularizer,
                                            bias,
                                            list(input_shape) if input_shape else None)


class AtrousConvolution1D(KerasLayer):
    """
    Applies an atrous convolution operator for filtering neighborhoods of 1D inputs.
    A.k.a dilated convolution or convolution with holes.
    Border mode currently supported for this layer is 'valid'.
    Bias will be included in this layer.
    You can also use AtrousConv1D as an alias of this layer.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    filter_length: The extension (spatial or temporal) of each filter.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Only 'valid' is supported for now.
    subsample_length: Factor by which to subsample output. Int. Default is 1.
    atrous_rate: Factor for kernel dilation. Also called filter_dilation elsewhere. Int. Default is 1.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Only 'True' is supported for now.
    input_shape: A shape tuple, not including batch.

    >>> atrousconv1d = AtrousConvolution1D(8, 3, input_shape=(3, 12))
    creating: createKerasAtrousConvolution1D
    """
    def __init__(self, nb_filter, filter_length, init="glorot_uniform",
                 activation=None, border_mode='valid', subsample_length=1, atrous_rate=1,
                 W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, bigdl_type="float"):
        if border_mode != "valid":
            raise ValueError("For AtrousConvolution1D, only border_mode='valid' is supported for now")
        if not bias:
            raise ValueError("For AtrousConvolution1D, only bias=True is supported for now")
        super(AtrousConvolution1D, self).__init__(None, bigdl_type,
                                                  nb_filter,
                                                  filter_length,
                                                  init,
                                                  activation,
                                                  subsample_length,
                                                  atrous_rate,
                                                  W_regularizer,
                                                  b_regularizer,
                                                  list(input_shape) if input_shape else None)


class AtrousConvolution2D(KerasLayer):
    """
    Applies an atrous Convolution operator for filtering windows of 2D inputs.
    A.k.a dilated convolution or convolution with holes.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    Bias will be included in this layer.
    You can also use AtrousConv2D as an alias of this layer.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Only 'valid' is supported for now.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    atrous_rate: Int tuple of length 2. Factor for kernel dilation.
                 Also called filter_dilation elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Only 'True' is supported for now.
    input_shape: A shape tuple, not including batch.

    >>> atrousconv2d = AtrousConvolution2D(12, 4, 3, input_shape=(3, 64, 64))
    creating: createKerasAtrousConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, init="glorot_uniform",
                 activation=None, border_mode="valid", subsample=(1, 1),
                 atrous_rate=(1, 1), dim_ordering="th", W_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, bigdl_type="float"):
        if border_mode != "valid":
            raise ValueError("For AtrousConvolution2D, only border_mode='valid' is supported for now")
        if not bias:
            raise ValueError("For AtrousConvolution2D, only bias=True is supported for now")
        super(AtrousConvolution2D, self).__init__(None, bigdl_type,
                                                  nb_filter,
                                                  nb_row,
                                                  nb_col,
                                                  init,
                                                  activation,
                                                  subsample,
                                                  atrous_rate,
                                                  dim_ordering,
                                                  W_regularizer,
                                                  b_regularizer,
                                                  list(input_shape) if input_shape else None)


class Deconvolution2D(KerasLayer):
    """
    Transposed convolution operator for filtering windows of 2D inputs.
    The need for transposed convolutions generally arises from the desire to use a transformation
    going in the opposite direction of a normal convolution, i.e., from something that has
    the shape of the output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with said convolution.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    You can also use Deconv2D as an alias of this layer.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of transposed convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    output_shape: Output shape of the transposed convolution operation. Tuple of int.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Only 'valid' is supported for now.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> deconv2d = Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14), input_shape=(3, 12, 12))
    creating: createKerasDeconvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, output_shape, init="glorot_uniform",
                 activation=None, border_mode="valid", subsample=(1, 1), dim_ordering="th",
                 W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, bigdl_type="float"):
        if border_mode != "valid":
            raise ValueError("For Deconvolution2D, only border_mode='valid' is supported for now")
        super(Deconvolution2D, self).__init__(None, bigdl_type,
                                              nb_filter,
                                              nb_row,
                                              nb_col,
                                              init,
                                              activation,
                                              subsample,
                                              dim_ordering,
                                              W_regularizer,
                                              b_regularizer,
                                              bias,
                                              list(input_shape) if input_shape else None)


class SeparableConvolution2D(KerasLayer):
    """
    Applies separable convolution operator for 2D inputs.
    Separable convolutions consist in first performing a depthwise spatial convolution (which acts
    on each input channel separately) followed by a pointwise convolution which mixes together the
    resulting output channels. The depthMultiplier argument controls how many output channels are
    generated per input channel in the depthwise step.
    You can also use SeparableConv2D as an alias of this layer.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    depth_multiplier: How many output channel to use per input channel for the depthwise convolution step.
                      Int. Default is 1.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    depthwise_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                           applied to the depthwise weights matrices. Default is None.
    pointwise_regularizer: An instance of [[Regularizer]], applied to the pointwise weights matrices.
                           Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> separableconv2d = SeparableConvolution2D(12, 3, 4, input_shape=(3, 32, 32))
    creating: createKerasSeparableConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, init="glorot_uniform",
                 activation=None, border_mode="valid", subsample=(1, 1), depth_multiplier=1,
                 dim_ordering="th", depthwise_regularizer=None, pointwise_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, bigdl_type="float"):
        super(SeparableConvolution2D, self).__init__(None, bigdl_type,
                                                     nb_filter,
                                                     nb_row,
                                                     nb_col,
                                                     init,
                                                     activation,
                                                     border_mode,
                                                     subsample,
                                                     depth_multiplier,
                                                     dim_ordering,
                                                     depthwise_regularizer,
                                                     pointwise_regularizer,
                                                     b_regularizer,
                                                     bias,
                                                     list(input_shape) if input_shape else None)


Conv1D = Convolution1D
Conv2D = Convolution2D
Conv3D = Convolution3D
Deconv2D = Deconvolution2D
AtrousConv1D = AtrousConvolution1D
AtrousConv2D = AtrousConvolution2D
SeparableConv2D = SeparableConvolution2D


class Cropping1D(KerasLayer):
    """
    Cropping layer for 1D input (e.g. temporal sequence).
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    cropping: Int tuple of length 2. How many units should be trimmed off at the beginning and
              end of the cropping dimension. Default is (1, 1).
    input_shape: A shape tuple, not including batch.

    >>> cropping1d = Cropping1D(cropping=(1, 2), input_shape=(8, 8))
    creating: createKerasCropping1D
    """
    def __init__(self, cropping=(1, 1), input_shape=None, bigdl_type="float"):
        super(Cropping1D, self).__init__(None, bigdl_type,
                                         cropping,
                                         list(input_shape) if input_shape else None)


class Cropping2D(KerasLayer):
    """
    Cropping layer for 2D input (e.g. picture).
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    cropping: Int tuple of tuple of length 2. How many units should be trimmed off at the beginning and
              end of the 2 cropping dimensions (i.e. height and width). Default is ((0, 0), (0, 0)).
    input_shape: A shape tuple, not including batch.

    >>> cropping2d = Cropping2D(cropping=((1, 2), (0, 1)), input_shape=(12, 12, 12))
    creating: createKerasCropping2D
    """
    def __init__(self, cropping=((0, 0), (0, 0)), dim_ordering="th",
                 input_shape=None, bigdl_type="float"):
        super(Cropping2D, self).__init__(None, bigdl_type,
                                         cropping[0],
                                         cropping[1],
                                         dim_ordering,
                                         list(input_shape) if input_shape else None)


class Cropping3D(KerasLayer):
    """
    Cropping layer for 3D data (e.g. spatial or spatio-temporal).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    cropping: Int tuple of tuple of length 3. How many units should be trimmed off at the beginning and
              end of the 3 cropping dimensions (i.e. kernel_dim1, kernel_dim2 and kernel_dim3).
              Default is ((1, 1), (1, 1), (1, 1)).
    input_shape: A shape tuple, not including batch.

    >>> cropping3d = Cropping3D(cropping=((0, 2), (1, 1), (3, 1)), input_shape=(4, 12, 12, 16))
    creating: createKerasCropping3D
    """
    def __init__(self, cropping=((1, 1), (1, 1), (1, 1)), dim_ordering="th",
                 input_shape=None, bigdl_type="float"):
        super(Cropping3D, self).__init__(None, bigdl_type,
                                         cropping[0],
                                         cropping[1],
                                         cropping[2],
                                         dim_ordering,
                                         list(input_shape) if input_shape else None)


class UpSampling1D(KerasLayer):
    """
    UpSampling layer for 1D inputs.
    Repeats each temporal step 'length' times along the time axis.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    length: Int. UpSampling factor. Default is 2.
    input_shape: A shape tuple, not including batch.

    >>> upsampling1d = UpSampling1D(length=3, input_shape=(3, 12))
    creating: createKerasUpSampling1D
    """
    def __init__(self, length=2, input_shape=None, bigdl_type="float"):
        super(UpSampling1D, self).__init__(None, bigdl_type,
                                           length,
                                           list(input_shape) if input_shape else None)


class UpSampling2D(KerasLayer):
    """
    UpSampling layer for 2D inputs.
    Repeats the rows and columns of the data by size[0] and size[1] respectively.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: Int tuple of length 2. UpSampling factors for rows and columns. Default is (2, 2).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> upsampling2d = UpSampling2D(size=(1, 3), input_shape=(3, 16, 16))
    creating: createKerasUpSampling2D
    """
    def __init__(self, size=(2, 2), dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(UpSampling2D, self).__init__(None, bigdl_type,
                                           size,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None)


class UpSampling3D(KerasLayer):
    """
    UpSampling layer for 2D inputs.
    Repeats the 1st, 2nd and 3rd dimensions of the data by size[0], size[1] and size[2] respectively.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: Int tuple of length 3. UpSampling factors for dim1, dim2 and dim3. Default is (2, 2, 2).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.

    >>> upsampling3d = UpSampling3D(size=(1, 2, 3), input_shape=(3, 16, 16, 16))
    creating: createKerasUpSampling3D
    """
    def __init__(self, size=(2, 2, 2), dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(UpSampling3D, self).__init__(None, bigdl_type,
                                           size,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None)


class ZeroPadding1D(KerasLayer):
    """
    Zero-padding layer for 1D input (e.g. temporal sequence).
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    padding: Int or int tuple of length 2.
             If int, how many zeros to add both at the beginning and at the end of the padding dimension.
             If tuple of length 2, how many zeros to add in the order '(left_pad, right_pad)'.
             Default is 1.
    input_shape: A shape tuple, not including batch.

    >>> zeropadding1d = ZeroPadding1D(padding=2, input_shape=(3, 6))
    creating: createKerasZeroPadding1D
    """
    def __init__(self, padding=1, input_shape=None, bigdl_type="float"):
        if isinstance(padding, int):
            padding = (padding, padding)
        super(ZeroPadding1D, self).__init__(None, bigdl_type,
                                            padding,
                                            list(input_shape) if input_shape else None)


class ZeroPadding2D(KerasLayer):
    """
    Zero-padding layer for 2D input (e.g. picture).
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    padding: Int tuple of length 2 or length 4.
             If tuple of length 2, how many zeros to add both at the beginning and at the end of rows and cols.
             If tuple of length 4, how many zeros to add in the order '(top_pad, bottom_pad, left_pad, right_pad)'.
             Default is (1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> zeropadding2d = ZeroPadding2D(padding=(2, 1), input_shape=(2, 8, 8))
    creating: createKerasZeroPadding2D
    """
    def __init__(self, padding=(1, 1), dim_ordering="th", input_shape=None, bigdl_type="float"):
        if len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        super(ZeroPadding2D, self).__init__(None, bigdl_type,
                                            padding,
                                            dim_ordering,
                                            list(input_shape) if input_shape else None)


class ZeroPadding3D(KerasLayer):
    """
    Zero-padding layer for 3D data (spatial or spatio-temporal).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    padding: Int tuple of length 3. How many zeros to add at the beginning and at the end of the 3 padding dimensions.
             Symmetric padding will be applied to each dimension. Default is (1, 1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> zeropadding3d = ZeroPadding3D(padding=(2, 1, 2), input_shape=(2, 8, 8, 10))
    creating: createKerasZeroPadding3D
    """
    def __init__(self, padding=(1, 1, 1), dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(ZeroPadding3D, self).__init__(None, bigdl_type,
                                            padding,
                                            dim_ordering,
                                            list(input_shape) if input_shape else None)


class MaxPooling1D(KerasLayer):
    """
    Applies max pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_length: Size of the region to which max pooling is applied.
    strides: Factor by which to downscale. 2 will halve the input.
             Default is None, and in this case it will be equal to pool_length..
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    input_shape: A shape tuple, not including batch.

    >>> maxpooling1d = MaxPooling1D(3, input_shape=(3, 24))
    creating: createKerasMaxPooling1D
    """
    def __init__(self, pool_length=2, stride=None, border_mode="valid",
                 input_shape=None, bigdl_type="float"):
        if not stride:
            stride = -1
        super(MaxPooling1D, self).__init__(None, bigdl_type,
                                           pool_length,
                                           stride,
                                           border_mode,
                                           list(input_shape) if input_shape else None)


class MaxPooling2D(KerasLayer):
    """
    Applies max pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 2 corresponding to the downscale vertically and horizontally.
               Default is (2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 2. Stride values. Default is None, and in this case it will be equal to pool_size.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> maxpooling2d = MaxPooling2D((2, 2), input_shape=(3, 32, 32))
    creating: createKerasMaxPooling2D
    """
    def __init__(self, pool_size=(2, 2), strides=None,
                 border_mode='valid', dim_ordering='th',
                 input_shape=None, bigdl_type="float"):
        super(MaxPooling2D, self).__init__(None, bigdl_type,
                                           pool_size,
                                           strides,
                                           border_mode,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None)


class MaxPooling3D(KerasLayer):
    """
    Applies max pooling operation for 3D data (spatial or spatio-temporal).
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 3. Factors by which to downscale (dim1, dim2, dim3).
               Default is (2, 2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 3. Stride values. Default is None, and in this case it will be equal to pool_size.
    border_mode: Only 'valid' is supported for now.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.

    >>> maxpooling3d = MaxPooling3D((2, 1, 3), input_shape=(3, 32, 32, 32))
    creating: createKerasMaxPooling3D
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode="valid",
                 dim_ordering="th", input_shape=None, bigdl_type="float"):
        if border_mode != "valid":
            raise ValueError("For MaxPooling3D, only border_mode='valid' is supported for now")
        super(MaxPooling3D, self).__init__(None, bigdl_type,
                                           pool_size,
                                           strides,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None)


class AveragePooling1D(KerasLayer):
    """
    Applies average pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_length: Size of the region to which max pooling is applied.
    strides: Factor by which to downscale. 2 will halve the input.
             Default is None, and in this case it will be equal to pool_length..
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    input_shape: A shape tuple, not including batch.

    >>> averagepooling1d = AveragePooling1D(input_shape=(3, 24))
    creating: createKerasAveragePooling1D
    """
    def __init__(self, pool_length=2, stride=None, border_mode="valid",
                 input_shape=None, bigdl_type="float"):
        if not stride:
            stride = -1
        super(AveragePooling1D, self).__init__(None, bigdl_type,
                                               pool_length,
                                               stride,
                                               border_mode,
                                               list(input_shape) if input_shape else None)


class AveragePooling2D(KerasLayer):
    """
    Applies average pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 2 corresponding to the downscale vertically and horizontally.
               Default is (2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 2. Stride values. Default is None, and in this case it will be equal to pool_size.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> averagepooling2d = AveragePooling2D((1, 2), input_shape=(2, 28, 32))
    creating: createKerasAveragePooling2D
    """
    def __init__(self, pool_size=(2, 2), strides=None, border_mode="valid",
                 dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(AveragePooling2D, self).__init__(None, bigdl_type,
                                               pool_size,
                                               strides,
                                               border_mode,
                                               dim_ordering,
                                               list(input_shape) if input_shape else None)


class AveragePooling3D(KerasLayer):
    """
    Applies average pooling operation for 3D data (spatial or spatio-temporal).
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 3. Factors by which to downscale (dim1, dim2, dim3).
               Default is (2, 2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 3. Stride values. Default is None, and in this case it will be equal to pool_size.
    border_mode: Only 'valid' is supported for now.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.

    >>> averagepooling3d = AveragePooling3D((1, 1, 2), input_shape=(3, 28, 32, 36))
    creating: createKerasAveragePooling3D
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode="valid",
                 dim_ordering="th", input_shape=None, bigdl_type="float"):
        if border_mode != "valid":
            raise ValueError("For AveragePooling3D, only border_mode='valid' is supported for now")
        super(AveragePooling3D, self).__init__(None, bigdl_type,
                                               pool_size,
                                               strides,
                                               dim_ordering,
                                               list(input_shape) if input_shape else None)


class GlobalMaxPooling1D(KerasLayer):
    """
    Applies global max pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.

    >>> globalmaxpooling1d = GlobalMaxPooling1D(input_shape=(4, 8))
    creating: createKerasGlobalMaxPooling1D
    """
    def __init__(self, input_shape=None, bigdl_type="float"):
        super(GlobalMaxPooling1D, self).__init__(None, bigdl_type,
                                                 list(input_shape) if input_shape else None)


class GlobalAveragePooling1D(KerasLayer):
    """
    Applies global average pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.

    >>> globalaveragepooling1d = GlobalAveragePooling1D(input_shape=(12, 12))
    creating: createKerasGlobalAveragePooling1D
    """
    def __init__(self, input_shape=None, bigdl_type="float"):
        super(GlobalAveragePooling1D, self).__init__(None, bigdl_type,
                                                     list(input_shape) if input_shape else None)


class GlobalMaxPooling2D(KerasLayer):
    """
    Applies global max pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> globalmaxpooling2d = GlobalMaxPooling2D(input_shape=(4, 32, 32))
    creating: createKerasGlobalMaxPooling2D
    """
    def __init__(self, dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(GlobalMaxPooling2D, self).__init__(None, bigdl_type,
                                                 dim_ordering,
                                                 list(input_shape) if input_shape else None)


class GlobalAveragePooling2D(KerasLayer):
    """
    Applies global average pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> globalaveragepooling2d = GlobalAveragePooling2D(input_shape=(4, 32, 32))
    creating: createKerasGlobalAveragePooling2D
    """
    def __init__(self, dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(GlobalAveragePooling2D, self).__init__(None, bigdl_type,
                                                     dim_ordering,
                                                     list(input_shape) if input_shape else None)


class GlobalMaxPooling3D(KerasLayer):
    """
    Applies global max pooling operation for 3D data.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.

    >>> globalmaxpooling3d = GlobalMaxPooling3D(input_shape=(4, 32, 32, 32))
    creating: createKerasGlobalMaxPooling3D
    """
    def __init__(self, dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(GlobalMaxPooling3D, self).__init__(None, bigdl_type,
                                                 dim_ordering,
                                                 list(input_shape) if input_shape else None)


class GlobalAveragePooling3D(KerasLayer):
    """
    Applies global average pooling operation for 3D data.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.

    >>> globalaveragepooling3d = GlobalAveragePooling3D(input_shape=(4, 16, 16, 20))
    creating: createKerasGlobalAveragePooling3D
    """
    def __init__(self, dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(GlobalAveragePooling3D, self).__init__(None, bigdl_type,
                                                     dim_ordering,
                                                     list(input_shape) if input_shape else None)


class SimpleRNN(KerasLayer):
    """
    A fully-connected recurrent neural network cell. The output is to be fed back to input.
    The input of this layer should be 3D, i.e. (batch, time steps, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: Hidden unit size. Dimension of internal projections and final output.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is 'tanh'.
    return_sequences: Whether to return the full sequence or only return the last output in the output sequence.
                      Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    input_shape: A shape tuple, not including batch.

    >>> simplernn = SimpleRNN(16, input_shape=(3, 32))
    creating: createKerasSimpleRNN
    """
    def __init__(self, output_dim, activation="tanh", return_sequences=False,
                 go_backwards=False, W_regularizer=None, U_regularizer=None,
                 b_regularizer=None, input_shape=None, bigdl_type="float"):
        super(SimpleRNN, self).__init__(None, bigdl_type,
                                        output_dim,
                                        activation,
                                        return_sequences,
                                        go_backwards,
                                        W_regularizer,
                                        U_regularizer,
                                        b_regularizer,
                                        list(input_shape) if input_shape else None)


class LSTM(KerasLayer):
    """
    Long Short Term Memory unit architecture.
    The input of this layer should be 3D, i.e. (batch, time steps, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: Hidden unit size. Dimension of internal projections and final output.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is 'tanh'.
    inner_activation: String representations of activation function for inner cells. Default is 'hard_sigmoid'.
    return_sequences: Whether to return the full sequence or only return the last output in the output sequence.
                      Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    input_shape: A shape tuple, not including batch.

    >>> lstm = LSTM(32, input_shape=(8, 16))
    creating: createKerasLSTM
    """
    def __init__(self, output_dim, activation="tanh", inner_activation="hard_sigmoid",
                 return_sequences=False, go_backwards=False, W_regularizer=None,
                 U_regularizer=None, b_regularizer=None, input_shape=None, bigdl_type="float"):
        super(LSTM, self).__init__(None, bigdl_type,
                                   output_dim,
                                   activation,
                                   inner_activation,
                                   return_sequences,
                                   go_backwards,
                                   W_regularizer,
                                   U_regularizer,
                                   b_regularizer,
                                   list(input_shape) if input_shape else None)


class GRU(KerasLayer):
    """
    Gated Recurrent Unit architecture.
    The input of this layer should be 3D, i.e. (batch, time steps, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: Hidden unit size. Dimension of internal projections and final output.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is 'tanh'.
    inner_activation: String representations of activation function for inner cells. Default is 'hard_sigmoid'.
    return_sequences: Whether to return the full sequence or only return the last output in the output sequence.
                      Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    input_shape: A shape tuple, not including batch.

    >>> gru = GRU(24, input_shape=(32, 32))
    creating: createKerasGRU
    """
    def __init__(self, output_dim, activation="tanh", inner_activation="hard_sigmoid",
                 return_sequences=False, go_backwards=False, W_regularizer=None,
                 U_regularizer=None, b_regularizer=None, input_shape=None, bigdl_type="float"):
        super(GRU, self).__init__(None, bigdl_type,
                                  output_dim,
                                  activation,
                                  inner_activation,
                                  return_sequences,
                                  go_backwards,
                                  W_regularizer,
                                  U_regularizer,
                                  b_regularizer,
                                  list(input_shape) if input_shape else None)


class ConvLSTM2D(KerasLayer):
    """
    Convolutional LSTM.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'same'.
    The convolution kernel for this layer is a square kernel with equal strides 'subsample'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel. Should be equal to nb_row as for a square kernel.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is 'tanh'.
    inner_activation: String representations of activation function for inner cells. Default is 'hard_sigmoid'.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    border_mode: Only 'same' is supported for now.
    subsample: Tuple of length 2. Factor by which to subsample output. Also called strides elsewhere.
               Only support subsample[0] equal to subsample[1] for now. Default is (1, 1).
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    return_sequences: Whether to return the full sequence or only return the last output in the output sequence.
                      Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    input_shape: A shape tuple, not including batch.

    >>> convlstm2d = ConvLSTM2D(24, 3, 3, input_shape=(4, 32, 32, 32))
    creating: createKerasConvLSTM2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, activation="tanh",
                 inner_activation="hard_sigmoid", dim_ordering="th", border_mode="same",
                 subsample=(1, 1), W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 return_sequences=False, go_backwards=False, input_shape=None, bigdl_type="float"):
        if nb_row != nb_col:
            raise ValueError("For ConvLSTM2D, only square kernel is supported for now")
        if border_mode != "same":
            raise ValueError("For ConvLSTM2D, only border_mode='same' is supported for now")
        if subsample[0] != subsample[1]:
            raise ValueError("For ConvLSTM2D, only equal strides is supported for now")
        super(ConvLSTM2D, self).__init__(None, bigdl_type,
                                         nb_filter,
                                         nb_row,
                                         activation,
                                         inner_activation,
                                         dim_ordering,
                                         subsample[0],
                                         W_regularizer,
                                         U_regularizer,
                                         b_regularizer,
                                         return_sequences,
                                         go_backwards,
                                         list(input_shape) if input_shape else None)


class LocallyConnected1D(KerasLayer):
    """
    Locally-connected layer for 1D inputs which works similarly to the TemporalConvolution layer, except that
    weights are unshared, that is, a different set of filters is applied at each different patch of the input.
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Dimensionality of the output.
    filter_length: The extension (spatial or temporal) of each filter.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Only 'valid' is supported for now.
    subsample_length: Factor by which to subsample output. Int. Default is 1.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> locallyconnected1d = LocallyConnected1D(6, 3, input_shape=(8, 12))
    creating: createKerasLocallyConnected1D
    """
    def __init__(self, nb_filter, filter_length, activation=None, border_mode="valid",
                 subsample_length=1, W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, bigdl_type="float"):
        if border_mode != "valid":
            raise ValueError("For LocallyConnected1D, only border_mode='valid' is supported for now")
        super(LocallyConnected1D, self).__init__(None, bigdl_type,
                                                 nb_filter,
                                                 filter_length,
                                                 activation,
                                                 subsample_length,
                                                 W_regularizer,
                                                 b_regularizer,
                                                 bias,
                                                 list(input_shape) if input_shape else None)


class LocallyConnected2D(KerasLayer):
    """
    Locally-connected layer for 2D inputs that works similarly to the SpatialConvolution layer, except that
    weights are unshared, that is, a different set of filters is applied at each different patch of the input.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> locallyconnected2d = LocallyConnected2D(12, 3, 4, input_shape=(3, 128, 128))
    creating: createKerasLocallyConnected2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, activation=None,
                 border_mode="valid", subsample=(1, 1), dim_ordering="th",
                 W_regularizer=None, b_regularizer=None, bias=True,
                 input_shape=None, bigdl_type="float"):
        super(LocallyConnected2D, self).__init__(None, bigdl_type,
                                                 nb_filter,
                                                 nb_row,
                                                 nb_col,
                                                 activation,
                                                 border_mode,
                                                 subsample,
                                                 dim_ordering,
                                                 W_regularizer,
                                                 b_regularizer,
                                                 bias,
                                                 list(input_shape) if input_shape else None)


class SpatialDropout1D(KerasLayer):
    """
    Spatial 1D version of Dropout.
    This version performs the same function as Dropout, however it drops entire 1D feature maps
    instead of individual elements. If adjacent frames within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate decrease.
    In this case, SpatialDropout1D will help promote independence between feature maps and should be used instead.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    input_shape: A shape tuple, not including batch.

    >>> spatialdropout1d = SpatialDropout1D(0.4, input_shape=(10, 12))
    creating: createKerasSpatialDropout1D
    """
    def __init__(self, p=0.5, input_shape=None, bigdl_type="float"):
        super(SpatialDropout1D, self).__init__(None, bigdl_type,
                                               float(p),
                                               list(input_shape) if input_shape else None)


class SpatialDropout2D(KerasLayer):
    """
    Spatial 2D version of Dropout.
    This version performs the same function as Dropout, however it drops entire 2D feature maps
    instead of individual elements. If adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate decrease.
    In this case, SpatialDropout2D will help promote independence between feature maps and should be used instead.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> spatialdropout2d = SpatialDropout2D(0.25, input_shape=(5, 12, 12))
    creating: createKerasSpatialDropout2D
    """
    def __init__(self, p=0.5, dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(SpatialDropout2D, self).__init__(None, bigdl_type,
                                               float(p),
                                               dim_ordering,
                                               list(input_shape) if input_shape else None)


class SpatialDropout3D(KerasLayer):
    """
    Spatial 3D version of Dropout.
    This version performs the same function as Dropout, however it drops entire 3D feature maps
    instead of individual elements. If adjacent voxels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate decrease.
    In this case, SpatialDropout3D will help promote independence between feature maps and should be used instead.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> spatialdropout3d = SpatialDropout3D(0.6, input_shape=(4, 12, 12, 16))
    creating: createKerasSpatialDropout3D
    """
    def __init__(self, p=0.5, dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(SpatialDropout3D, self).__init__(None, bigdl_type,
                                               float(p),
                                               dim_ordering,
                                               list(input_shape) if input_shape else None)


class GaussianDropout(KerasLayer):
    """
    Apply multiplicative 1-centered Gaussian noise.
    As it is a regularization layer, it is only active at training time.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Drop probability. Float between 0 and 1.
       The multiplicative noise will have standard deviation 'sqrt(p/(1-p))'.
    input_shape: A shape tuple, not including batch.

    >>> gaussiandropout = GaussianDropout(0.45, input_shape=(4, 8))
    creating: createKerasGaussianDropout
    """
    def __init__(self, p, input_shape=None, bigdl_type="float"):
        super(GaussianDropout, self).__init__(None, bigdl_type,
                                              float(p),
                                              list(input_shape) if input_shape else None)


class GaussianNoise(KerasLayer):
    """
    Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
    Gaussian Noise is a natural choice as corruption process for real valued inputs.
    As it is a regularization layer, it is only active at training time.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    sigma: Float, standard deviation of the noise distribution.
    input_shape: A shape tuple, not including batch.

    >>> gaussiannoise = GaussianNoise(0.45, input_shape=(3, 4, 5))
    creating: createKerasGaussianNoise
    """
    def __init__(self, sigma, input_shape=None, bigdl_type="float"):
        super(GaussianNoise, self).__init__(None, bigdl_type,
                                            float(sigma),
                                            list(input_shape) if input_shape else None)


class Masking(KerasLayer):
    """
    Use a mask value to skip timesteps for a sequence.
    Masks a sequence by using a mask value to skip timesteps.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    mask_value: Float, mask value. For each timestep in the input tensor (dimension #1 in the tensor),
                if all values in the input tensor at that timestep are equal to `mask_value`,
                then the timestep will masked (skipped) in all downstream layers.
    input_shape: A shape tuple, not including batch.

    >>> masking = Masking(0.3, input_shape=(6, 8))
    creating: createKerasMasking
    """
    def __init__(self, mask_value=0.0, input_shape=None, bigdl_type="float"):
        super(Masking, self).__init__(None, bigdl_type,
                                      float(mask_value),
                                      list(input_shape) if input_shape else None)


class SReLU(KerasLayer):
    """
    S-shaped Rectified Linear Unit.
    It follows:
    f(x) = t^r + a^r(x - t^r) for x >= t^r
    f(x) = x for t^r > x > t^l,
    f(x) = t^l + a^l(x - t^l) for x <= t^l

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    t_left_init: String representations of initialization method for the left part intercept.
                 Default is 'zero'.
    a_left_init: String representations of initialization method for the left part slope.
                 Default is 'glorot_uniform'.
    t_right_init: String representations of initialization method for the right part intercept.
                  Default is 'glorot_uniform'.
    a_right_init: String representations of initialization method for the right part slope.
                  Default is 'one'.
    shared_axes: Int tuple. The axes along which to share learnable parameters for the activation function.
                 Default is None.
                 For example, if the incoming feature maps are from a 2D convolution with output shape
                 (batch, height, width, channels), and you wish to share parameters across space so that
                 each filter only has one set of parameters, set 'SharedAxes=(1,2)'.
    input_shape: A shape tuple, not including batch.

    >>> srelu = SReLU(input_shape=(4, 5))
    creating: createKerasSReLU
    """
    def __init__(self, t_left_init='zero', a_left_init='glorot_uniform',
                 t_right_init='glorot_uniform', a_right_init='one',
                 shared_axes=None, input_shape=None, bigdl_type="float"):
        super(SReLU, self).__init__(None, bigdl_type,
                                    t_left_init,
                                    a_left_init,
                                    t_right_init,
                                    a_right_init,
                                    shared_axes,
                                    list(input_shape) if input_shape else None)


class ELU(KerasLayer):
    """
    Exponential Linear Unit.
    It follows:
    f(x) =  alpha * (exp(x) - 1.) for x < 0,
    f(x) = x for x >= 0.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    alpha: Float, scale for the negative factor.
    input_shape: A shape tuple, not including batch.

    >>> elu = ELU(1.2, input_shape=(4, 5))
    creating: createKerasELU
    """
    def __init__(self, alpha=1.0, input_shape=None, bigdl_type="float"):
        super(ELU, self).__init__(None, bigdl_type,
                                  float(alpha),
                                  list(input_shape) if input_shape else None)


class LeakyReLU(KerasLayer):
    """
    Leaky version of a Rectified Linear Unit.
    It allows a small gradient when the unit is not active:
    f(x) = alpha * x for x < 0,
    f(x) = x for x >= 0.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    alpha: Float >= 0. Negative slope coefficient.
    input_shape: A shape tuple, not including batch.

    >>> leakyrelu = LeakyReLU(0.02, input_shape=(4, 5))
    creating: createKerasLeakyReLU
    """
    def __init__(self, alpha=0.01, input_shape=None, bigdl_type="float"):
        super(LeakyReLU, self).__init__(None, bigdl_type,
                                        float(alpha),
                                        list(input_shape) if input_shape else None)


class ThresholdedReLU(KerasLayer):
    """
    Thresholded Rectified Linear Unit.
    It follows:
    f(x) = x for x > theta,
    f(x) = 0 otherwise.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    theta: Float >= 0. Threshold location of activation.
    input_shape: A shape tuple, not including batch.

    >>> thresholdedrelu = ThresholdedReLU(input_shape=(10, 12))
    creating: createKerasThresholdedReLU
    """
    def __init__(self, theta=1.0, input_shape=None, bigdl_type="float"):
        super(ThresholdedReLU, self).__init__(None, bigdl_type,
                                              float(theta),
                                              list(input_shape) if input_shape else None)


class TimeDistributed(KerasLayer):
    """
    TimeDistributed wrapper.
    Apply a layer to every temporal slice of an input.
    The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    layer: A layer instance.
    input_shape: A shape tuple, not including batch.

    >>> timedistributed = TimeDistributed(Dense(8), input_shape=(10, 12))
    creating: createKerasDense
    creating: createKerasTimeDistributed
    """
    def __init__(self, layer, input_shape=None, bigdl_type="float"):
        super(TimeDistributed, self).__init__(None, bigdl_type,
                                              layer,
                                              list(input_shape) if input_shape else None)


class Bidirectional(KerasLayer):
    """
    Bidirectional wrapper for RNNs.
    Bidirectional currently requires RNNs to return the full sequence, i.e. return_sequences = True.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    Example of creating a bidirectional LSTM:
    Bidirectiona(LSTM(12, return_sequences=True), merge_mode="sum", input_shape=(32, 32))

    # Arguments
    layer: An instance of a recurrent layer.
    merge_mode: Mode by which outputs of the forward and backward RNNs will be combined.
                Must be one of: 'sum', 'mul', 'concat', 'ave'. Default is 'concat'.
    input_shape: A shape tuple, not including batch.

    >>> bidiretional = Bidirectional(LSTM(10, return_sequences=True), input_shape=(12, 16))
    creating: createKerasLSTM
    creating: createKerasBidirectional
    """
    def __init__(self, layer, merge_mode="concat", input_shape=None, bigdl_type="float"):
        super(Bidirectional, self).__init__(None, bigdl_type,
                                            layer,
                                            merge_mode,
                                            list(input_shape) if input_shape else None)