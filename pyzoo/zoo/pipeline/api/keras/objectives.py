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

from zoo.pipeline.api.keras.base import ZooKerasCreator
from bigdl.nn.criterion import Criterion
from bigdl.util.common import JTensor

if sys.version >= '3':
    long = int
    unicode = str


class LossFunction(ZooKerasCreator, Criterion):
    """
    The base class for Keras-style API objectives in Analytics Zoo.
    """
    def __init__(self, jvalue, bigdl_type, *args):
        super(Criterion, self).__init__(jvalue, bigdl_type, *args)

    @classmethod
    def of(cls, jloss, bigdl_type="float"):
        """
        Create a Python LossFunction from a JavaObject.

        # Arguments
        jloss: A java criterion object which created by Py4j
        """
        loss = LossFunction(bigdl_type, jloss)
        loss.value = jloss
        loss.bigdl_type = bigdl_type
        return loss


class SparseCategoricalCrossEntropy(LossFunction):
    """
    A loss often used in multi-class classification problems with SoftMax
    as the last layer of the neural network.

    By default, same as Keras, input(y_pred) is supposed to be probabilities of each class,
    and target(y_true) is supposed to be the class label starting from 0.

    # Arguments
    log_prob_as_input: Boolean. Whether to accept log-probabilities or probabilities
                       as input. Default is False and inputs should be probabilities.
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.
    weights: A Numpy array. Weights of each class if you have an unbalanced training set.
    size_average: Boolean. Whether losses are averaged over observations for each
                  mini-batch. Default is True. If False, the losses are instead
                  summed for each mini-batch.
    padding_value: Int. If the target is set to this value, the training process
                   will skip this sample. In other words, the forward process will
                   return zero output and the backward process will also return
                   zero grad_input. Default is -1.

    >>> loss = SparseCategoricalCrossEntropy()
    creating: createZooKerasSparseCategoricalCrossEntropy
    >>> import numpy as np
    >>> np.random.seed(1128)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> loss = SparseCategoricalCrossEntropy(weights=weights)
    creating: createZooKerasSparseCategoricalCrossEntropy
    """
    def __init__(self, log_prob_as_input=False, zero_based_label=True,
                 weights=None, size_average=True, padding_value=-1, bigdl_type="float"):
        super(SparseCategoricalCrossEntropy, self).__init__(None, bigdl_type,
                                                            log_prob_as_input,
                                                            zero_based_label,
                                                            JTensor.from_ndarray(weights),
                                                            size_average,
                                                            padding_value)


class MeanAbsoluteError(LossFunction):
    """
    A loss that measures the mean absolute value of the element-wise difference
    between the input and the target.

    # Arguments
    size_average: Boolean. Whether losses are averaged over observations for each
              mini-batch. Default is True. If False, the losses are instead
              summed for each mini-batch.

    >>> loss = MeanAbsoluteError()
    creating: createZooKerasMeanAbsoluteError
    """
    def __init__(self, size_average=True, bigdl_type="float"):
        super(MeanAbsoluteError, self).__init__(None, bigdl_type,
                                                size_average)


mae = MAE = MeanAbsoluteError

class BinaryCrossentropy(LossFunction):
    """
        A loss that measures the Binary Cross Entropy between the target and the output

        # Arguments
        size_average: Boolean. Whether losses are averaged over observations for each
                  mini-batch. Default is True. If False, the losses are instead
                  summed for each mini-batch.
        weights: weights over the input dimension
        ev: numeric operator
        T numeric type

        >>> metrics = BinaryCrossentropy()
        creating: createZooKerasBinaryCrossentropy
        """

    def __init__(self, weights, size_average=True, bigdl_type="float"):
        super(BinaryCrossentropy, self).__init__(None, weights, bigdl_type,
                                                size_average)

class CategoricalCrossentropy(LossFunction):
    """
        This is same with cross entropy criterion, except the target tensor is a one-hot tensor

        # Arguments
        T numeric type

        >>> metrics = CategoricalCrossentropy()
        creating: createZooKerasCategoricalCrossentropy
        """

    def __init__(self, bigdl_type="float"):
        super(CategoricalCrossentropy, self).__init__(None, bigdl_type)

class CosineProximity(LossFunction):
    """
        The negative of the mean cosine proximity between predictions and targets.
        The cosine proximity is defined as below:
        x'(i) = x(i) / sqrt(max(sum(x(i)^2), 1e-12))
        y'(i) = y(i) / sqrt(max(sum(x(i)^2), 1e-12))
        cosine_proximity(x, y) = mean(-1 * x'(i) * y'(i))
        Both batch and un-batched inputs are supported

        # Arguments


        >>> metrics = CosineProximity()
        creating: createZooKerasCosineProximity
        """

    def __init__(self, bigdl_type="float"):
        super(CosineProximity, self).__init__(None, bigdl_type)

class Hinge(LossFunction):
    """
        Creates a criterion that optimizes a two-class classification (squared) hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.
         When margin = 1, sizeAverage = True and squared = False, this is the same as hinge loss in keras;
         When margin = 1, sizeAverage = False and squared = True, this is the same as squared_hinge loss in keras.

        # Arguments:
        margin: if unspecified, is by default 1.
        size_average: whether to average the loss
        squared: whether to calculate the squared hinge loss

        >>> metrics = Hinge()
        creating: createZooKerasHinge
        """

    def __init__(self, margin=1.0, size_average=True, squared=False, bigdl_type="float"):
        super(Hinge, self).__init__(None, margin, size_average, squared, bigdl_type)

class KullbackLeiblerDivergence(LossFunction):
    """
        This method is same as `kullback_leibler_divergence` loss in keras.
        Loss calculated as:y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)

        # Arguments
        T: The numeric type in the criterion, usually which are [[Float]] or [[Double]]


        >>> metrics = KullbackLeiblerDivergence()
        creating: createZooKerasKullbackLeiblerDivergence
        """

    def __init__(self, bigdl_type="float"):
        super(KullbackLeiblerDivergence, self).__init__(None, bigdl_type)

class MeanAbsolutePercentageError(LossFunction):
    """
        This method is same as `mean_absolute_percentage_error` loss in keras.
        It caculates diff = K.abs((y - x) / K.clip(K.abs(y), K.epsilon(), Double.MaxValue))and return 100 * K.mean(diff) as outpout
        Here, the x and y can have or not have a batch.

        # Arguments
        T: The numeric type in the criterion, usually which are [[Float]] or [[Double]]


        >>> metrics = MeanAbsolutePercentageError()
        creating: createZooKerasMeanAbsolutePercentageError
        """

    def __init__(self, bigdl_type="float"):
        super(MeanAbsolutePercentageError, self).__init__(None, bigdl_type)

class MeanSquaredError(LossFunction):
    """
    A loss that measures the mean absolute value of the element-wise difference
    between the input and the target.

    # Arguments
    size_average: Boolean. Whether losses are averaged over observations for each
              mini-batch. Default is True. If False, the losses are instead
              summed for each mini-batch.

    >>> metrics = MeanAbsoluteError()
    creating: createZooKerasMeanAbsoluteError
    """
    def __init__(self, size_average=True, bigdl_type="float"):
        super(MeanAbsoluteError, self).__init__(None, bigdl_type,
                                                size_average)