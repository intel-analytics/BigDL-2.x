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


class BatchNormalization(ZooKeras2Layer):
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
    beta_initializer: Name of the initialization function for shift parameter. Default is 'zero'.
    gamma_initializer: Name of the initialization function for scale parameter. Default is 'one'.
    data_format: Format of input data. Either 'channels_first' or 'channels_last'.
                  Default is 'channels_first'. For 'channels_first,
                  axis along which to normalize is 1.For 'channels_last', axis is 3.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> batchnormalization = BatchNormalization(input_shape=(3, 12, 12))
    creating: createZooKeras2BatchNormalization
    """
    def __init__(self,
                 epsilon=0.001,
                 mode=0,
                 axis=1,
                 momentum=0.99,
                 beta_initializer="zero",
                 gamma_initializer="one",
                 data_format="channels_first",
                 input_shape=None,
                 **kwargs):
        if mode != 0:
            raise ValueError("For BatchNormalization, only mode=0 is supported for now")
        if data_format == "channels_first" and axis != 1:
            raise ValueError("For BatchNormalization with channels_first data_format, "
                             "only axis=1 is supported for now")
        if data_format == "channels_last" and axis != -1 and axis != 3:
            raise ValueError("For BatchNormalization with channels_last data_format, "
                             "only axis=-1 is supported for now")
        super(BatchNormalization, self).__init__(None,
                                                 float(epsilon),
                                                 float(momentum),
                                                 beta_initializer,
                                                 gamma_initializer,
                                                 data_format,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)
