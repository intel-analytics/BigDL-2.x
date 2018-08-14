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

from bigdl.util.common import *
from zoo.pipeline.api.keras.base import ZooKerasCreator

if sys.version >= '3':
    long = int
    unicode = str


class AUC(JavaValue):
    """
    Metric for binary(0/1) classification, support single label and multiple labels.

    # Arguments
    threshold_num: The number of thresholds. The quality of approximation
                   may vary depending on threshold_num.

    >>> meter = AUC(20)
    creating: createAUC
    """
    def __init__(self, threshold_num=200, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, threshold_num)


class Accuracy(ZooKerasCreator, JavaValue):
    """
    Measures top1 accuracy for classification problems.

    # Arguments
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.

    >>> acc = Accuracy()
    creating: createZooKerasAccuracy
    """
    def __init__(self, zero_based_label=True, bigdl_type="float"):
        super(Accuracy, self).__init__(None, bigdl_type,
                                       zero_based_label)


class Top5Accuracy(ZooKerasCreator, JavaValue):
    """
    Measures top5 accuracy for classification problems.

    # Arguments
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.

    >>> acc = Top5Accuracy()
    creating: createZooKerasTop5Accuracy
    """
    def __init__(self, zero_based_label=True, bigdl_type="float"):
        super(Top5Accuracy, self).__init__(None, bigdl_type,
                                           zero_based_label)
