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

from bigdl.nn.layer import Layer

import numpy as np
import six
import tempfile
import os
import sys
from pyspark import RDD

from zoo.common.nncontext import getOrCreateSparkContext
from zoo.common import JTensor, Sample
from zoo.feature.image import ImageSet
from bigdl.util.common import callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


def to_sample_rdd(x, y, sc, num_slices=None):
    """
    Conver x and y into RDD[Sample]
    :param sc: SparkContext
    :param x: ndarray and the first dimension should be batch
    :param y: ndarray and the first dimension should be batch
    :param numSlices:
    :return:
    """
    x_rdd = sc.parallelize(x, num_slices)
    y_rdd = sc.parallelize(y, num_slices)
    return x_rdd.zip(y_rdd).map(lambda item: Sample.from_ndarray(item[0], item[1]))


class TFNet(Layer):
    def __init__(self, path, input_names=None, output_names=None, bigdl_type="float"):
        if input_names is None and output_names is None:
            super(TFNet, self).__init__(None, bigdl_type,
                                        path)
        else:
            if isinstance(input_names, six.string_types):
                input_names = [input_names]
            if isinstance(output_names, six.string_types):
                output_names = [output_names]
            super(TFNet, self).__init__(None, bigdl_type,
                                        path,
                                        input_names,
                                        output_names)

    @staticmethod
    def check_input(input):
        """
        :param input: ndarray or list of ndarray or JTensor or list of JTensor.
        :return: (list of JTensor, isTable)
        """

        def to_jtensor(i):
            if isinstance(i, np.ndarray):
                return JTensor.from_ndarray(i)
            elif isinstance(i, JTensor):
                return i
            else:
                raise Exception("Error unknown input type %s" % type(i))

        if type(input) is list:
            if len(input) == 0:
                raise Exception('Error when checking: empty input')
            return list(map(lambda i: to_jtensor(i), input)), True
        else:
            return [to_jtensor(input)], False

    def predict(self, x, batch_per_thread=1, distributed=True):
        """
        Use a model to do prediction.
        """
        if isinstance(x, ImageSet):
            results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                    self.value,
                                    x,
                                    batch_per_thread)
            return ImageSet(results)
        if distributed:
            if isinstance(x, np.ndarray):
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]), getOrCreateSparkContext())
            elif isinstance(x, RDD):
                data_rdd = x
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))
            results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                    self.value,
                                    data_rdd,
                                    batch_per_thread)
            return results.map(lambda result: Layer.convert_output(result))
        else:
            if isinstance(x, np.ndarray) or isinstance(x, list):
                results = callBigDlFunc(self.bigdl_type, "zooPredict",
                                        self.value,
                                        self._to_jtensors(x),
                                        batch_per_thread)
                return [Layer.convert_output(result) for result in results]
            else:
                raise TypeError("Unsupported prediction data type: %s" % type(x))

    @staticmethod
    def from_export_folder(folder):
        if not os.path.isdir(folder):
            raise ValueError(folder + " does not exist")
        return TFNet(folder)

    @staticmethod
    def from_session(sess, inputs, outputs,
                     generate_backward=False, allow_non_differentiable_input=True):
        from zoo.util.tf import export_tf
        temp = tempfile.mkdtemp()
        try:
            export_tf(sess, temp, inputs, outputs,
                      generate_backward, allow_non_differentiable_input)
            net = TFNet.from_export_folder(temp)
        finally:
            import shutil
            shutil.rmtree(temp)

        return net
