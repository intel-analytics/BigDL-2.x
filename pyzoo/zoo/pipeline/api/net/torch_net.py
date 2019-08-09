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
import torch
import os
import tempfile
import shutil
import numpy as np
import sys

from pyspark import RDD
from bigdl.nn.layer import Layer
from zoo import getOrCreateSparkContext
from zoo.feature.image import ImageSet
from bigdl.util.common import callBigDlFunc
from zoo.pipeline.api.net.tfnet import to_sample_rdd

if sys.version >= '3':
    long = int
    unicode = str


class TorchNet(Layer):
    """
    TorchNet wraps a TorchScript model as a single layer, thus the Pytorch model can be used for
    distributed inference or training.
    :param path: path to the TorchScript model.
    """

    def __init__(self, path, bigdl_type="float"):
        super(TorchNet, self).__init__(None, bigdl_type, path)

    @staticmethod
    def from_pytorch(module, input_shape=None, sample_input=None):
        """
        Create a TorchNet directly from PyTorch model, e.g. model in torchvision.models.
        Users need to specify sample_input or input_shape.
        :param module: a PyTorch model
        :param input_shape: list of integers, or tuple of list for multiple inputs models. E.g.
                            for ResNet, this may be [1, 3, 224, 224]
        :param sample_input. A sample of Torch Tensor or tuple to trace the model.
        """
        if not input_shape and not sample_input:
            raise Exception("please specify input_shape or sample_input")

        sample = TorchNet.get_sample_input(input_shape, sample_input)
        temp = tempfile.mkdtemp()

        # save model
        traced_script_module = torch.jit.trace(module, sample)
        path = os.path.join(temp, "model.pt")
        traced_script_module.save(path)

        net = TorchNet(path)
        shutil.rmtree(temp)

        return net

    @staticmethod
    def get_sample_input(shape, sample):
        if sample:
            return sample
        elif isinstance(shape, list):
            return torch.rand(shape)
        elif isinstance(shape, tuple):
            return tuple(map(lambda s: torch.rand(s), shape))
        else:
            raise Exception("please specify shape as list of ints or tuples of int lists")

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
