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
import io

import torch

from bigdl.nn.layer import Layer
from bigdl.util.common import JTensor
from zoo.common.utils import callZooFunc
from pyspark.serializers import CloudPickleSerializer
from zoo.pipeline.api.torch.utils import trainable_param
from zoo.pipeline.api.torch import zoo_pickle_module
from importlib.util import find_spec

if sys.version < '3.7':
    print("WARN: detect python < 3.7, if you meet zlib not available " +
          "exception on yarn, please update your python to 3.7")

if find_spec('jep') is None:
    raise Exception("jep not found, please install jep first.")


class TorchModel(Layer):
    """
    TorchModel wraps a PyTorch model as a single layer, thus the PyTorch model can be used for
    distributed inference or training.
    The implement of TorchModel is different from TorchNet, this TorchModel is running running
    pytorch model in an embedding Cpython interpreter, while TorchNet transfer the pytorch model
    to TorchScript and run with libtorch.
    """

    def __init__(self, module_bytes, weights, bigdl_type="float"):
        weights = JTensor.from_ndarray(weights)
        self.module_bytes = module_bytes
        self.value = callZooFunc(
            bigdl_type, self.jvm_class_constructor(), module_bytes, weights)
        self.bigdl_type = bigdl_type

    @staticmethod
    def from_pytorch(model):
        """
        Create a TorchNet directly from PyTorch model, e.g. model in torchvision.models.
        :param model: a PyTorch model
        """
        weights = []
        for param in trainable_param(model):
            weights.append(param.view(-1))
        flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
        bys = io.BytesIO()
        torch.save(model, bys, pickle_module=zoo_pickle_module)
        net = TorchModel(bys.getvalue(), flatten_weight)
        return net

    def to_pytorch(self):
        """
        Convert to pytorch model
        :return: a pytorch model
        """
        new_weight = self.get_weights()
        assert(len(new_weight) == 1, "TorchModel's weights should be one tensor")
        # set weights
        m = torch.load(io.BytesIO(self.module_bytes), pickle_module=zoo_pickle_module)
        w = torch.Tensor(new_weight[0])
        torch.nn.utils.vector_to_parameters(w, trainable_param(m))

        # set named buffers
        new_extra_params = callZooFunc(self.bigdl_type, "getModuleExtraParameters", self.value)
        if len(new_extra_params) != 0:
            idx = 0
            for named_buffer in m.named_buffers():
                named_buffer[1].copy_(torch.reshape(
                    torch.Tensor(new_extra_params[idx].to_ndarray()), named_buffer[1].size()))
                idx += 1
        return m
