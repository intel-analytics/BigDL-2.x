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
import torch.nn as nn
import os
import tempfile
import shutil
from bigdl.nn.criterion import Criterion

class LossWrapperModule(nn.Module):
    def __init__(self, lossFunc):
        super(LossWrapperModule, self).__init__()
        self.func = lossFunc

    def forward(self, x, y):
        return self.func(x, y)


class TorchCriterion(Criterion):
    """
    TorchCriterion wraps a TorchScript model as a single layer, thus the Pytorch model can be used for
    distributed inference or training.
    :param path: path to the TorchScript model.
    """

    def __init__(self, path, bigdl_type="float"):
        super(TorchCriterion, self).__init__(None, bigdl_type, path)

    @staticmethod
    def from_pytorch(lossFunc, input_shape, label_shape=None, sample_input=None, sample_label=None):
        """
        Create a TorchCriterion directly from PyTorch function
        :param lossFunc: a function that take two parameters: input and label
        :param input_shape: list of integers.
        :param label_shape: list of integers. If not specified, it will be set equal to input_shape
        :param sample_input: a sample of input.
        :param sample_label: a sample of label.
        """
        temp = tempfile.mkdtemp()

        sample_input = sample_input if sample_input else torch.rand(input_shape)
        sample_label = sample_label if sample_label else torch.rand(label_shape)

        traced_script_loss = torch.jit.trace(LossWrapperModule(lossFunc),
                                             (sample_input, sample_label))
        lossPath = os.path.join(temp, "loss.pt")
        traced_script_loss.save(lossPath)

        criterion = TorchCriterion(lossPath)
        shutil.rmtree(temp)

        return criterion

