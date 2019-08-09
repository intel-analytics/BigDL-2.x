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
import sys
import os
import tempfile
import shutil
from bigdl.nn.criterion import Criterion
from .torch_net import TorchNet

if sys.version >= '3':
    long = int
    unicode = str


class LossWrapperModule(nn.Module):
    def __init__(self, lossFunc):
        super(LossWrapperModule, self).__init__()
        self.func = lossFunc

    def forward(self, x, y):
        return self.func(x, y)


class TorchCriterion(Criterion):
    """
    TorchCriterion wraps a loss function for distributed inference or training.
    Use TorchCriterion.from_pytorch to initialize.
    """

    def __init__(self, path, bigdl_type="float"):
        """
        :param path: path to the TorchScript model.
        :param bigdl_type:
        """
        super(TorchCriterion, self).__init__(None, bigdl_type, path)

    @staticmethod
    def from_pytorch(loss, input_shape=None, label_shape=None,
                     sample_input=None, sample_label=None):
        """
        Create a TorchCriterion directly from PyTorch function. We need user to provide a sample
        input and label to trace the loss function. User may just specify the input and label shape.
        For specific data type or multiple input models, users can send sample_input and
        sample_label.
        :param loss: this can be a torch loss (e.g. nn.MSELoss()) or
                     a function that take two Tensor parameters: input and label. E.g.
                     def lossFunc(input, target):
                         return nn.CrossEntropyLoss().forward(input, target.flatten().long())

        :param input_shape: list of integers.
        :param label_shape: list of integers. If not specified, it will be set equal to input_shape
        :param sample_input: a sample of input.
        :param sample_label: a sample of label.
        """
        if not input_shape and not label_shape and not sample_input and not sample_label:
            raise Exception("please specify input_shape and label_shape, or sample_input"
                            " and sample_label")

        temp = tempfile.mkdtemp()

        # use input_shape as label shape when label_shape is not specified
        if not label_shape:
            label_shape = input_shape

        sample_input = TorchNet.get_sample_input(input_shape, sample_input)
        sample_label = TorchNet.get_sample_input(label_shape, sample_label)

        traced_script_loss = torch.jit.trace(LossWrapperModule(loss),
                                             (sample_input, sample_label))
        lossPath = os.path.join(temp, "loss.pt")
        traced_script_loss.save(lossPath)

        criterion = TorchCriterion(lossPath)
        shutil.rmtree(temp)

        return criterion
