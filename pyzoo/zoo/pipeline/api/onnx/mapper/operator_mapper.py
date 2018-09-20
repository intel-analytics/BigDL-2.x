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


import zoo.pipeline.api.keras.layers as zlayers
import zoo.pipeline.api.autograd as zautograd
from zoo.pipeline.api.onnx.onnx_helper import OnnxHelper
import zoo.pipeline.api.onnx.mapper
import importlib
import numpy as np


class OperatorMapper(object):
    # converting NodeProto message
    # we don't differentiate the data input and weights here, they are all included into inputs.
    def __init__(self, node, initializer, inputs):
        self.node = node
        self.op_name = node.op_type
        node_name = node.name.strip()
        # it would be None if user doesn't set it manually
        self.node_name = node_name if node_name else None
        self.onnx_attr = OnnxHelper.parse_attr(node.attribute)  # dict(name: value)
        self._initializer = initializer
        self._input_list = inputs
        self.model_inputs = self._extract_model_inputs()
        self.model_trainable_values = self._extract_trainable_values()
        self.output = node.output[0]

    @staticmethod
    def of(node, _params, inputs):
        m = importlib.import_module("zoo.pipeline.api.onnx.mapper." + node.op_type.lower())
        cls = getattr(m, node.op_type + "Mapper")
        return cls(node, _params, inputs)

    def _to_zoo_input(self, input, is_constant=None):
        is_parameter = True if input.name in self._initializer else False
        if isinstance(input.zvalue, zautograd.Variable) or isinstance(input.zvalue,
                                                                      zautograd.Parameter):
            return input
        if isinstance(input.zvalue, np.ndarray):
            if is_parameter or is_constant:
                shape = input.zvalue.shape
            else:
                shape = input.zvalue.shape[1:]
        elif isinstance(input.zvalue, list):
            if is_parameter or is_constant:
                shape = input.zvalue
            else:
                shape = input.zvalue[1:]
        else:
            raise Exception("not supported type " + str(type(input.zvalue)))

        input.data = input.zvalue
        if is_constant:
            input.zvalue = zautograd.Parameter(shape=shape, init_weight=input.zvalue,
                                               trainable=False)
        elif is_parameter:
            input.zvalue = zautograd.Parameter(shape=shape, init_weight=input.zvalue, )
        else:
            input.zvalue = zlayers.Input(
                shape=shape, name=input.name)
        return input

    def to_tensor(self):
        """
        Convert a node to tensor
        """
        out_tensor = self._to_tensor()
        if self.node_name:
            out_tensor.set_name(self.node_name)
        assert isinstance(out_tensor, zautograd.Variable) or isinstance(out_tensor,
                                                                        zautograd.Parameter)
        if self.model_trainable_values:
            out_tensor.node.element().set_weights(
                self.to_zoo_format(self.model_trainable_values))
        return out_tensor

    def _to_tensor(self):
        raise Exception("Please define the content")

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        return [self._to_zoo_input(i) for i in self._input_list]

    def _extract_trainable_values(self):
        """
        :return: list of ndarray for weights
        """
        return None

    def to_zoo_format(self, trainable_values):
        """
        Convert ONNX _initializer to Zoo format
        :return: list of ndarray
        """
        return trainable_values
