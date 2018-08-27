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


class OperatorMapper(object):
    # converting NodeProto message
    def __init__(self, node, _params, _all_tensors):
        self.node = node
        self.op_name = node.op_type
        node_name = node.name.strip()
        # it would be None if user doesn't set it manually
        self.node_name = node_name if node_name else None
        self.onnx_attr = OnnxHelper.parse_attr(node.attribute)  # dict(name: value)
        self.params = [_params[i] for i in node.input if i in _params]
        self.inputs = [_all_tensors[i] for i in node.input if i in _all_tensors]
        self.all_tensors = _all_tensors
        assert len(node.output) == 1, "we only support single output for now"
        self.output = node.output[0]

    @staticmethod
    def of(node, _params, _all_tensors):
        m = importlib.import_module("zoo.pipeline.api.onnx.mapper." + node.op_type.lower())
        cls = getattr(m, node.op_type + "Mapper")
        return cls(node, _params, _all_tensors)

    def action(self):
        """
        Convert a node to KerasLayer
        """
        operator = self.create_operator()
        operator.set_name(self.node_name)
        if not isinstance(operator, zautograd.Variable):
            z_tensor = operator(self.inputs)
            operator.set_weights(self.format_params(self.params))
        else:
            z_tensor = operator
            operator.node.element().set_weights(self.format_params(self.params))

        self.all_tensors[self.output] = z_tensor  # update the all_tensors
        return z_tensor

    def create_operator(self):
        raise Exception("Please define the content")

    def format_params(self, params):
        """
        Convert ONNX params to Zoo format
        :return: list of ndarray
        """
        return params
