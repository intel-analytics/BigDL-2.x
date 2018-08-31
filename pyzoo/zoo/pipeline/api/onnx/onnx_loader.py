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
import zoo.pipeline.api.keras.models as zmodels
import onnx

from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
from zoo.pipeline.api.onnx.onnx_helper import OnnxHelper


class OnnxLoader(object):
    def __init__(self, onnx_graph):
        self.graph = onnx_graph
        self._all_tensors = {}  # including the original input tensor or the immediate tensor.
        self._params = {}  # name -> ndarray
        self._inputs = {}  # the original input tensor only.

    @classmethod
    def from_path(cls, onnx_path):
        onnx_model = onnx.load(onnx_path)
        return cls(onnx_model.graph)

    def to_keras(self):
        """Convert a Onnx model to KerasNet model.
      """
        # parse network inputs, aka parameters
        for init_tensor in self.graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self._params[init_tensor.name] = OnnxHelper.to_numpy(init_tensor)

        # converting GraphProto message
        # i: ValueInfoProto
        for i in self.graph.input:  # input including param (coming from initializer) and input
            if i.name in self._params:
                pass  # we've add that via graph.initializer
            else:
                # We need to remove the batch dim.
                self._inputs[i.name] = zlayers.Input(shape=OnnxHelper.get_shape_from_node(i)[1:],
                                                     name=i.name)
                self._all_tensors[i.name] = self._inputs[i.name]

        # constructing nodes, nodes are stored as directed acyclic graph
        # converting NodeProto message
        for node in self.graph.node:
            converter = OperatorMapper.of(node,
                                          self._params, self._all_tensors)
            converter.action()

        output_tensors = []
        for i in self.graph.output:
            if i.name not in self._all_tensors:
                raise Exception("The output haven't been calculate")
            output_tensors.append(self._all_tensors[i.name])

        # TODO: fix the order problem
        model = zmodels.Model(input=list(self._inputs.values()), output=output_tensors)
        return model
