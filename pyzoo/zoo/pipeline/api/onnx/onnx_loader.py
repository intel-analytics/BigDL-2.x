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
from collections import OrderedDict

import onnx

import zoo.pipeline.api.keras.models as zmodels
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
from zoo.pipeline.api.onnx.onnx_helper import OnnxHelper
import warnings


class OnnxInput(object):
    """
    .. note:: `zoo.pipeline.api.onnx` is deprecated in 0.10.0
    This will be removed in future releases.
     """
    def __init__(self, name, zvalue, data=None):
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        self.name = name
        self.zvalue = zvalue  # zvalue is a Input or Parameter
        self.data = data  # data is a ndarray


class OnnxLoader(object):
    """
    .. note:: `zoo.pipeline.api.onnx` is deprecated in 0.10.0
    This will be removed in future releases.
    """
    training = False

    def __init__(self, onnx_graph):
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        self.graph = onnx_graph
        self._all_tensors = {}  # including the original input tensor or the immediate tensor.
        self.initializer = {}  # name -> ndarray
        self._inputs = OrderedDict()  # the original input tensor only.

    @classmethod
    def from_path(cls, onnx_path, is_training=False):
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        onnx_model = onnx.load(onnx_path)
        try:
            zmodel = OnnxLoader(onnx_model.graph).to_keras()
        except Exception as e:
            print(onnx_model)
            raise e
        zmodel.training(is_training=is_training)
        return zmodel

    @staticmethod
    # inputs_dict is a list of batch data
    def run_node(node, inputs, is_training=False):
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        inputs_list = []
        assert len(inputs) == len(list(node.input))
        for node_input, input_data in zip(node.input, inputs):
            inputs_list.append(OnnxInput(node_input, input_data))
        mapper = OperatorMapper.of(node, set(), inputs_list)
        out_tensor = mapper.to_tensor()

        model = zmodels.Model(input=[i.zvalue for i in mapper.model_inputs], output=out_tensor)
        data = [i.data for i in mapper.model_inputs]
        model.training(is_training)
        output = model.forward(data if len(data) > 1 else data[0])
        result = {}
        if isinstance(output, list):
            assert len(output) == len(node.output)
            for o, no in zip(output, node.output):
                result[no] = o
        else:
            result[node.output[0]] = output
        return result

    def to_keras(self):
        """Convert a Onnx model to KerasNet model.
      """
        warnings.warn("deprecated in 0.10.0, and will be removed in future release")
        # parse network inputs, aka parameters
        for init_tensor in self.graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self.initializer[init_tensor.name] = OnnxInput(name=init_tensor.name,
                                                           zvalue=OnnxHelper.to_numpy(init_tensor))

        # converting GraphProto message
        # i: ValueInfoProto
        for i in self.graph.input:
            if i.name in self.initializer:
                # we should have added that via graph._initializer
                self._all_tensors[i.name] = self.initializer[i.name]
            else:
                self._inputs[i.name] = OnnxInput(name=i.name,
                                                 zvalue=OnnxHelper.get_shape_from_node(i))
                self._all_tensors[i.name] = self._inputs[i.name]

        # constructing nodes, nodes are stored as directed acyclic graph
        # converting NodeProto message
        for node in self.graph.node:
            inputs = []
            for i in node.input:
                if i == "":
                    continue
                if i not in self._all_tensors:
                    raise Exception("Cannot find {}".format(i))
                inputs.append(self._all_tensors[i])

            mapper = OperatorMapper.of(node,
                                       self.initializer, inputs)
            # update inputs and all_tensors
            for input in mapper.model_inputs:
                # Only update the original inputs here.
                if input.name in self._inputs:
                    self._inputs[input.name] = input.zvalue
                self._all_tensors[input.name] = input.zvalue
            tensor = mapper.to_tensor()
            output_ids = list(node.output)
            assert len(output_ids) == 1 or node.op_type == "Dropout",\
                "Only support single output for now"
            self._all_tensors[output_ids[0]] = OnnxInput(name=tensor.name, zvalue=tensor)

        output_tensors = []
        for i in self.graph.output:
            if i.name not in self._all_tensors:
                raise Exception("The output haven't been calculate")
            output_tensors.append(self._all_tensors[i.name].zvalue)
        model = zmodels.Model(input=list(self._inputs.values()), output=output_tensors)
        return model
