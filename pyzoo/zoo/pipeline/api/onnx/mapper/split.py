# -*- coding: utf-8 -*
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
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
import zoo.pipeline.api.keras.layers as zlayers
import zoo.pipeline.api.autograd as autograd
import bigdl.nn.layer as blayer


class SplitMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(SplitMapper, self).__init__(node, _params, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        return [self._to_zoo_input(self._input_list[0])]

    def _to_tensor(self):
        input = self.model_inputs[0].zvalue
        axis = self.onnx_attr['axis']
        split = self.onnx_attr['split']
        output = list(range(len(split)))
        start = 0
        for i in range(len(split)):
            output[i] = input.slice(dim=int(axis), start_index=int(start), length=int(split[i]))
            start = start + split[i]
        return output
