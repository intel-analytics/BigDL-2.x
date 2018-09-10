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
import bigdl.nn.layer as blayer
import numpy as np

import zoo.pipeline.api.keras.layers as zlayers
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
from zoo.pipeline.api.onnx.onnx_helper import OnnxHelper
from zoo.pipeline.api.onnx.onnx_loader import OnnxInput


class ConstantMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(ConstantMapper, self).__init__(node, _params, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        input = OnnxInput(name=self.op_name, zvalue=OnnxHelper.to_numpy(self.onnx_attr['value']))
        return [self._to_zoo_input(input, is_constant=True)]

    def _to_tensor(self):
        return self.model_inputs[0].zvalue


