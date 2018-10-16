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
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
from zoo.pipeline.api.onnx.onnx_helper import OnnxHelper
import zoo.pipeline.api.keras.layers as zlayers
import zoo.pipeline.api.autograd as zautograd
import numpy as np


class MatMulMapper(OperatorMapper):
    def __init__(self, node, initializer, inputs):
        super(MatMulMapper, self).__init__(node, initializer, inputs)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        def gen_input(input):
            if isinstance(input.zvalue, np.ndarray):
                return self._to_zoo_input(input, is_constant=True)
            else:
                return self._to_zoo_input(input)
        return [gen_input(i) for i in self._input_list]

    def _to_tensor(self):
        x = self.model_inputs[0]
        y = self.model_inputs[1]
        return zautograd.mm(x.zvalue, y.zvalue)
