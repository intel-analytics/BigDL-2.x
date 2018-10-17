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
import numpy as np


class PowMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(PowMapper, self).__init__(node, _params, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        return [self._to_zoo_input(self._input_list[0])]

    def _extract_trainable_values(self):
        if len(self._input_list) > 2:
            return [self._input_list[1].zvalue, self._input_list[2].zvalue]
        else:
            return [self._input_list[1].zvalue]  # without bias

    def to_zoo_format(self, trainable_values):
        """
        Convert ONNX _initializer to Zoo format
        :return: list of ndarray
        """
        if len(trainable_values) > 1:
            return [np.expand_dims(trainable_values[0], 0), trainable_values[1]]
        else:
            return np.expand_dims(trainable_values[0], 0)

    def _to_tensor(self):
        exponent = self.model_trainable_values[0]
        neg = zlayers.Power(exponent)
        return neg(self.model_inputs[0].zvalue)

