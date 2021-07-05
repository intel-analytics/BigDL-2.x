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
import zoo.pipeline.api.keras.layers as zlayers
import numpy as np
from zoo.pipeline.api.autograd import Parameter
import bigdl.nn.layer as blayer
import zoo.pipeline.api.autograd as autograd


class UnsqueezeMapper(OperatorMapper):
    def __init__(self, node, initializer, _all_tensors):
        super(UnsqueezeMapper, self).__init__(node, initializer, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        input = self._input_list[0]
        if isinstance(input.zvalue, np.ndarray):
            self.is_batch = False
            return [self._to_zoo_input(input, is_constant=True)]
        else:
            self.is_batch = True
            return [self._to_zoo_input(input)]

    def _to_tensor(self):
        data = self.model_inputs[0].zvalue
        dim = sorted(tuple([int(i) for i in self.onnx_attr['axes']]))
        for i in dim:
            data = autograd.expand_dims(data, axis=i)
        return data
