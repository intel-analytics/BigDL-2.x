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
from zoo.pipeline.api.autograd import Parameter
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper


class ReshapeMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(ReshapeMapper, self).__init__(node, _params, _all_tensors)

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
        origin_target = self._input_list[1].zvalue
        target = origin_target.get_weight() if isinstance(origin_target, Parameter) else \
            origin_target
        if self.is_batch:
            targetshape = [int(i) for i in target][1:]
            return zlayers.Reshape(targetshape)(self.model_inputs[0].zvalue)
        else:
            targetshape = [int(i) for i in target]
            return zlayers.KerasLayerWrapper(blayer.Reshape(targetshape, batch_mode=False))(data)
