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


class BatchNormalizationMapper(OperatorMapper):
    bigdl_type = "float"
    def __init__(self, node, _params, _all_tensors):
        super(BatchNormalizationMapper, self).__init__(node, _params, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        return [self._to_zoo_input(self._input_list[0])]

    def _to_tensor(self):
        input = self.model_inputs[0]
        rank = len(input.zvalue.shape)

        if (rank == 4):
            epsilon = float(self.onnx_attr['epsilon'])
            mode = self.onnx_attr['mode'] if "mode" in self.onnx_attr else 0
            axis = self.onnx_attr['axis'] if "axis" in self.onnx_attr else 1
            momentum = float(self.onnx_attr['momentum'] if "momentum" in self.onnx_attr else 0.99)
            beta_init = self.onnx_attr['beta_init'] if "beta_init" in self.onnx_attr else 'zero'
            gamma_init = self.onnx_attr['gamma_init'] if "gamma_init" in self.onnx_attr else 'one'
            dim_ordering = "th"

            batch_norm= zlayers.BatchNormalization(epsilon=epsilon,
                                                   mode=mode,
                                                   axis=axis,
                                                   momentum=momentum,
                                                   beta_init=beta_init,
                                                   gamma_init=gamma_init,
                                                   dim_ordering=dim_ordering)
            batch_norm.set_running_mean(zlayers.BatchNormalization.get_running_mean(self))
            batch_norm.set_running_std(zlayers.BatchNormalization.get_running_std(self))
            return batch_norm(input.zvalue)
        else:
            raise Exception("not supported.")
