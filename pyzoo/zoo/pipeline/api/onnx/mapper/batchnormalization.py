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
import zoo.pipeline.api.autograd as autograd
import numpy as np


class BatchNormalizationMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(BatchNormalizationMapper, self).__init__(node, _params, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        return [self._to_zoo_input(self._input_list[0])]

    def _extract_trainable_values(self):
        if len(self._input_list) == 5:
            if isinstance(self._input_list[1].zvalue, autograd.Parameter):
                return [self._input_list[1].zvalue.get_weight(),
                        self._input_list[2].zvalue.get_weight()]
            else:
                return [self._input_list[1].zvalue, self._input_list[2].zvalue]
        else:
            return None

    def to_zoo_format(self, trainable_values):
        """
        Convert ONNX _initializer to Zoo format
        :return: list of ndarray
        """
        return trainable_values

    def _to_tensor(self):
        input = self.model_inputs[0]
        rank = len(input.zvalue.shape)

        if (rank == 4):
            epsilon = float(self.onnx_attr['epsilon']) if "epsilon" in self.onnx_attr else 0.001
            momentum = float(self.onnx_attr['momentum'] if "momentum" in self.onnx_attr else 0.99)
            dim_ordering = "th"
            if len(self._input_list) == 5:
                mean = self._input_list[3].zvalue
                variance = self._input_list[4].zvalue
            else:
                mean = self._input_list[1].zvalue
                variance = self._input_list[2].zvalue
            batch_norm = zlayers.BatchNormalization(epsilon=epsilon,
                                                    momentum=momentum,
                                                    dim_ordering=dim_ordering)
            norm_tensor = batch_norm(input.zvalue)
            norm_tensor.node.element().set_running_mean(mean)
            norm_tensor.node.element().set_running_std(variance)
            return norm_tensor
        else:
            raise Exception("not supported.")
