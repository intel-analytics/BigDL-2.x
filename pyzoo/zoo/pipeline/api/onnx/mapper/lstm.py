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

import numpy as np
import zoo.pipeline.api.keras.layers as zlayers
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper


class LSTMMapper(OperatorMapper):
    def __init__(self, node, initializer, _all_tensors):
        super(LSTMMapper, self).__init__(node, initializer, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        return [self._to_zoo_input(self._input_list[0])]

    def _extract_trainable_values(self):
        if len(self._input_list) == 5:
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
        output_dim = self.onnx_attr['output_dim']
        W_regularizer = self.onnx_attr['W_regularizer']
        U_regularizer = self.onnx_attr['U_regularizer']
        b_regularizer = self.onnx_attr['b_regularizer']
        dropout_W = self.onnx_attr['dropout_W']
        dropout_U = self.onnx_attr['dropout_U']
        lstm = zlayers.recurrent.LSTM(output_dim=output_dim,
                                      W_regularizer=W_regularizer,
                                      U_regularizer=U_regularizer,
                                      b_regularizer=b_regularizer,
                                      dropout_U=dropout_U,
                                      dropout_W=dropout_W)
        lstm_tensor = lstm(input.zvalue)
        return lstm_tensor
