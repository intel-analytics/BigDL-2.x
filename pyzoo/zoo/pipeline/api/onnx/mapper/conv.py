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


class ConvMapper(OperatorMapper):
    def __init__(self, node, initializer, inputs):
        super(ConvMapper, self).__init__(node, initializer, inputs)

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
        input = self.model_inputs[0]
        W_weights = self.model_trainable_values[0]
        rank = len(input.zvalue.shape)

        if (rank == 4):  # NCHW
            nb_filter = W_weights.shape[0]
            nb_row = int(self.onnx_attr['kernel_shape'][0])
            nb_col = int(self.onnx_attr['kernel_shape'][1])
            subSample = [int(i) for i in
                         self.onnx_attr['strides']] if "strides" in self.onnx_attr else (1, 1)
            dim_ordering = "th"
            assert 'dilations' not in self.onnx_attr or self.onnx_attr['dilations'] == (
                1, 1), "we only support dilations == (1, 1)"
            assert 'group' not in self.onnx_attr or self.onnx_attr[
                'group'] == 1, "we only support group == 1"
            bias = True if (len(self._input_list) > 2) else False

            border_mode, pads = OnnxHelper.get_padds(self.onnx_attr)

            conv = zlayers.Convolution2D(nb_filter=nb_filter,
                                         nb_row=nb_row,
                                         nb_col=nb_col,
                                         subsample=subSample,
                                         dim_ordering=dim_ordering,
                                         bias=bias,
                                         border_mode=border_mode,
                                         pads=pads)
            return conv(input.zvalue)
        else:
            raise Exception("not supported.")
