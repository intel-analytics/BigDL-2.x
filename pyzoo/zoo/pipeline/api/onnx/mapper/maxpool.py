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


class MaxPoolMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(MaxPoolMapper, self).__init__(node, _params, _all_tensors)

    def format_params(self, params):
        """
        Convert ONNX params to Zoo format
        :return: list of ndarray
        """
        return [np.expand_dims(params[0], 0), params[1]]

    def create_operator(self):
        assert len(self.inputs) == 1, "Conv accept single input only"
        rank = len(self.inputs[0].get_input_shape())

        if (rank == 4):  # NCHWinputs
            pool_size = [int(i) for i in self.onnx_attr['kernel_shape']]
            strides = [int(i) for i in self.onnx_attr['strides']]

            border_mode, pads = OnnxHelper.get_padds(self.onnx_attr)
            if border_mode is None:
                border_mode = "valid"

            maxpool = zlayers.MaxPooling2D(pool_size = pool_size,
                                           strides = strides,
                                           border_mode=border_mode,
                                           )
            return maxpool
        else:
            raise Exception("not supported.")
