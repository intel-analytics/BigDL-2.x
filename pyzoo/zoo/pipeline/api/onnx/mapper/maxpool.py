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

    def _to_tensor(self):
        assert len(self.model_inputs) == 1, "MaxPool accept single input only"
        rank = len(self.model_inputs[0].zvalue.shape)
        if "storage_order" in self.onnx_attr.keys():
            assert self.onnx_attr['storage_order'] == 0

        if (rank == 4):  # NCHW
            pool_size = [int(i) for i in self.onnx_attr['kernel_shape']]
            if "strides" in self.onnx_attr.keys():
                strides = [int(i) for i in self.onnx_attr['strides']]
            else:
                strides = [1 for i in self.onnx_attr['kernel_shape']]

            border_mode, pads = OnnxHelper.get_padds(self.onnx_attr)

            maxpool = zlayers.MaxPooling2D(pool_size=pool_size,
                                           strides=strides,
                                           border_mode=border_mode,
                                           pads=pads)
            return maxpool(self.model_inputs[0].zvalue)
        elif (rank == 3):
            pool_length = int(self.onnx_attr['kernel_shape'][0])
            if "strides" in self.onnx_attr.keys():
                stride = int(self.onnx_attr['strides'][0])
            else:
                stride = 1
            border_mode, pads = OnnxHelper.get_padds(self.onnx_attr)
            if border_mode is None and pads is None:
                border_mode = 'valid'
            if pads is None:
                pads = 0
            permute = zlayers.Permute(dims=(2, 1))(self.model_inputs[0].zvalue)
            maxpool = zlayers.MaxPooling1D(pool_length=pool_length,
                                           stride=stride,
                                           border_mode=border_mode,
                                           pad=pads)(permute)
            return zlayers.Permute(dims=(2, 1))(maxpool)
        else:
            raise Exception("not supported.")
