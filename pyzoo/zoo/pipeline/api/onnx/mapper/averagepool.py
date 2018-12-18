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


class AveragePoolMapper (OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(AveragePoolMapper, self).__init__(node, _params, _all_tensors)

    def _to_tensor(self):
        assert len(self.model_inputs) == 1, "AveragePool accept single input only"
        rank = len(self.model_inputs[0].zvalue.shape)
        if (rank == 4):  # NCHW
            poolSize = [int(i) for i in self.onnx_attr['kernel_shape']]
            strides = [int(i) for i in self.onnx_attr['strides']]
            count_include_pad = bool(self.onnx_attr['count_include_pad'])\
                if "count_include_pad" in self.onnx_attr else False
            dim_ordering = "th"
            border_mode, pads = OnnxHelper.get_padds(self.onnx_attr)
            averagepool2d = zlayers.AveragePooling2D(pool_size=poolSize,
                                                     strides=strides,
                                                     dim_ordering=dim_ordering,
                                                     pads=pads,
                                                     count_include_pad=count_include_pad)
            return averagepool2d(self.model_inputs[0].zvalue)
        else:
            raise Exception("not supported.")
