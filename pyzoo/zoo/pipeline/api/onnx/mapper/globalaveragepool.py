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
import zoo.pipeline.api.autograd as zautograd
import numpy as np


class GlobalAveragePoolMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(GlobalAveragePoolMapper, self).__init__(node, _params, _all_tensors)

    def _to_tensor(self):
        x = self.model_inputs[0].zvalue
        y = zlayers.GlobalAveragePooling2D()(x)
        '''
        Input data tensor from the previous operator; dimensions for image case are (N x C x H x W),
        where N is the batch size, C is the number of channels, and H and W are the height
        and the width of the data.
        Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1.
        '''
        return zautograd.expand_dims(zautograd.expand_dims(y, axis=2), axis=3)
