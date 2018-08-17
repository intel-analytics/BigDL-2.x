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


class ConvMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(ConvMapper, self).__init__(node, _params, _all_tensors)

    def create_operator(self):
        assert len(self.inputs) == 1, "Conv accept single input only"
        rank = len(self.inputs[0].get_input_shape())
        W_weights = self.params[0]
        if (rank == 4):  # NCHW
            nb_filter = W_weights.shape[0]
            nb_row = int(self.onnx_attr['kernel_shape'][0])
            nb_col = int(self.onnx_attr['kernel_shape'][1])
            subSample = [int(i) for i in self.onnx_attr['strides']]
            dim_ordering = "th"
            assert self.onnx_attr['dilations'] == (1, 1), "we only support dilations == (1, 1)"
            assert self.onnx_attr['group'] == 1, "we only support group == 1"
            bias = True if (len(self.params) > 1) else False

            # TODO: activation?? init?? W_regularizer??
            border_mode, pads = OnnxHelper.get_padds(self.onnx_attr)

            conv = zlayers.Convolution2D(nb_filter=nb_filter,
                                         nb_row=nb_row,
                                         nb_col=nb_col,
                                         subsample=subSample,
                                         dim_ordering=dim_ordering,
                                         bias=bias,
                                         border_mode=border_mode,
                                         pads=pads)
            return conv
        else:
            raise Exception("not supported.")
