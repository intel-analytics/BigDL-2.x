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


class GemmMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(GemmMapper, self).__init__(node, _params, _all_tensors)

    def format_params(self, params):
        """
        Convert ONNX params to Zoo format
        :return: list of ndarray
        """
        assert len(self.params) == 2
        y = self.params[0]
        z = self.params[1]
        if "transB" in self.onnx_attr and self.onnx_attr['transB']:
            y = np.transpose(y)
        alpha = self.onnx_attr["alpha"] if "alpha" in self.onnx_attr else 1.0
        beta = self.onnx_attr["beta"] if "beta" in self.onnx_attr else 1.0
        # The format of weight in BigDL is : input * output, so we need to transpose the y here.
        # There's no exception if you don't transpose it
        # as the `set_weights` method doesn't check the shape and respect the total items only.
        weights = [np.transpose(alpha * y), beta * z]
        return weights

    def create_operator(self):
        assert len(self.inputs) == 1, "Gemm accept single input only"
        input_shape = self.inputs[0].get_input_shape()
        assert len(input_shape) == 2, "we only accept 2D input"
        x = self.inputs[0]
        z = self.params[1]
        if "transA" in self.onnx_attr and self.onnx_attr['transA']:
            # TODO: add transpose operator for this x = x.transpose()
            raise Exception("we don't support this for now")
        layer = zlayers.Dense(len(z))
        return layer
