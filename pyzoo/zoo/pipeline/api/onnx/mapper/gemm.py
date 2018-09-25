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
    def __init__(self, node, initializer, _all_tensors):
        super(GemmMapper, self).__init__(node, initializer, _all_tensors)

    def _extract_model_inputs(self):
        return [self._to_zoo_input(self._input_list[0])]

    def _extract_trainable_values(self):
        y = self._input_list[1]
        z = self._input_list[2]

        if "transB" in self.onnx_attr and self.onnx_attr['transB']:
            y.zvalue = np.transpose(y.zvalue)
        alpha = self.onnx_attr["alpha"] if "alpha" in self.onnx_attr else 1.0
        beta = self.onnx_attr["beta"] if "beta" in self.onnx_attr else 1.0
        return [alpha * y.zvalue, beta * z.zvalue]

    def to_zoo_format(self, trainable_values):
        """
        Convert ONNX _initializer to Zoo format
        :return: list of ndarray
        """

        # The format of weight in BigDL is : input * output, so we need to transpose the y here.
        # There's no exception if you don't transpose it
        # as the `set_weights` method doesn't check the shape and respect the total items only.
        return [np.transpose(trainable_values[0]), trainable_values[1]]

    def _to_tensor(self):
        x = self.model_inputs[0]
        z = self.model_trainable_values[1]
        assert len(x.zvalue.shape) == 2, "we only accept 2D input"

        if "transA" in self.onnx_attr and self.onnx_attr['transA']:
            # TODO: add transpose operator for this x = x.transpose()
            raise Exception("we don't support this for now")
        layer = zlayers.Dense(len(z))
        return layer(x.zvalue)
