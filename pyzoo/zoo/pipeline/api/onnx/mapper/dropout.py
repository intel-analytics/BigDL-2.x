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
import zoo.pipeline.api.keras.layers as zlayers
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
from zoo.pipeline.api.onnx.onnx_loader import OnnxLoader


class DropoutMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(DropoutMapper, self).__init__(node, _params, _all_tensors)

    def _to_tensor(self):
        if "mask" in self.onnx_attr:
            raise Exception("We don't support mask for now")
        ratio = float(self.onnx_attr["ratio"])
        if (not OnnxLoader.training) and "is_test" in self.onnx_attr and self.onnx_attr['is_test']:
            return self.model_inputs[0].zvalue
        dropout = zlayers.Dropout(p=ratio)
        return dropout(self.model_inputs[0].zvalue)
