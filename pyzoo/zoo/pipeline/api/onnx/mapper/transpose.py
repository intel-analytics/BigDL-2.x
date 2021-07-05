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
import zoo.pipeline.api.keras.layers as zlayers
import bigdl.nn.layer as blayer


class TransposeMapper(OperatorMapper):
    def __init__(self, node, initializer, _all_tensors):
        super(TransposeMapper, self).__init__(node, initializer, _all_tensors)

    def _to_tensor(self):
        data = self.model_inputs[0].zvalue
        if "perm" in self.onnx_attr:
            perm = self.onnx_attr['perm']
        else:
            perm = [int(i) for i in range(len(data.shape)-1, -1, -1)]

        assert perm[0] == 0, "Currently transpose 0 is not supported."
        dims = [int(i) for i in perm[1:]]
        return zlayers.Permute(dims)(data)
