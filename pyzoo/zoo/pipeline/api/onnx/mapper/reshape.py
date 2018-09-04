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


class ReshapeMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(ReshapeMapper, self).__init__(node, _params, _all_tensors)

    def create_operator(self):
        assert len(self.inputs) == 2, "Reshape should have 2 inputs"
        data = self.inputs[0]
        targetshape = self.inputs[1]
        if targetshape[0] == -1 and targetshape[1] == -1:
            raise Exception("not supported.")
        else:
            reshape = zlayers.Reshape(targetshape)
            return reshape
