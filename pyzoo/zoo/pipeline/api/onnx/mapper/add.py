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


class AddMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(AddMapper, self).__init__(node, _params, _all_tensors)

    def create_operator(self):
        assert len(self.inputs) == 2, "Add should have 2 inputs"
        x = self.inputs[0]
        y = self.inputs[1]
        return x + y
