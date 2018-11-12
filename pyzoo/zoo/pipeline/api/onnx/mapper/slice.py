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
from zoo.pipeline.api.autograd import Parameter
import bigdl.nn.layer as blayer

class IndexSelectMapper(OperatorMapper):
    def __init__(self, node, initializer, _all_tensors):
        super(IndexSelectMapper, self).__init__(node, initializer, _all_tensors)

    def _to_tensor(self):
        dim = 1
        index = 1

        if "dim" in self.onnx_attr.keys():
            dim = int(self.onnx_attr['dim'])
        assert dim >= 0, "Currently dim>=0 is required."
        assert dim == 1, "Currently dim=0 is not supported"

        data = self.model_inputs[0].zvalue

        return data.index_select(dim=dim, index=index)


