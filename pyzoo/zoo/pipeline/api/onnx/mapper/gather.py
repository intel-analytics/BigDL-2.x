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
import zoo.pipeline.api.autograd as zautograd
import bigdl.nn.layer as blayer


class GatherMapper(OperatorMapper):
    def __init__(self, node, initializer, _all_tensors):
        super(GatherMapper, self).__init__(node, initializer, _all_tensors)

    def _extract_model_inputs(self):
        return [self._to_zoo_input(i) for i in self._input_list]

    def _to_tensor(self):
        data = self.model_inputs[0].zvalue
        indices = self.model_inputs[1].zvalue

        if self._initializer and isinstance(data, zautograd.Parameter):
            embedding = zlayers.Embedding(input_dim=data.shape[0],
                                          output_dim=data.shape[1],
                                          weights=data.get_weight(),
                                          input_length=indices.shape[1])
            return embedding(indices)
        else:
            dim = int(self.onnx_attr['axis'])
            assert dim >= 1, "Currently only dim>=1 is supported."
            assert indices.shape == (1,), "Currently only one index is supported."
            index = int(indices.get_weight().max())
            return zautograd.expand_dims(data.index_select(dim=dim, index=index), axis=dim)
