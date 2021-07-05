# -*- coding: utf-8 -*
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
from zoo.pipeline.api.onnx.mapper.operator_mapper import OperatorMapper
import zoo.pipeline.api.keras.layers as zlayers
import zoo.pipeline.api.autograd as autograd
import bigdl.nn.layer as blayer


class SliceMapper(OperatorMapper):
    def __init__(self, node, _params, _all_tensors):
        super(SliceMapper, self).__init__(node, _params, _all_tensors)

    def _extract_model_inputs(self):
        """
        :return: list of OnnxInput
        """
        return [self._to_zoo_input(self._input_list[0])]

    def _to_tensor(self):
        lens = []
        input = self.model_inputs[0].zvalue
        ends = self.onnx_attr['ends']
        starts = self.onnx_attr['starts']
        if "axes" in self.onnx_attr.keys():
            axes = self.onnx_attr['axes']
        else:
            axes = range(len(starts))
        for j in range(len(starts)):
            lens.append(ends[j] - starts[j])
            # slice(, len=-1) equals to slice(, len=length)
            # y = x[:2,0:-1] means start is(0,0) and ends is(2,-1)
            # which is equivalent to slice(,len=-2) as "end=-1" is exclusive here.
            if lens[j] < 0:
                lens[j] -= 1
        for i in range(len(starts)):
            input = input.slice(dim=int(axes[i]), start_index=int(starts[i]), length=int(lens[i]))
        return input
