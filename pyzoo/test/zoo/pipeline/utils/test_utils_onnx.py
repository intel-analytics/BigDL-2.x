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

from __future__ import print_function

import numpy as np
import onnx
import torch

from zoo.pipeline.api.onnx.onnx_loader import OnnxLoader
from .test_utils import ZooTestCase
from bigdl.util.common import to_list

np.random.seed(1337)  # for reproducibility


class OnnxTestCase(ZooTestCase):

    def dump_pytorch_to_onnx(self, pytorch_model, input_shape_with_batch):
        import uuid
        tmp_dir = self.create_temp_dir()
        onnx_path = "%s/%s" % (tmp_dir, uuid.uuid4().hex + ".onnx")
        dummy_input = self._generate_pytorch_input(input_shape_with_batch)
        torch.onnx.export(pytorch_model, dummy_input, onnx_path)
        print("Creating an Onnx model: " + onnx_path)
        return onnx_path

    def _generate_pytorch_input(self, input_shape_with_batch):
        dummy_input = [torch.autograd.Variable(torch.randn(shape)) for shape in
                       input_shape_with_batch]
        if len(dummy_input) == 1:
            dummy_input = dummy_input[0]
        return dummy_input

    def _convert_ndarray_to_tensor(self, input_data_with_batch):
        tensors = [torch.from_numpy(input) for input in input_data_with_batch]
        if len(tensors) == 1:
            return tensors[0]
        else:
            return tensors

    def compare_with_pytorch(self, pytorch_model, input_shape_with_batch):
        input_shape_with_batch = to_list(input_shape_with_batch)
        onnx_path = self.dump_pytorch_to_onnx(pytorch_model, input_shape_with_batch)
        onnx_model = onnx.load(onnx_path)
        # TODO: we only consider single  output for now.
        input_data_with_batch = [np.random.uniform(0, 1, shape).astype(np.float32) for shape in
                                 input_shape_with_batch]
        # coutput = caffe2.python.onnx.backend.run_model(onnx_model, input_data_with_batch)[0]

        pytorch_out = pytorch_model.forward(self._convert_ndarray_to_tensor(input_data_with_batch))
        zmodel = OnnxLoader(onnx_model.graph).to_keras()
        zoutput = zmodel.forward(
            input_data_with_batch[0] if len(input_data_with_batch) == 1 else input_data_with_batch)
        self.assert_allclose(pytorch_out.detach().numpy(), zoutput)

    def gen_rnd(self, shape, low=-1.0, high=1.0):
        return np.random.uniform(low, high, np.prod(shape)) \
            .reshape(shape) \
            .astype(np.float32)
