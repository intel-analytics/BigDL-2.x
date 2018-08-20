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

np.random.seed(1337)  # for reproducibility


class OnnxTestCase(ZooTestCase):

    def dump_pytorch_to_onnx(self, pytorch_model, input_shape_with_batch):
        import uuid
        tmp_dir = self.create_temp_dir()
        onnx_path = "%s/%s" % (tmp_dir,  uuid.uuid4().hex + ".onnx")
        dummy_input = torch.autograd.Variable(torch.randn(input_shape_with_batch))

        torch.onnx.export(pytorch_model, dummy_input, onnx_path)
        print("Creating an Onnx model: " + onnx_path)
        return onnx_path

    def compare_with_pytorch(self, pytorch_model, input_shape_with_batch):
        onnx_path = self.dump_pytorch_to_onnx(pytorch_model, input_shape_with_batch)
        onnx_model = onnx.load(onnx_path)
        # TODO: we only consider single input and output for now.
        input_data_with_batch = np.random.uniform(0, 1, input_shape_with_batch).astype(np.float32)
        # coutput = caffe2.python.onnx.backend.run_model(onnx_model, input_data_with_batch)[0]
        pytorch_out = pytorch_model.forward(torch.from_numpy(input_data_with_batch))
        zmodel = OnnxLoader(onnx_model.graph).to_keras()
        zoutput = zmodel.forward(input_data_with_batch)
        self.assert_allclose(pytorch_out.detach().numpy(), zoutput)
