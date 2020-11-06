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

import os
import pytest
import numpy as np

from unittest import TestCase
from zoo.pipeline.inference import InferenceModel

from zoo.pipeline.api.torch import zoo_pickle_module
import torch
import torchvision
from bigdl.util.common import create_tmp_path


class TestInferenceModelTorch(TestCase):

    def test_load_torch(self):
        torch_model = torchvision.models.resnet18()
        tmp_path = create_tmp_path() + ".pt"
        torch.save(torch_model, tmp_path, pickle_module=zoo_pickle_module)
        model = InferenceModel(10)
        model.load_torch(tmp_path)
        input_data = np.random.random([4, 3, 224, 224])
        output_data = model.predict(input_data)
        os.remove(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__])
