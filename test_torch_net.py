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

import torch
import torchvision
import numpy as np
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.net.torch_net import TorchNet


class TestTF(ZooTestCase):
    def test_torch_net_predict(self):
        model = torchvision.models.resnet18(pretrained=True).eval()
        net = TorchNet.from_pytorch(model, [1, 3, 224, 224])

        dummpy_input = torch.ones(1, 3, 224, 224)
        result = net.predict(dummpy_input.numpy()).collect()
        assert np.allclose(result[0][0:5].tolist(),
                           np.asarray(
                               [-0.03913354128599167, 0.11446280777454376, -1.7967549562454224,
                                -1.2342952489852905, -0.819004476070404]))


if __name__ == "__main__":
    pytest.main([__file__])
