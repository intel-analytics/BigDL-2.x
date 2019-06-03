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

import torchvision
import numpy as np
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.net.torch_net import TorchNet


class TestTF(ZooTestCase):

    def test_torch_net_predict(self):
        model = torchvision.models.resnet18()
        net = TorchNet.from_pytorch(model, [1, 3, 224, 224])

        result = net.predict(np.random.rand(1, 3, 224, 224))
        print(result.collect())

if __name__ == "__main__":
    pytest.main([__file__])
