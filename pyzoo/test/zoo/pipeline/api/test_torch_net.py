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
from torch import nn
import torchvision
import numpy as np
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.net.torch_net import TorchNet


class TestTF(ZooTestCase):
    def test_torch_net_predict_resnet(self):
        model = torchvision.models.resnet18(pretrained=True).eval()
        net = TorchNet.from_pytorch(model, [1, 3, 224, 224])

        dummpy_input = torch.ones(1, 3, 224, 224)
        result = net.predict(dummpy_input.numpy()).collect()
        assert np.allclose(result[0][0:5].tolist(),
                           np.asarray(
                               [-0.03913354128599167, 0.11446280777454376, -1.7967549562454224,
                                -1.2342952489852905, -0.819004476070404]))

    def test_linear_gradient_match(self):

        input = [[1., -0.5], [0.5, -1.]]
        label = [[0.1], [-1]]
        torch_input = torch.tensor(input)
        torch_label = torch.tensor(label)

        model = nn.Linear(2, 1)
        criterion = nn.MSELoss()

        torch_output = model.forward(torch_input)
        torch_loss = criterion.forward(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = model.weight.grad.tolist()[0] + model.bias.grad.tolist()

        torch_net = TorchNet.from_pytorch(model, [1, 2], criterion.forward, pred_shape=[1, 1], label_shape=[1, 1])

        az_output = torch_net.forward(np.array(input))
        az_label = np.array(label)
        az_loss = torch_net.backward(az_output, az_label)
        az_grad = list(torch_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_grad, az_grad.tolist())

    def test_conv2D_gradient_match(self):

        class SimpleTorchModel(nn.Module):
            def __init__(self):
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 48)
                self.conv1 = nn.Conv2d(3, 2, 2)
                self.dense2 = nn.Linear(2, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = x.view(-1, 3, 4, 4)
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = F.sigmoid(self.dense2(x))
                return x


        input = [[1., -0.5], [0.5, -1.]]
        label = [[1., -0.5]]
        torch_input = torch.tensor(input)
        torch_label = torch.tensor(label)

        model = SimpleTorchModel()
        criterion = nn.MSELoss()

        torch_output = model.forward(torch_input)
        torch_loss = criterion.forward(torch_output, torch_label)
        torch_loss.backward()
        torch_grad = model.dense1.weight.grad.flatten().tolist() + model.dense1.bias.grad.flatten().tolist() + \
                     model.conv1.weight.grad.flatten().tolist() + model.conv1.bias.grad.flatten().tolist() + \
                     model.dense2.weight.grad.flatten().tolist() + model.dense2.bias.grad.flatten().tolist()

        torch_net = TorchNet.from_pytorch(model, [1, 2], criterion.forward, pred_shape=[1, 1], label_shape=[1, 1])

        az_output = torch_net.forward(np.array(input))
        az_label = np.array(label)
        az_loss = torch_net.backward(az_output, az_label)
        az_grad = list(torch_net.parameters().values())[0]['gradWeight']

        assert np.allclose(torch_grad, az_grad.tolist())




if __name__ == "__main__":
    pytest.main([__file__])
