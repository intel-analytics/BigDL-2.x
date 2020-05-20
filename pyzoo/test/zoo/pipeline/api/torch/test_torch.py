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
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytest

from unittest import TestCase
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.common.nncontext import *


class TestPytorch(TestCase):

    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_spark_on_local(4)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_torchmodel_constructor(self):
        class TwoInputModel(nn.Module):
            def __init__(self):
                super(TwoInputModel, self).__init__()
                self.dense1 = nn.Linear(2, 2)
                self.dense2 = nn.Linear(3, 1)

            def forward(self, x1, x2):
                x1 = self.dense1(x1)
                x2 = self.dense2(x2)
                return x1, x2

        TorchModel.from_pytorch(TwoInputModel())

    def test_torchloss_constructor(self):
        # two inputs test
        criterion = nn.MSELoss()

        TorchLoss.from_pytorch(criterion)

    def test_torch_net_predict_resnet(self):
        torch.random.manual_seed(1)
        pytorch_model = torchvision.models.resnet18(pretrained=False).eval()
        zoo_model = TorchModel.from_pytorch(pytorch_model)
        zoo_model.evaluate()

        dummy_input = torch.ones(1, 3, 224, 224)
        pytorch_result = pytorch_model(dummy_input).data.numpy()
        zoo_result = zoo_model.forward(dummy_input.numpy())
        print(pytorch_result)
        print(zoo_result)
        assert np.allclose(pytorch_result, zoo_result, rtol=1.e-6, atol=1.e-6)

    #def test_linear_gradient_match(self):
    #    input = [[0.5, 1.], [-0.3, 1.2]]
    #    label = [[0.6], [-0.9]]
    #    torch_input = torch.tensor(input)
    #    torch_label = torch.tensor(label)

    #    model = nn.Linear(2, 1)
    #    criterion = nn.MSELoss()

    #    torch_output = model.forward(torch_input)
    #    torch_loss = criterion.forward(torch_output, torch_label)
    #    torch_loss.backward()
    #    torch_grad = model.weight.grad.tolist()[0] + model.bias.grad.tolist()

    #    # AZ part
    #    az_net = TorchModel.from_pytorch(model)
    #    az_criterion = TorchLoss.from_pytorch(criterion)

    #    az_input = np.array(input)
    #    az_label = np.array(label)

    #    az_output = az_net.forward(az_input)
    #    az_loss_output = az_criterion.forward(az_output, az_label)
    #    az_loss_backward = az_criterion.backward(az_output, az_label)
    #    az_model_backward = az_net.backward(az_input, az_loss_backward)

    #    az_grad = list(az_net.parameters().values())[0]['gradWeight']

    #    assert np.allclose(torch_loss.tolist(), az_loss_output)
    #    assert np.allclose(torch_grad, az_grad.tolist())

    #def test_conv2D_gradient_match(self):
    #    class SimpleTorchModel(nn.Module):
    #        def __init__(self):
    #            super(SimpleTorchModel, self).__init__()
    #            self.dense1 = nn.Linear(2, 48)
    #            self.conv1 = nn.Conv2d(3, 2, 2)
    #            self.dense2 = nn.Linear(2, 1)

    #        def forward(self, x):
    #            x = self.dense1(x)
    #            x = x.view(-1, 3, 4, 4)
    #            x = torch.relu(self.conv1(x))
    #            x = F.max_pool2d(x, 2)
    #            x = x.view(x.size(0), -1)
    #            x = torch.sigmoid(self.dense2(x))
    #            return x

    #    input = [[1., -0.5], [0.5, -1.]]
    #    label = [[1., -0.5]]
    #    torch_input = torch.tensor(input)
    #    torch_label = torch.tensor(label)

    #    torch_model = SimpleTorchModel()
    #    torch_criterion = nn.MSELoss()

    #    torch_output = torch_model.forward(torch_input)
    #    torch_loss = torch_criterion.forward(torch_output, torch_label)
    #    torch_loss.backward()
    #    torch_grad = torch_model.dense1.weight.grad.flatten().tolist() + \
    #        torch_model.dense1.bias.grad.flatten().tolist() + \
    #        torch_model.conv1.weight.grad.flatten().tolist() + \
    #        torch_model.conv1.bias.grad.flatten().tolist() + \
    #        torch_model.dense2.weight.grad.flatten().tolist() + \
    #        torch_model.dense2.bias.grad.flatten().tolist()

    #    # AZ part
    #    az_net = TorchModel.from_pytorch(torch_model)
    #    az_criterion = TorchLoss.from_pytorch(torch_criterion)

    #    az_input = np.array(input)
    #    az_label = np.array(label)
    #    az_output = az_net.forward(np.array(input))
    #    az_loss_output = az_criterion.forward(az_output, az_label)
    #    az_loss_backward = az_criterion.backward(az_output, az_label)
    #    az_model_backward = az_net.backward(az_input, az_loss_backward)

    #    az_grad = list(az_net.parameters().values())[0]['gradWeight']

    #    assert np.allclose(torch_loss.tolist(), az_loss_output)
    #    assert np.allclose(torch_grad, az_grad.tolist())

    #def test_cross_entrophy_match(self):
    #    input = [[0.5, 1.], [-0.3, 1.2]]
    #    label = [3, 6]
    #    torch_input = torch.tensor(input)
    #    torch_label = torch.tensor(label).long()

    #    model = nn.Linear(2, 10)
    #    criterion = nn.CrossEntropyLoss()

    #    torch_output = model.forward(torch_input)
    #    torch_loss = criterion.forward(torch_output, torch_label)
    #    torch_loss.backward()
    #    torch_grad = model.weight.grad.flatten().tolist() + model.bias.grad.tolist()

    #    # AZ part
    #    az_net = TorchModel.from_pytorch(model)
    #    az_criterion = TorchLoss.from_pytorch(criterion)

    #    az_input = np.array(input)
    #    az_label = np.array(label)

    #    az_output = az_net.forward(az_input)
    #    az_loss_output = az_criterion.forward(az_output, az_label)
    #    az_loss_backward = az_criterion.backward(az_output, az_label)
    #    az_model_backward = az_net.backward(az_input, az_loss_backward)

    #    az_grad = list(az_net.parameters().values())[0]['gradWeight']

    #    assert np.allclose(torch_loss.tolist(), az_loss_output)
    #    assert np.allclose(torch_grad, az_grad.tolist())

    #def test_Lenet_gradient_match(self):
    #    class LeNet(nn.Module):
    #        def __init__(self):
    #            super(LeNet, self).__init__()
    #            self.conv1 = nn.Conv2d(1, 20, 5, 1)
    #            self.conv2 = nn.Conv2d(20, 50, 5, 1)
    #            self.fc1 = nn.Linear(4 * 4 * 50, 500)
    #            self.fc2 = nn.Linear(500, 10)

    #        def forward(self, x):
    #            x = F.relu(self.conv1(x))
    #            x = F.max_pool2d(x, 2, 2)
    #            x = F.relu(self.conv2(x))
    #            x = F.max_pool2d(x, 2, 2)
    #            x = x.view(-1, 4 * 4 * 50)
    #            x = F.relu(self.fc1(x))
    #            x = self.fc2(x)
    #            return F.log_softmax(x, dim=1)

    #    input = np.random.rand(2, 1, 28, 28)
    #    label = [7, 3]
    #    torch_input = torch.tensor(input).float()
    #    torch_label = torch.tensor(label).long()

    #    torch_model = LeNet()
    #    torch_criterion = nn.CrossEntropyLoss()

    #    torch_output = torch_model.forward(torch_input)
    #    torch_loss = torch_criterion.forward(torch_output, torch_label)
    #    torch_loss.backward()
    #    torch_grad = torch_model.conv1.weight.grad.flatten().tolist() + \
    #        torch_model.conv1.bias.grad.flatten().tolist() + \
    #        torch_model.conv2.weight.grad.flatten().tolist() + \
    #        torch_model.conv2.bias.grad.flatten().tolist() + \
    #        torch_model.fc1.weight.grad.flatten().tolist() + \
    #        torch_model.fc1.bias.grad.flatten().tolist() + \
    #        torch_model.fc2.weight.grad.flatten().tolist() + \
    #        torch_model.fc2.bias.grad.flatten().tolist()

    #    # AZ part
    #    az_net = TorchModel.from_pytorch(torch_model)

    #    def lossFunc(input, target):
    #        return torch_criterion.forward(input, target.flatten().long())

    #    az_criterion = TorchLoss.from_pytorch(lossFunc, [1, 10], [1, 1])

    #    az_input = np.array(input)
    #    az_label = np.array(label)
    #    az_output = az_net.forward(np.array(input))
    #    az_loss_output = az_criterion.forward(az_output, az_label)
    #    az_loss_backward = az_criterion.backward(az_output, az_label)
    #    az_model_backward = az_net.backward(az_input, az_loss_backward)

    #    az_grad = list(az_net.parameters().values())[0]['gradWeight']

    #    assert np.allclose(torch_loss.tolist(), az_loss_output)
    #    assert np.allclose(torch_grad, az_grad.tolist(), atol=1.e-5, rtol=1.e-3)

    #def test_model_inference_with_multiple_output(self):
    #    class TwoOutputModel(nn.Module):
    #        def __init__(self):
    #            super(TwoOutputModel, self).__init__()
    #            self.dense1 = nn.Linear(2, 1)

    #        def forward(self, x):
    #            x1 = self.dense1(x)
    #            return x, x1

    #    torch_model = TwoOutputModel()
    #    az_net = TorchModel.from_pytorch(TwoOutputModel())

    #    az_input = np.array([[0.5, 1.], [-0.3, 1.2]])
    #    az_output = az_net.forward(az_input)
    #    assert (len(az_output) == 2)
    #    assert (az_output[0].shape == (2, 2))
    #    assert (az_output[1].shape == (2, 1))

    def test_model_to_pytorch(self):
        class SimpleTorchModel(nn.Module):
            def __init__(self):
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.dense1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        torch_model = SimpleTorchModel()

        az_model = TorchModel.from_pytorch(torch_model)
        weights = az_model.get_weights()
        weights[0][0] = 1.0
        az_model.set_weights(weights)
        exported_model = az_model.to_pytorch()
        p = list(exported_model.parameters())
        print(p[0])
        assert p[0][0][0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
