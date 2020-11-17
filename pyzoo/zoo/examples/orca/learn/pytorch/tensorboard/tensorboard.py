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
import os
import argparse
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def model_creator(config):
    model = Net()
    return model
    
def optimizer_creator(model, config):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer
    
def train_data_creator(config):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=True,
        transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.get("batch_size",32),
                                              shuffle=True, num_workers=2)
    return trainloader
    
def validation_data_creator(config):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.get("batch_size", 32),
                                             shuffle=False, num_workers=2)
    return testloader
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser = argparse.ArgumentParser()

    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster.')
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="The number of nodes to be used in the cluster. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--cores", type=int, default=4,
                        help="The number of cpu cores you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--memory", type=str, default="10g",
                        help="The memory you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument("--workers_per_node", type=int, default=1,
                        help="The number of workers to run on each node")
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input worker batch for training per executor(default: 16)')
    args = parser.parse_args()
    
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores, num_nodes=args.num_nodes, memory=args.memory)
    
    criterion = nn.CrossEntropyLoss()
    zoo_estimator = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=criterion, config={}, backend="pytorch")
    stats = zoo_estimator.fit(train_data_creator, epochs=args.epochs, batch_size=args.batch_size)
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    print("start")
    for stat in stats:
        writer.add_scalar("training_loss", stat['train_loss'], stat['epoch'])
    print("Train stats: {}".format(stats))
    val_stats = zoo_estimator.evaluate(validation_data_creator)
    print("validation stats: {}".format(val_stats))
    stop_orca_context()

if __name__ == '__main__':
    main()
