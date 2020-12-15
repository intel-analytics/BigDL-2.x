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
# ==============================================================================
# Most of the Pytorch code is adapted from Pytorch's tutorial for
# visualizing training with tensorboard
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
#
from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator


def train_data_creator(config):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST('./data',
                                                 download=True,
                                                 train=True,
                                                 transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainloader


def validation_data_creator(config):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return testloader


# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


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


def main():
    parser = argparse.ArgumentParser(description='PyTorch Tensorboard Example')

    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or k8s.')
    args = parser.parse_args()
    if args.cluster_mode == "local":
        init_orca_context()
    elif args.cluster_mode == "yarn":
        init_orca_context(cluster_mode=args.cluster_mode, cores=4, num_nodes=2)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # plot some random training images
    dataiter = iter(train_data_creator(config={}))
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)

    # inspect the model using tensorboard
    writer.add_graph(model_creator(config={}), images)
    writer.close()

    # training loss vs. epochs
    criterion = nn.CrossEntropyLoss()
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optimizer_creator,
                                          loss=criterion,
                                          backend="torch_distributed")
    stats = orca_estimator.fit(train_data_creator, epochs=5, batch_size=4)

    for stat in stats:
        writer.add_scalar("training_loss", stat['train_loss'], stat['epoch'])
    print("Train stats: {}".format(stats))
    val_stats = orca_estimator.evaluate(validation_data_creator)
    print("Validation stats: {}".format(val_stats))
    orca_estimator.shutdown()

    stop_orca_context()

if __name__ == '__main__':
    main()
