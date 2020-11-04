from __future__ import print_function
import os
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch


os.environ['KMP_DUPLICATE_LIB_OK']='True'
           
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
    parser.add_argument('--dir', default='./data', metavar='N',
                        help='the folder store mnist data')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training per executor(default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing per executor(default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
               
    if args.cluster_mode == "local":
        init_orca_context(cores=1, memory="20g")
    elif args.cluster_mode == "yarn":
        init_orca_context(
            cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
            driver_memory="10g", driver_cores=1,
            conf={"spark.rpc.message.maxSize": "1024",
                  "spark.task.maxFailures": "1",
                  "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})
    
    model = Net()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    zoo_estimator = Estimator.from_torch(model=model, optimizer=optimizer, loss=criterion,
                                         backend="bigdl")
    zoo_estimator.fit(data=trainloader, epochs=args.epochs, validation_data=testloader,
                      validation_methods=[Accuracy()], checkpoint_trigger=EveryEpoch())
    zoo_estimator.evaluate(data=testloader, validation_methods=[Accuracy()])
    stop_orca_context()


if __name__ == '__main__':
    main()