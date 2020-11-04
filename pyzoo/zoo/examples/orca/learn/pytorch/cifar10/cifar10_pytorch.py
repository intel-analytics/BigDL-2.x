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

    def forward(self, x, target=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
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
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return trainloader

def validation_data_creator(config):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
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
    args = parser.parse_args()

    #torch.manual_seed(args.seed)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    '''               
    if args.cluster_mode == "local":
        init_orca_context(cores=1, memory="20g")
    elif args.cluster_mode == "yarn":
        init_orca_context(
            cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
            driver_memory="10g", driver_cores=1,
            conf={"spark.rpc.message.maxSize": "1024",
                  "spark.task.maxFailures": "1",
                  "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})
    '''
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores, num_nodes=args.num_nodes, memory=args.memory,  env={"http_proxy": "http://10.239.4.100:913", "https_proxy": "https://10.239.4.100:913"} )

    #model.train()
    criterion = nn.CrossEntropyLoss()
    zoo_estimator = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=criterion, config={}, backend="pytorch")
    zoo_estimator.fit(train_data_creator, epochs=args.epochs)
    zoo_estimator.evaluate(validation_data_creator)
    stop_orca_context()


if __name__ == '__main__':
    main()