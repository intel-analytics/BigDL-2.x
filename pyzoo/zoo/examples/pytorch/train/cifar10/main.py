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
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.pipeline.estimator import *
from bigdl.optim.optimizer import SGD, Adam
from zoo.common.nncontext import *
from zoo.feature.common import FeatureSet
from zoo.pipeline.api.keras.metrics import Accuracy


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--dir', default='/tmp/data', metavar='N',
                        help='the folder store cifar10 data')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training per executor(default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing per executor(default: 1000)')
    parser.add_argument('--epochs', type=int, default=135, metavar='N',
                        help='number of epochs to train (default: 135)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lrd', type=float, default=0.0, metavar='LRD',
                        help='learning rate decay(default: 0.0)')
    parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                        help='weight decay(default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                        help='momentum (default: 0.9)')
    parser.add_argument('--dampening', type=float, default=0.0, metavar='dampening',
                        help='dampening (default: 0.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 准备数据并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(args.dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_set = datasets.CIFAR10(args.dir, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    # init on yarn when HADOOP_CONF_DIR and ZOO_CONDA_NAME is provided.
    if os.environ.get('HADOOP_CONF_DIR') is None:
        sc = init_spark_on_local(cores=1, conf={"spark.driver.memory": "20g"})
    else:
        num_executors = 2
        num_cores_per_executor = 4
        hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
        zoo_conda_name = os.environ.get('ZOO_CONDA_NAME')  # The name of the created conda-env
        sc = init_spark_on_yarn(
            hadoop_conf=hadoop_conf_dir,
            conda_name=zoo_conda_name,
            num_executor=num_executors,
            executor_cores=num_cores_per_executor,
            executor_memory="2g",
            driver_memory="10g",
            driver_cores=1,
            spark_conf={"spark.rpc.message.maxSize": "1024",
                        "spark.task.maxFailures": "1",
                        "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    model = ResNet18()
    model.train()
    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(args.lr, args.lrd, args.wd, args.momentum, args.dampening)
    zoo_model = TorchModel.from_pytorch(model)
    zoo_criterion = TorchLoss.from_pytorch(criterion)
    zoo_estimator = Estimator(zoo_model, optim_methods=optimizer)
    train_featureset = FeatureSet.pytorch_dataloader(train_loader)
    test_featureset = FeatureSet.pytorch_dataloader(test_loader)
    from bigdl.optim.optimizer import MaxEpoch, EveryEpoch
    zoo_estimator.train_minibatch(train_featureset, zoo_criterion,
                                  end_trigger=MaxEpoch(args.epochs),
                                  checkpoint_trigger=EveryEpoch(),
                                  validation_set=test_featureset,
                                  validation_method=[Accuracy()])


if __name__ == '__main__':
    main()
