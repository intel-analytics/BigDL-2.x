from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from zoo.pipeline.api.net.torch_net import TorchModel, TorchLoss
from zoo.pipeline.estimator import *
from bigdl.optim.optimizer import SGD, Step
from zoo.pipeline.api.keras.optimizers import Adam
from zoo.common.nncontext import *
from zoo.feature.common import FeatureSet
from bigdl.optim.optimizer import Loss
from zoo.pipeline.api.keras.metrics import Accuracy
import time
from zoo.pipeline.api.keras.objectives import SparseCategoricalCrossEntropy

model_names = sorted(name for name in torchvision.models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(torchvision.models.__dict__[name]))

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--cores', default=1, type=int,
                        help='num of CPUs to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    # sc = init_nncontext()
    sc = init_spark_on_local(cores=args.cores, conf={"spark.driver.memory": "20g"})
    # hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
    # num_executors = 5
    # num_cores_per_executor = args.cores
    # os.environ['ZOO_MKL_NUMTHREADS'] = str(num_cores_per_executor)
    # os.environ['OMP_NUM_THREADS'] = str(num_cores_per_executor)
    # sc = init_spark_on_yarn(
    #     hadoop_conf=hadoop_conf_dir,
    #     conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
    #     num_executor=num_executors,
    #     executor_cores=num_cores_per_executor,
    #     executor_memory="20g",
    #     driver_memory="20g",
    #     driver_cores=1,
    #     spark_conf={"spark.rpc.message.maxSize": "1024",
    #                 "spark.task.maxFailures":  "1",
    #                 "spark.scheduler.minRegisteredResourcesRatio": "1",
    #                 "spark.scheduler.maxRegisteredResourcesWaitingTime": "100s",
    #                 "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    # Data loading code
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True,
                                            download=True, transform=transform)
    # train_loader = torch.utils.py.data.DataLoader(trainset, batch_size=100,
    #                                           shuffle=True, num_workers=0)
    
    valset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False,
                                           download=True, transform=transforms.ToTensor())
    # val_loader = torch.utils.py.data.DataLoader(valset, batch_size=100,
    #                                      shuffle=False, num_workers=0)
    # 3x3 convolution
    def conv3x3(in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=stride, padding=1, bias=False)
    
    # Residual block
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = conv3x3(in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample
    
        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out
    
    # ResNet
    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=10):
            super(ResNet, self).__init__()
            self.in_channels = 16
            self.conv = conv3x3(3, 16)
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self.make_layer(block, 16, layers[0])
            self.layer2 = self.make_layer(block, 32, layers[1], 2)
            self.layer3 = self.make_layer(block, 64, layers[2], 2)
            self.avg_pool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64, num_classes)
    
        def make_layer(self, block, out_channels, blocks, stride=1):
            downsample = None
            if (stride != 1) or (self.in_channels != out_channels):
                downsample = nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride=stride),
                    nn.BatchNorm2d(out_channels))
            layers = []
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for i in range(1, blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)
    
        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    model = ResNet(ResidualBlock, [2, 2, 2])
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    lr_schedule = Step(20*500, 1.0/3)
    adam = Adam(lr=0.001, schedule=lr_schedule)
    zooModel = TorchModel.from_pytorch(model)
    zooCriterion = TorchLoss.from_pytorch(criterion)

    estimator = Estimator(zooModel, optim_methods=adam)
    train_featureSet = FeatureSet.pytorch_dataset(trainset, batch_size=100, shuffle=True)
    test_featureSet = FeatureSet.pytorch_dataset(valset, batch_size=100, shuffle=False)
    c = train_featureSet.to_dataset().size()
    print(c)
    # estimator.evaluate_minibatch(train_featureSet, [Accuracy()])
    from bigdl.optim.optimizer import MaxEpoch, EveryEpoch, MaxIteration
    from zoo.pipeline.api.keras.metrics import Accuracy
    estimator.train_minibatch(train_featureSet, zooCriterion, end_trigger=MaxEpoch(80), checkpoint_trigger=EveryEpoch(),
                              validation_set=test_featureSet, validation_method=[Accuracy()])
    #Estimator.print()

if __name__ == '__main__':
    main()
