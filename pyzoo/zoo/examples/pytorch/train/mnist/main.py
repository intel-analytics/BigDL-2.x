from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.pipeline.estimator import *
from bigdl.optim.optimizer import SGD, Adam
from zoo.common.nncontext import *
from zoo.feature.common import FeatureSet
from bigdl.optim.optimizer import Loss
from zoo.pipeline.api.keras.metrics import Accuracy
from torchnet.dataset.splitdataset import SplitDataset
import time
from zoo.pipeline.api.keras.objectives import SparseCategoricalCrossEntropy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # data_set = datasets.MNIST('/tmp/data', train=True, download=True,
    #                           transform=transforms.Compose([
    #                               transforms.ToTensor(),
    #                               transforms.Normalize((0.1307,), (0.3081,))
    #                           ]))
    # partitioner = {'1': 0.1, '2': 0.5}
    # sdata_set = SplitDataset(data_set, partitioner)
    # sdata_set.select('1')
    # train_loader = torch.utils.py.data.DataLoader(
    #     sdata_set,
    #     batch_size=args.batch_size, shuffle=True)
    # print(len(train_loader))
    # a = enumerate(train_loader)
    # print(next(a))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False)

    # sc = init_spark_on_local(cores=1, conf={"spark.driver.memory": "20g"})
    num_executors = 3
    num_cores_per_executor = 4
    hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
    init_spark_on_yarn(
        hadoop_conf=hadoop_conf_dir,
        conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
        num_executor=num_executors,
        executor_cores=num_cores_per_executor,
        executor_memory="2g",
        driver_memory="10g",
        driver_cores=1,
        spark_conf={"spark.rpc.message.maxSize": "1024",
                    "spark.task.maxFailures":  "1",
                    "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    model = Net()
    criterion = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model.train()
    adam = Adam(1e-3)
    zooModel = TorchModel.from_pytorch(model)
    zooCriterion = TorchLoss.from_pytorch(criterion)
    # from bigdl.models.lenet.lenet5 import build_model
    # zooModel = build_model(10)
    # def lossFunc(input, target):
    #     return nn.NLLLoss().forward(input, target.flatten().long())
    #
    # zooCriterion = TorchCriterion.from_pytorch(lossFunc, [1, 2], torch.LongTensor([1]))
    # zooCriterion = SparseCategoricalCrossEntropy(zero_based_label=True)
    # from bigdl.nn.criterion import ClassNLLCriterion
    # zooCriterion = ClassNLLCriterion()
    estimator = Estimator(zooModel, optim_methods=adam)
    # train_featureSet = FeatureSet.data_loader(train_loader)
    # test_featureSet = FeatureSet.data_loader(test_loader)
    train_featureSet = FeatureSet.pytorch_dataloader(train_loader)
    test_featureSet = FeatureSet.pytorch_dataloader(test_loader)
    # c = train_featureSet.to_dataset().size()
    # print(c)
    # print(test_featureSet.to_dataset().size())
    # estimator.evaluate_minibatch(train_featureSet, [Accuracy()])
    from bigdl.optim.optimizer import MaxEpoch, EveryEpoch, MaxIteration
    estimator.train_minibatch(train_featureSet, zooCriterion, end_trigger=MaxEpoch(3), checkpoint_trigger=EveryEpoch(),
                              validation_set=test_featureSet, validation_method=[Accuracy()])

if __name__ == '__main__':
    main()
