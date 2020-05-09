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
from bigdl.optim.optimizer import SGD, Adam
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
    hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
    num_executors = 30
    num_cores_per_executor = args.cores
    os.environ['ZOO_MKL_NUMTHREADS'] = str(num_cores_per_executor)
    os.environ['OMP_NUM_THREADS'] = str(num_cores_per_executor)
    #sc = init_spark_on_local(cores=args.cores, conf={"spark.driver.memory": "20g"})
    sc = init_spark_on_yarn(
        hadoop_conf=hadoop_conf_dir,
        conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
        num_executor=num_executors,
        executor_cores=num_cores_per_executor,
        executor_memory="20g",
        driver_memory="20g",
        driver_cores=1,
        spark_conf={"spark.rpc.message.maxSize": "1024",
                    "spark.task.maxFailures":  "1",
                    "spark.scheduler.minRegisteredResourcesRatio": "1",
                    "spark.scheduler.maxRegisteredResourcesWaitingTime": "100s",
                    "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    model = torchvision.models.resnet50()
    model.train()
    #adam = Adam()
    #zooModel = TorchNet2.from_pytorch(model)
    #Estimator.test3(model)
    #Estimator.test4(model)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

   #criterion = nn.CrossEntropyLoss()
   #import time
   #for i, (images, target) in enumerate(train_loader):
   #    s = time.time()
   #    output = model(images)
   #    loss = criterion(output, target)
   #    print(str(i) + ": " + str(loss.data.item()) + " " + str(time.time() - s))

   # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

   #num_executors = 1
   #num_cores_per_executor = 4
   #hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
   #sc = init_nncontext(conf={"spark.driver.memory": "10g"})
   #from pyspark.conf import SparkConf
   # spark_conf = SparkConf().setMaster("local[16]").setAppName("resnet") \
   #              .set("spark.driver.memory", "10g") \
   #              .set("spark.rpc.message.maxSize", "1024") \
   #              .setExecutorEnv("OMP_NUM_THREADS", "16")
   # spark_conf = {"spark.driver.memory": "10g", "spark.rpc.message.maxSize": "1024"}
   #sc = init_nncontext()
   #sc = init_spark_on_local(cores=16, conf={"spark.driver.memory": "10g"})
   #sc = init_spark_on_yarn(
   #    hadoop_conf=hadoop_conf_dir,
   #    conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
   #    num_executor=num_executors,
   #    executor_cores=num_cores_per_executor,
   #    executor_memory="10g",
   #    driver_memory="10g",
   #    driver_cores=1,
   #    spark_conf={"spark.rpc.message.maxSize": "1024",
   #                "spark.task.maxFailures":  "1",
   #                "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})
    model.train()
    adam = Adam()
    zooModel = TorchModel.from_pytorch(model)
    # from bigdl.models.lenet.lenet5 import build_model
    # zooModel = build_model(10)
    criterion = nn.CrossEntropyLoss()
    def lossFunc(input, target):
        return criterion.forward(input, target.flatten().long())

    # zooCriterion = TorchCriterion.from_pytorch(lossFunc, [1, 1000], torch.LongTensor([1]))
    zooCriterion = TorchLoss.from_pytorch(criterion)
    # zooCriterion = SparseCategoricalCrossEntropy(zero_based_label=True)
    # from bigdl.nn.criterion import ClassNLLCriterion
    # zooCriterion = ClassNLLCriterion()
    estimator = Estimator(zooModel, optim_methods=adam)
    train_featureSet = FeatureSet.data_loader(train_loader)
    test_featureSet = FeatureSet.data_loader(val_loader)
    c = train_featureSet.to_dataset().size()
    print(c)
    # estimator.evaluate_minibatch(train_featureSet, [Accuracy()])
    from bigdl.optim.optimizer import MaxEpoch, EveryEpoch, MaxIteration
    from zoo.pipeline.api.keras.metrics import Accuracy
    estimator.train_minibatch(train_featureSet, zooCriterion, end_trigger=MaxEpoch(1), checkpoint_trigger=EveryEpoch(),
                              validation_set=test_featureSet, validation_method=[Accuracy()])
    #Estimator.print()

if __name__ == '__main__':
    main()
