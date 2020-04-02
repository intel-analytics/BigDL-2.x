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

# Reference: https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html

import argparse
import mxnet as mx
from mxnet import gluon
from mxnet.test_utils import get_mnist_iterator
from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext
from zoo.ray.mxnet import MXNetTrainer


def get_data_iters(config, kv):
    return get_mnist_iterator(config["batch_size"], (1, 28, 28),
                              num_parts=kv.num_workers, part_index=kv.rank)


def get_model(config):
    from mxnet.gluon import nn
    import mxnet.ndarray as F

    class LeNet(gluon.Block):
        def __init__(self, **kwargs):
            super(LeNet, self).__init__(**kwargs)
            with self.name_scope():
                # layers created in name_scope will inherit name space
                # from parent layer.
                self.conv1 = nn.Conv2D(20, kernel_size=(5, 5))
                self.pool1 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
                self.conv2 = nn.Conv2D(50, kernel_size=(5, 5))
                self.pool2 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
                self.fc1 = nn.Dense(500)
                self.fc2 = nn.Dense(10)

        def forward(self, x):
            x = self.pool1(F.tanh(self.conv1(x)))
            x = self.pool2(F.tanh(self.conv2(x)))
            # 0 means copy over size from corresponding dimension.
            # -1 means infer size from the rest of dimensions.
            x = x.reshape((0, -1))
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            return x

    net = LeNet()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=[mx.cpu()])
    return net


def get_loss(config):
    return gluon.loss.SoftmaxCrossEntropyLoss()


def get_metrics(config):
    return mx.metric.Accuracy()


def create_config(args):
    config = {
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "optimizer": "sgd",
        "optimizer_params": {'learning_rate': args.lr},
        "seed": 42
    }
    if args.num_servers:
        config["num_servers"] = args.num_servers
    if args.log_interval:
        config["log_interval"] = args.log_interval
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a LeNet model for handwritten digit recognition.')
    parser.add_argument("--hadoop_conf", type=str,
                        help="turn on yarn mode by passing the path to the hadoop"
                             "Configuration folder. Otherwise, turn on local mode.")
    parser.add_argument("--conda_name", type=str,
                        help="The name of conda environment.")
    parser.add_argument("--executor_cores", type=int, default=8,
                        help="The number of driver's cpu cores you want to use."
                             "You can change it depending on your own cluster setting.")
    parser.add_argument('-n', '--num-workers', type=int, default=2,
                        help='number of worker nodes to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help='number of server nodes to be launched, \
                        in default it is equal to NUM_WORKERS')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='training batch size per worker.')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate. default is 0.02.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Number of batches to wait before logging.')
    opt = parser.parse_args()

    if opt.hadoop_conf:
        sc = init_spark_on_yarn(
            hadoop_conf=opt.hadoop_conf,
            conda_name=opt.conda_name,
            num_executor=opt.num_workers,
            executor_cores=opt.executor_cores)
    else:
        sc = init_spark_on_local(cores="*")
    ray_ctx = RayContext(sc=sc)
    ray_ctx.init()

    config = create_config(opt)
    trainer = MXNetTrainer(config, get_data_iters, get_model, get_loss, get_metrics)
    trainer.train(nb_epoch=opt.epochs)
    ray_ctx.stop()
    sc.stop()
