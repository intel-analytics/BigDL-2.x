# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import division

import argparse

import ray
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from mxnet.test_utils import get_cifar10
from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext
from zoo.ray.mxnet import MXNetTrainer


class DummyIter(mx.io.DataIter):
    def __init__(self, batch_size, data_shape, batches=50):
        super(DummyIter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,)
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = [('softmax_label', self.label_shape)]
        self.batch = mx.io.DataBatch(data=[mx.nd.zeros(self.data_shape)],
                                     label=[mx.nd.zeros(self.label_shape)])
        self._batches = 0
        self.batches = batches

    def next(self):
        if self._batches < self.batches:
            self._batches += 1
            return self.batch
        else:
            self._batches = 0
            raise StopIteration


def dummy_iterator(batch_size, data_shape):
    return DummyIter(batch_size, data_shape), DummyIter(batch_size, data_shape)


def get_cifar10_iterator(batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    get_cifar10()

    train = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        # mean_img="data/cifar/mean.bin",
        resize=resize,
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=True,
        rand_mirror=True,
        num_parts=num_parts,
        part_index=part_index)

    val = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        # mean_img="data/cifar/mean.bin",
        resize=resize,
        rand_crop=False,
        rand_mirror=False,
        data_shape=data_shape,
        batch_size=batch_size,
        num_parts=num_parts,
        part_index=part_index)

    return train, val


def get_data_iters(config, kv):
    """get dataset iterators"""
    if config["dataset"] == "cifar10":
        return get_cifar10_iterator(config["batch_size"], (3, 32, 32),
                                    num_parts=kv.num_workers, part_index=kv.rank)
    elif config["dataset"] == "dummy":
        shape_dim = 299 if config["model"] == 'inceptionv3' else 224
        return dummy_iterator(config["batch_size"], (3, shape_dim, shape_dim))
    else:
        raise ValueError("Unsupported dataset")


def get_model(config):
    """Model initialization."""
    model = config["model"]
    context = [mx.cpu()]
    kwargs = {'ctx': context, 'pretrained': config["use_pretrained"], 'classes': 10}
    if model.startswith('resnet'):
        kwargs['thumbnail'] = config["use_thumbnail"]
    elif model.startswith('vgg'):
        kwargs['batch_norm'] = config["batch_norm"]

    net = models.get_model(model, **kwargs)
    if not config["use_pretrained"]:
        if model in ['alexnet']:
            net.initialize(mx.init.Normal())
        else:
            net.initialize(mx.init.Xavier(magnitude=2))
    net.cast("float32")
    net.collect_params().reset_ctx(context)
    return net


def get_loss(config):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    return loss


def get_metrics(config):
    return CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])


def create_config(args):
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "num_workers": args.num_workers,
        "kvstore": args.kvstore,
        "batch_size": args.batch_size,
        "use_thumbnail": args.use_thumbnail,
        "batch_norm": args.batch_norm,
        "use_pretrained": args.use_pretrained,
        "optimizer": 'sgd',
        "optimizer_params": {'learning_rate': args.lr,  # TODO: add learning rate decay
                             'wd': args.wd,
                             'momentum': args.momentum,
                             'multi_precision': True},
        "seed": 123
    }
    if args.num_servers:
        config["num_servers"] = args.num_servers
    if args.log_interval:
        config["log_interval"] = args.log_interval
    return config


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('-n', '--num-workers', required=True, type=int,
                        help='number of worker nodes to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help='number of server nodes to be launched, \
                        in default it is equal to NUM_WORKERS')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use. options are cifar10 and dummy.')
    parser.add_argument('--use_thumbnail', action='store_true',
                        help='use thumbnail or not in resnet. default is false.')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per worker.')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--kvstore', type=str, default='dist_sync',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    opt = parser.parse_args()

    # ray.init()

    # sc = init_spark_on_local(cores="*")
    num_executors = 2 * opt.num_workers if not opt.num_servers else opt.num_workers + opt.num_servers
    sc = init_spark_on_yarn(
        hadoop_conf="/opt/work/hadoop-2.7.2/etc/hadoop",
        conda_name="mxnet",
        # 1 executor for ray head node. The remaining executors for raylets.
        # Each executor is given enough cores to be placed on one node.
        # Each MXNetRunner will run in one executor, namely one node.
        num_executor=num_executors,
        executor_cores=28,
        executor_memory="10g",
        driver_memory="2g",
        driver_cores=16,
        extra_executor_memory_for_ray="5g")
    ray_ctx = RayContext(sc=sc,
                         object_store_memory="10g",
                         env={"http_proxy": "10.239.4.101:913",
                              "https_proxy": "10.239.4.101:913",
                              "OMP_NUM_THREADS": "28",
                              "KMP_AFFINITY": "granularity=fine,compact,1,0",
                              "MXNET_SUBGRAPH_BACKEND": "MKLDNN"})
    ray_ctx.init(object_store_memory="10g")

    config = create_config(opt)
    trainer = MXNetTrainer(config, get_data_iters, get_model, get_loss, get_metrics, worker_cpus=28)
    for epoch in range(opt.epochs):
        train_stats = trainer.train()
        val_stats = trainer.validate()
        for stat in train_stats:
            if len(stat.keys()) > 1:  # Worker
                print(stat)
        for stat in val_stats:
            if len(stat.keys()) > 1:  # Worker
                print(stat)
    ray_ctx.stop()
    sc.stop()
    # ray.shutdown()
