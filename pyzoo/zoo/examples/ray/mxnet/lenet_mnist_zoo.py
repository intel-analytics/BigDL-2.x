import argparse
import ray
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.test_utils import get_mnist_iterator
from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray.util.raycontext import RayContext
from zoo.ray.mxnet import MXNetTrainer


# Reference: https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html


def get_data_iters(config, kv):
    return get_mnist_iterator(config["batch_size"], (1, 28, 28),
                              num_parts=kv.num_workers, part_index=kv.rank)


def get_model(config):
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
        "kvstore": args.kvstore,
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
    # CLI
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('-n', '--num-workers', required=True, type=int,
                        help='number of worker nodes to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help='number of server nodes to be launched, \
                        in default it is equal to NUM_WORKERS')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--epochs', type=int, default=4,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate. default is 0.02.')
    parser.add_argument('--kvstore', type=str, default='dist_sync',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Number of batches to wait before logging.')
    opt = parser.parse_args()

    # ray.init()

    # sc = init_spark_on_local(cores=8)
    sc = init_spark_on_yarn(
        hadoop_conf="/opt/work/hadoop-2.7.2/etc/hadoop",
        conda_name="mxnet",
        num_executor=opt.num_workers,
        executor_cores=16,
        executor_memory="10g",
        driver_memory="2g",
        driver_cores=4,
        extra_executor_memory_for_ray="30g")
    ray_ctx = RayContext(sc=sc,
                         object_store_memory="2g",
                         env={"http_proxy": "http://child-prc.intel.com:913",
                              "https_proxy": "http://child-prc.intel.com:913"})
    ray_ctx.init(object_store_memory="2g")

    config = create_config(opt)
    trainer = MXNetTrainer(get_data_iters, get_model, get_loss, get_metrics, config)
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
