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

import argparse

import numpy as np
import torch
import torch.nn as nn

from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.orca.data import SparkXShards
from zoo.ray import RayContext
from zoo.orca.learn.pytorch.pytorch_horovod_estimator import PyTorchHorovodEstimator
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def model_creator(config):
    """Returns a torch.nn.Module object."""

    return Net()


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))


def scheduler_creator(optimizer, config):
    """Returns a learning rate scheduler wrapping the optimizer.
    You will need to set ``TorchTrainer(scheduler_step_freq="epoch")``
    for the scheduler to be incremented correctly.
    If using a scheduler for validation loss, be sure to call
    ``trainer.update_scheduler(validation_loss)``.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)


def train_example(data_shards, validation_data_shards):

    trainer1 = PyTorchHorovodEstimator(
        model_creator=model_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=nn.NLLLoss,
        scheduler_creator=None,
        config={
            "lr": 1e-3,  # used in optimizer_creator
            "batch_size": 256,  # used in data_creator
        })

    last_val_stats = None
    # train 5 epochs
    for i in range(5):
        stats = trainer1.train(data_shards)
        print("train stats: {}".format(stats))
        val_stats = trainer1.validate(validation_data_shards)
        print("validata stats: {}".format(val_stats))
        last_val_stats = val_stats
    return last_val_stats


parser = argparse.ArgumentParser()
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                         " configuration folder. Otherwise, turn on local mode.")
parser.add_argument("--slave_num", type=int, default=2,
                    help="The number of slave nodes")
parser.add_argument("--conda_name", type=str,
                    help="The name of conda environment.")
parser.add_argument("--executor_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--executor_memory", type=str, default="10g",
                    help="The size of slave(executor)'s memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_memory", type=str, default="2g",
                    help="The size of driver's memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--extra_executor_memory_for_ray", type=str, default="20g",
                    help="The extra executor memory to store some data."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--object_store_memory", type=str, default="4g",
                    help="The memory to store data on local."
                         "You can change it depending on your own cluster setting.")

if __name__ == "__main__":

    args = parser.parse_args()
    if args.hadoop_conf:
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name=args.conda_name,
            num_executor=args.slave_num,
            executor_cores=args.executor_cores,
            executor_memory=args.executor_memory,
            driver_memory=args.driver_memory,
            driver_cores=args.driver_cores,
            extra_executor_memory_for_ray=args.extra_executor_memory_for_ray)
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()
    else:
        sc = init_spark_on_local(conf={"spark.driver.memory": "20g"})
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
        ray_ctx.init()

    from bigdl.dataset import mnist

    (train_images_data, train_labels_data) = mnist.read_data_sets("/tmp/mnist", "train")
    (test_images_data, test_labels_data) = mnist.read_data_sets("/tmp/mnist", "train")

    train_images_data = (train_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    train_labels_data = train_labels_data.astype(np.int)
    test_images_data = (test_images_data - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
    test_labels_data = test_labels_data.astype(np.int)

    import zoo.orca.data

    data_shards = zoo.orca.data.partition({"x": train_images_data.reshape((-1, 1, 28, 28)).astype(np.float32), "y": train_labels_data})
    val_data_shards = zoo.orca.data.partition({"x": test_images_data.reshape((-1, 1, 28, 28)).astype(np.float32), "y": test_labels_data})
    val_stats = train_example(data_shards, val_data_shards)

    assert val_stats["val_accuracy"] > 0.97
