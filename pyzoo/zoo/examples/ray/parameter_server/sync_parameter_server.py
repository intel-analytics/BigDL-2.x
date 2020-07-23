# This file is adapted from https://github.com/ray-project/ray/blob
# /master/examples/parameter_server/sync_parameter_server.py
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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import model
import numpy as np
import ray

from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray import RayContext

os.environ["LANG"] = "C.UTF-8"
parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num_workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--iterations", default=50, type=int,
                    help="Iteration time.")
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                    "configuration folder. Otherwise, turn on local mode.")
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


@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        self.net = model.SimpleCNN(learning_rate=learning_rate)

    def apply_gradients(self, *gradients):
        self.net.apply_gradients(np.mean(gradients, axis=0))
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()


@ray.remote
class Worker(object):
    def __init__(self, worker_index, batch_size=50):
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.mnist = model.download_mnist_retry(seed=worker_index)
        self.net = model.SimpleCNN()

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.mnist.train.next_batch(self.batch_size)
        return self.net.compute_gradients(xs, ys)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.hadoop_conf:
        sc = init_spark_on_yarn(
            hadoop_conf=args.hadoop_conf,
            conda_name=args.conda_name,
            num_executors=args.num_workers,
            executor_cores=args.executor_cores,
            executor_memory=args.executor_memory,
            driver_memory=args.driver_memory,
            driver_cores=args.driver_cores,
            extra_executor_memory_for_ray=args.extra_executor_memory_for_ray)
        ray_ctx = RayContext(
            sc=sc,
            object_store_memory=args.object_store_memory)
    else:
        sc = init_spark_on_local(cores=args.driver_cores)
        ray_ctx = RayContext(sc=sc, object_store_memory=args.object_store_memory)
    ray_ctx.init()

    # Create a parameter server.
    net = model.SimpleCNN()
    ps = ParameterServer.remote(1e-4 * args.num_workers)

    # Create workers.
    workers = [Worker.remote(worker_index)
               for worker_index in range(args.num_workers)]

    # Download MNIST.
    mnist = model.download_mnist_retry()

    i = 0
    current_weights = ps.get_weights.remote()
    print("Begin iteration")
    while i < args.iterations:
        # Compute and apply gradients.
        gradients = [worker.compute_gradients.remote(current_weights)
                     for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)

        if i % 10 == 0:
            # Evaluate the current model.
            net.variables.set_flat(ray.get(current_weights))
            test_xs, test_ys = mnist.test.next_batch(1000)
            accuracy = net.compute_accuracy(test_xs, test_ys)
            print("Iteration {}: accuracy is {}".format(i, accuracy))
        i += 1
    ray_ctx.stop()
    sc.stop()
