# This file is adapted from https://github.com/ray-project/ray/blob
# /master/examples/parameter_server/async_parameter_server.py
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
import time

import model
import ray

from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray import RayContext

os.environ["LANG"] = "C.UTF-8"
parser = argparse.ArgumentParser(description="Run the asynchronous parameter "
                                             "server example.")
parser.add_argument("--num_workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--iterations", default=50, type=int,
                    help="Iteration time.")
parser.add_argument("--hadoop_conf", type=str,
                    help="turn on yarn mode by passing the path to the hadoop"
                    "Configuration folder. Otherwise, turn on local mode.")
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
    def __init__(self, keys, values):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        for key, value in zip(keys, values):
            self.weights[key] += value

    def pull(self, keys):
        return [self.weights[key] for key in keys]


@ray.remote
def worker_task(ps, worker_index, batch_size=50):
    # Download MNIST.
    print("Worker " + str(worker_index))
    mnist = model.download_mnist_retry(seed=worker_index)

    # Initialize the model.
    net = model.SimpleCNN()
    keys = net.get_weights()[0]

    while True:
        # Get the current weights from the parameter server.
        weights = ray.get(ps.pull.remote(keys))
        net.set_weights(keys, weights)
        # Compute an update and push it to the parameter server.
        xs, ys = mnist.train.next_batch(batch_size)
        gradients = net.compute_update(xs, ys)
        ps.push.remote(keys, gradients)

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

    # Create a parameter server with some random weights.
    net = model.SimpleCNN()
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values)

    # Start some training tasks.
    worker_tasks = [worker_task.remote(ps, i) for i in range(args.num_workers)]

    # Download MNIST.
    mnist = model.download_mnist_retry()
    print("Begin iteration")
    i = 0
    while i < args.iterations:
        # Get and evaluate the current model.
        print("-----Iteration" + str(i) + "------")
        current_weights = ray.get(ps.pull.remote(all_keys))
        net.set_weights(all_keys, current_weights)
        test_xs, test_ys = mnist.test.next_batch(1000)
        accuracy = net.compute_accuracy(test_xs, test_ys)
        print("Iteration {}: accuracy is {}".format(i, accuracy))
        i += 1
        time.sleep(1)
    ray_ctx.stop()
    sc.stop()
