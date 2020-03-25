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

import ray
import pyarrow as pa
import pandas as pd
import random
import os
from pyspark.context import SparkContext
from zoo.ray.util.raycontext import RayContext
from zoo.xshard.shard import RayDataShard


def read_csv(file_path, context):
    if isinstance(context, RayContext):
        return read_csv_ray(context, file_path)
    elif isinstance(context, SparkContext):
        pass
    else:
        raise Exception("Context type should be RayContext or SparkContext")


def read_json(file_path, context):
    if isinstance(context, RayContext):
        return read_json_ray(context, file_path)
    elif isinstance(context, SparkContext):
        pass
    else:
        raise Exception("Context type should be RayContext or SparkContext")


def read_csv_ray(context, file_path):
    read_file_ray(context, file_path, "csv")


def read_json_ray(context, file_path):
    read_file_ray(context, file_path, "json")


def read_file_ray(context, file_path, file_type):
    if len(file_path) == 1:
        # only one file
        if os.path.splitext(file_path)[-1] == file_type:
            file_paths = file_path
        # directory
        else:
            file_paths = [os.path.abspath(x) for x in os.listdir(file_path)]
    else:
        file_paths = file_path
    # shuffle
    shuffled_indexes = random.shuffle(range(len(file_paths)))
    num_executors = context.num_ray_nodes
    num_cores = context.ray_node_cpu_cores
    num_partitions = num_executors * num_cores

    def chunk(ys, n):
        random.shuffle(ys)
        size = len(ys) // n
        leftovers = ys[size * n:]
        for c in range(n):
            if leftovers:
                extra = [leftovers.pop()]
            else:
                extra = []
            yield ys[c * size:(c + 1) * size] + extra

    partition_list = list(chunk(file_paths, num_partitions))
    shards = [RayPandasShard(num_cpus=1).remote() for i in range(num_partitions)]
    data_shard = RayDataShard(shards)
    [shards[i].read_file_partitions(partition_list[i], file_type) for i in len(data_shard.shards)]
    return data_shard

@ray.remote
class RayPandasShard(object):
    def __init__(self, data=None):
        self.data = data

    def read_file_partitions(self, paths, file_type):
        df_list = []
        prefix = paths[0].split("://")[0]
        if prefix == "hdfs":
            fs = pa.hdfs.connect()
            print("Start loading files")
            for path in paths:
                # For parquet files
                # df = fs.read_parquet(path).to_pandas()
                with fs.open(path, 'rb') as f:
                    if file_type == "json":
                        df = pd.read_json(f, orient='columns', lines=True)
                    elif file_type == "csv":
                        df = pd.read_csv(f)
                    else:
                        raise Exception("Unsupported file type")
                    df_list.append(df)
        self.data = pd.concat(df_list)
        return self.data

    def apply(self, func):
        return func(self.data)

    def get_data(self):
        return self.data


