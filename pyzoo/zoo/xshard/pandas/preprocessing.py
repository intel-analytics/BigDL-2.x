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
import os

import boto3
import ray
import pandas as pd
import pyarrow as pa
from bigdl.util.common import get_node_and_core_number
from pyspark.context import SparkContext

from zoo.common import get_file_list
from zoo.ray.util.raycontext import RayContext
from zoo.xshard.shard import RayDataShards, ScDataShards
from zoo.xshard.utils import *


def list_s3_file(file_path, file_type):
    path_parts = file_path[1].split('/')
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    keys = []
    resp = s3_client.list_objects_v2(Bucket=bucket,
                                     Prefix=key)
    for obj in resp['Contents']:
        keys.append(obj['Key'])
    files = list(dict.fromkeys(keys))
    # only get json/csv files
    files = [file for file in files if os.path.splitext(file)[1] == "." + file_type]
    file_paths = [os.path.join("s3://" + bucket, file) for file in files]
    return file_paths


def read_csv(file_path, context):
    if isinstance(context, RayContext):
        return read_csv_ray(context, file_path)
    elif isinstance(context, SparkContext):
        return read_csv_sc(context, file_path)
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
    return read_file_ray(context, file_path, "csv")


def read_json_ray(context, file_path):
    return read_file_ray(context, file_path, "json")


def read_csv_sc(context, file_path):
    return read_file_sc(context, file_path, "csv")


def read_json_sc(context, file_path):
    return read_file_sc(context, file_path, "json")


def read_file_ray(context, file_path, file_type):
    file_path_splits = file_path.split(',')
    if len(file_path_splits) == 1:
        # only one file
        if os.path.splitext(file_path)[-1] == "." + file_type:
            file_paths = [file_path]
        # directory
        else:
            file_url_splits = file_path.split("://")
            prefix = file_url_splits[0]
            if prefix == "hdfs":
                server_address = file_url_splits[1].split('/')[0]
                fs = pa.hdfs.connect()
                files = fs.ls(file_path)
                # only get json/csv files
                files = [file for file in files if os.path.splitext(file)[1] == "." + file_type]
                file_paths = ["hdfs://" + server_address + file for file in files]
            elif prefix == "s3":
                list_s3_file(file_url_splits, file_type)
            else:
                file_paths = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    else:
        file_paths = file_path_splits

    num_executors = context.num_ray_nodes
    num_cores = context.ray_node_cpu_cores
    num_partitions = num_executors * num_cores
    # remove empty partitions
    file_partition_list = [partition for partition
                           in list(chunk(file_paths, num_partitions)) if partition]
    shards = [RayPandasShard.remote() for i in range(len(file_partition_list))]
    [shard.read_file_partitions.remote(file_partition_list[i], file_type)
     for i, shard in enumerate(shards)]
    # shard_partitions = [[shard] for shard in shards]
    data_shards = RayDataShards(shards)
    return data_shards


def read_file_sc(context, file_path, file_type):
    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]
    node_num, core_num = get_node_and_core_number()

    data_paths = list_s3_file(file_url_splits, file_type) if prefix == "s3" else get_file_list(file_path)
    rdd = context.parallelize(data_paths, node_num * 20)

    if prefix == "hdfs":
        def loadFile(iterator):
            import pandas as pd
            import pyarrow as pa
            fs = pa.hdfs.connect()

            for x in iterator:
                with fs.open(x, 'rb') as f:
                    if file_type == "csv":
                        df = pd.read_csv(f, header=0)
                    elif file_type == "json":
                        df = pd.read_json(f, orient='columns', lines=True)
                    else:
                        raise Exception("Unsupported file type")
                    yield df

        pd_rdd = rdd.mapPartitions(loadFile)
    elif prefix == "s3":
        def loadFile(iterator):
            access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
            secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            s3_client = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            ).client('s3', verify=False)
            for x in iterator:
                path_parts = x.split("://")[1].split('/')
                bucket = path_parts.pop(0)
                key = "/".join(path_parts)
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                if file_type == "json":
                    df = pd.read_json(obj['Body'], orient='columns', lines=True)
                elif file_type == "csv":
                    df = pd.read_csv(obj['Body'])
                else:
                    raise Exception("Unsupported file type")
                yield df

        pd_rdd = rdd.mapPartitions(loadFile)
    else:
        def loadFile(iterator):
            import pandas as pd
            for x in iterator:
                if file_type == "csv":
                    df = pd.read_csv(x, header=0)
                elif file_type == "json":
                    df = pd.read_json(orient='columns', lines=True)
                else:
                    raise Exception("Unsupported file type")
                yield df

        pd_rdd = rdd.mapPartitions(loadFile)

    data_shards = ScDataShards(pd_rdd)
    return data_shards


# @ray.remote
class RayPandasShard(object):
    def __init__(self, data=None):
        self.data = data

    def read_file_partitions(self, paths, file_type):
        df_list = []
        prefix = paths[0].split("://")[0]
        import time
        start = time.time()
        if prefix == "hdfs":
            fs = pa.hdfs.connect()
            print("Start loading files")
            for path in paths:
                with fs.open(path, 'rb') as f:
                    if file_type == "json":
                        df = pd.read_json(f, orient='columns', lines=True)
                    elif file_type == "csv":
                        df = pd.read_csv(f)
                    else:
                        raise Exception("Unsupported file type")
                    df_list.append(df)
        elif prefix == "s3":
            access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
            secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            s3_client = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            ).client('s3', verify=False)
            for path in paths:
                path_parts = path.split("://")[1].split('/')
                bucket = path_parts.pop(0)
                key = "/".join(path_parts)
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                if file_type == "json":
                    df = pd.read_json(obj['Body'], orient='columns', lines=True)
                elif file_type == "csv":
                    df = pd.read_csv(obj['Body'])
                else:
                    raise Exception("Unsupported file type")
                df_list.append(df)
        else:
            for path in paths:
                if file_type == "json":
                    df = pd.read_json(path, orient='columns', lines=True)
                elif file_type == "csv":
                    df = pd.read_csv(path)
                else:
                    raise Exception("Unsupported file type")
                df_list.append(df)
        self.data = pd.concat(df_list)
        end = time.time()
        print("read shard time: ", end - start)
        return self.data

    def apply(self, func, *args):
        self.data = func(self.data, *args)

    def get_data(self):
        return self.data

