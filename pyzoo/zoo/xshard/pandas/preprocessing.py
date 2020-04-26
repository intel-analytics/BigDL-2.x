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

import ray
from pyspark.context import SparkContext

from zoo.ray import RayContext
from zoo.xshard.shard import RayDataShards, RayPartition
from zoo.xshard.utils import *


def read_csv(file_path, context):
    """
    Read csv files to DataShards
    :param file_path: could be a csv file, multiple csv file paths separated by comma,
     a directory containing csv files.
     Supported file systems are local file system, hdfs, and s3.
    :param context: SparkContext or RayContext
    :return: DataShards
    """
    if isinstance(context, RayContext):
        return read_file_ray(context, file_path, "csv")
    elif isinstance(context, SparkContext):
        pass
    else:
        raise Exception("Context type should be RayContext or SparkContext")


def read_json(file_path, context):
    """
    Read json files to DataShards
    :param file_path: could be a json file, multiple json file paths separated by comma,
     a directory containing json files.
     Supported file systems are local file system, hdfs, and s3.
    :param context: SparkContext or RayContext
    :return: DataShards
    """
    if isinstance(context, RayContext):
        return read_file_ray(context, file_path, "json")
    elif isinstance(context, SparkContext):
        pass
    else:
        raise Exception("Context type should be RayContext or SparkContext")


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
                import pyarrow as pa
                fs = pa.hdfs.connect()
                files = fs.ls(file_path)
                # only get json/csv files
                files = [file for file in files if os.path.splitext(file)[1] == "." + file_type]
                file_paths = ["hdfs://" + server_address + file for file in files]
            elif prefix == "s3":
                import boto3
                path_parts = file_url_splits[1].split('/')
                bucket = path_parts.pop(0)
                key = "/".join(path_parts)
                env = context.ray_service.env
                access_key_id = env["AWS_ACCESS_KEY_ID"]
                secret_access_key = env["AWS_SECRET_ACCESS_KEY"]
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
            else:
                # only get json/csv files
                file_paths = [os.path.join(file_path, file)
                              for file in os.listdir(file_path)
                              if os.path.splitext(file)[1] == "." + file_type]
    else:
        file_paths = file_path_splits

    num_executors = context.num_ray_nodes
    num_cores = context.ray_node_cpu_cores
    num_partitions = num_executors * num_cores
    # remove empty partitions
    file_partition_list = [partition for partition
                           in list(chunk(file_paths, num_partitions)) if partition]
    # create shard actor to read data
    shards = [RayPandasShard.remote() for i in range(len(file_partition_list))]
    done_ids, undone_ids = \
        ray.wait([shard.read_file_partitions.remote(file_partition_list[i], file_type)
                  for i, shard in enumerate(shards)], num_returns=len(shards))
    assert len(undone_ids) == 0

    # create initial partition
    partitions = [RayPartition([shard]) for shard in shards]
    data_shards = RayDataShards(partitions)
    return data_shards


@ray.remote
class RayPandasShard(object):
    """
    Actor to read csv/json file to Pandas DataFrame and manipulate data
    """

    def __init__(self, data=None):
        self.data = data

    def read_file_partitions(self, paths, file_type):
        df_list = []
        import pandas as pd
        prefix = paths[0].split("://")[0]
        if prefix == "hdfs":
            import pyarrow as pa
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
            import boto3
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
        return 0

    def apply(self, func, *args):
        self.data = func(self.data, *args)
        return 0

    def get_data(self):
        return self.data
