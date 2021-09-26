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


# split list into n chunks
def chunk(lst, n):
    size = len(lst) // n
    leftovers = lst[size * n:]
    for c in range(n):
        if leftovers:
            extra = [leftovers.pop()]
        else:
            extra = []
        yield lst[c * size:(c + 1) * size] + extra


def flatten(list_of_list):
    flattend = [item for sublist in list_of_list for item in sublist]
    return flattend


def list_s3_file(file_path, file_type, env):
    path_parts = file_path.split('/')
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    access_key_id = env["AWS_ACCESS_KEY_ID"]
    secret_access_key = env["AWS_SECRET_ACCESS_KEY"]

    # access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    # secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    keys = []
    resp = s3_client.list_objects_v2(Bucket=bucket,
                                     Prefix=key)
    for obj in resp['Contents']:
        keys.append(obj['Key'])
    # only get json/csv files
    files = [file for file in keys if os.path.splitext(file)[1] == "." + file_type]
    file_paths = [os.path.join("s3://" + bucket, file) for file in files]
    return file_paths


def extract_one_path(file_path, file_type, context):
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
            env = context.env
            file_paths = list_s3_file(file_url_splits[1], file_type, env)
        else:
            # local file system
            # only get json/csv files
            file_paths = [os.path.join(file_path, file)
                          for file in os.listdir(file_path)
                          if os.path.splitext(file)[1] == "." + file_type]
    return file_paths
