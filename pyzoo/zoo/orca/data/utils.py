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

from zoo.common import get_file_list
import numpy as np


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

    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    # file
    if os.path.splitext(file_path)[1] != '':
        return ["s3://" + file_path]
    else:
        keys = []
        resp = s3_client.list_objects_v2(Bucket=bucket,
                                         Prefix=key)
        for obj in resp['Contents']:
            keys.append(obj['Key'])
        # only get json/csv files
        files = [file for file in keys if os.path.splitext(file)[1] == "." + file_type]
        file_paths = [os.path.join("s3://" + bucket, file) for file in files]
        return file_paths


def extract_one_path(file_path, file_type, env):
    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]
    if prefix == "s3":
        file_paths = list_s3_file(file_url_splits[1], file_type, env)
    elif prefix == "hdfs":
        import pyarrow as pa
        fs = pa.hdfs.connect()
        if fs.isfile(file_path):
            return [file_path]
        else:
            file_paths = get_file_list(file_path)
            # only get json/csv files
            file_paths = [file for file in file_paths
                          if os.path.splitext(file)[1] == "." + file_type]
    else:
        if os.path.isfile(file_path):
            return [file_path]
        else:
            file_paths = get_file_list(file_path)
            # only get json/csv files
            file_paths = [file for file in file_paths
                          if os.path.splitext(file)[1] == "." + file_type]
    return file_paths


def open_text(path):
    # Return a list of lines
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            lines = f.read().decode("utf-8").split("\n")
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        lines = data["Body"].read().decode("utf-8").split("\n")
    else:  # Local path
        lines = []
        for line in open(path):
            lines.append(line)
    return [line.strip() for line in lines]


def open_image(path):
    from PIL import Image
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        from io import BytesIO
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            return Image.open(BytesIO(f.read()))
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        from io import BytesIO
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(data["Body"].read()))
    else:  # Local path
        return Image.open(path)


def check_data_type_and_to_list(data):
    result = {}
    assert isinstance(data, dict), "each shard should be an dict"
    assert "x" in data, "key x should in each shard"
    x = data["x"]
    if isinstance(x, np.ndarray):
        new_x = [x]
    elif isinstance(x, list) and all([isinstance(xi, np.ndarray) for xi in x]):
        new_x = x
    else:
        raise ValueError("value of x should be a ndarray or a list of ndarrays")
    result["x"] = new_x
    if "y" in data:
        y = data["y"]
        if isinstance(y, np.ndarray):
            new_y = [y]
        elif isinstance(y, list) and all([isinstance(yi, np.ndarray) for yi in y]):
            new_y = y
        else:
            raise ValueError("value of x should be a ndarray or a list of ndarrays")
        result["y"] = new_y
    return result


def get_spec(data):
    data = check_data_type_and_to_list(data)
    feature_spec = [(feat.dtype, feat.shape[1:])
                    for feat in data["x"]]
    if "y" in data:
        label_spec = [(label.dtype, label.shape[1:])
                      for label in data["y"]]
    else:
        label_spec = None
    return feature_spec, label_spec


# todo this might be very slow
def flatten_xy(data):
    data = check_data_type_and_to_list(data)
    features = data["x"]

    has_label = "y" in data
    labels = data["y"] if has_label else None
    length = features[0].shape[0]

    for i in range(length):
        fs = [feat[i] for feat in features]
        if has_label:
            ls = [l[i] for l in labels]
            yield (fs, ls)
        else:
            yield (fs,)


# todo this might be very slow
def to_sample(data):
    from bigdl.util.common import Sample
    data = check_data_type_and_to_list(data)
    features = data["x"]
    labels = data["y"]
    length = features[0].shape[0]

    for i in range(length):
        fs = [feat[i] for feat in features]
        ls = [l[i] for l in labels]
        yield Sample.from_ndarray(np.array(fs), np.array(ls))
