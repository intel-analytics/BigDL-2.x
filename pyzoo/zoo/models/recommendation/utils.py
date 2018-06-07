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

import numpy as np

from bigdl.util.common import JTensor, callBigDlFunc, Sample

from zoo.models.recommendation import UserItemFeature


def hash_bucket(content, bucket_size=1000, start=0):
    return (hash(str(content)) % bucket_size + bucket_size) % bucket_size + start


def categorical_from_vocab_list(sth, vocab_list, default=-1, start=0):
    if sth in vocab_list:
        return vocab_list.index(sth) + start
    else:
        return default + start


def get_boundaries(target, boundaries, default=-1, start=0):
    if target == '?':
        return default + start
    else:
        for i in range(len(boundaries)):
            if target < boundaries[i]:
                return i + start
        return len(boundaries) + start


def get_negative_samples(indexed):
    return callBigDlFunc("float", "getNegativeSamples",
                         indexed)


def get_wide_tensor(row, column_info):
    wide_columns = column_info.wide_base_cols + column_info.wide_cross_cols
    wide_dims = column_info.wide_base_dims + column_info.wide_cross_dims
    wide_length = len(wide_columns)
    acc = 0
    indices = []
    for i in range(0, wide_length):
        index = row[wide_columns[i]]
        if i == 0:
            res = index
        else:
            acc += wide_dims[i-1]
            res = acc + index
        indices.append(res)
    values = np.array([i + 1 for i in indices])
    shape = np.array([sum(wide_dims)])
    return JTensor.sparse(values, np.array(indices), shape)


def get_deep_tensor(row, column_info):
    deep_columns1 = column_info.indicator_cols
    deep_columns2 = column_info.embed_cols + column_info.continuous_cols
    deep_dims1 = column_info.indicator_dims
    deep_length = sum(deep_dims1) + len(deep_columns2)
    deep_tensor = np.zeros((deep_length, ))
    acc = 0
    for i in range(0, len(deep_columns1)):
        index = row[deep_columns1[i]]
        if i == 0:
            res = index
        else:
            acc += deep_dims1[i-1]
            res = acc + index
        deep_tensor[res] = 1
    for i in range(0, len(deep_columns2)):
        deep_tensor[i + sum(deep_dims1)] = float(row[deep_columns2[i]])
    return deep_tensor


def row_to_sample(row, column_info, model_type="wide_n_deep"):
    wide_tensor = get_wide_tensor(row, column_info)
    deep_tensor = JTensor.from_ndarray(get_deep_tensor(row, column_info))
    label = row[column_info.label]
    model_type = model_type.lower()
    if model_type == "wide_n_deep":
        feature = [wide_tensor, deep_tensor]
    elif model_type == "wide":
        feature = wide_tensor
    elif model_type == "deep":
        feature = deep_tensor
    else:
        raise TypeError("Unsupported model_type: %s" % model_type)
    return Sample.from_jtensor(feature, label)


def to_user_item_feature(row, column_info, model_type="wide_n_deep"):
    return UserItemFeature(row["userId"], row["itemId"],
                           row_to_sample(row, column_info, model_type))
