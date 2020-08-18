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

from bigdl.util.common import JTensor, Sample

from zoo.common.utils import callZooFunc
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
    return callZooFunc("float", "getNegativeSamples",
                       indexed)


def get_wide_tensor(row, column_info):
    """
    prepare tensor for wide part of WideAndDeep model based on SparseDense

    :param row: Row of userId, itemId, features and label
    :param column_info: ColumnFeatureInfo specify information of different features
    :return: an array of tensors as input for wide part of a WideAndDeep model
    """

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
            acc += wide_dims[i - 1]
            res = acc + index
        indices.append(res)
    values = np.ones(len(indices))
    shape = np.array([sum(wide_dims)])
    return JTensor.sparse(values, np.array(indices), shape)


def get_deep_tensors(row, column_info):
    """
    convert a row to tensors given column feature information of a WideAndDeep model

    :param row: Row of userId, itemId, features and label
    :param column_info: ColumnFeatureInfo specify information of different features
    :return: an array of tensors as input for deep part of a WideAndDeep model
    """

    ind_col = column_info.indicator_cols
    emb_col = column_info.embed_cols
    cont_col = column_info.continuous_cols

    ind_tensor = np.zeros(sum(column_info.indicator_dims), )
    # setup indicators
    acc = 0
    for i in range(0, len(ind_col)):
        index = row[ind_col[i]]
        if i == 0:
            res = index
        else:
            acc += column_info.indicator_dims[i - 1]
            res = acc + index
        ind_tensor[res] = 1

    emb_tensor = np.zeros(len(emb_col), )
    for i in range(0, len(emb_col)):
        emb_tensor[i] = float(row[emb_col[i]])

    cont_tensor = np.zeros(len(cont_col), )
    for i in range(0, len(cont_col)):
        cont_tensor[i] = float(row[cont_col[i]])

    has_ind = len(ind_col) > 0
    has_emd = len(emb_col) > 0
    has_cont = len(cont_col) > 0
    if (has_ind and has_emd and has_cont):
        deep_tensor = [ind_tensor, emb_tensor, cont_tensor]
    elif ((not has_ind) and has_emd and has_cont):
        deep_tensor = [emb_tensor, cont_tensor]
    elif (has_ind and (not has_emd) and has_cont):
        deep_tensor = [ind_tensor, cont_tensor]
    elif (has_ind and has_emd and (not has_cont)):
        deep_tensor = [ind_tensor, emb_tensor]
    elif ((not has_ind) and (not has_emd) and has_cont):
        deep_tensor = [cont_tensor]
    elif ((not has_ind) and has_emd and (not has_cont)):
        deep_tensor = [emb_tensor]
    elif (has_ind and (not has_emd) and (not has_cont)):
        deep_tensor = [ind_tensor]
    else:
        raise TypeError("Empty deep tensors")
    return deep_tensor


def row_to_sample(row, column_info, model_type="wide_n_deep"):
    """
    convert a row to sample given column feature information of a WideAndDeep model

    :param row: Row of userId, itemId, features and label
    :param column_info: ColumnFeatureInfo specify information of different features
    :return: TensorSample as input for WideAndDeep model
    """

    wide_tensor = get_wide_tensor(row, column_info)
    deep_tensor = get_deep_tensors(row, column_info)
    deep_tensors = [JTensor.from_ndarray(ele) for ele in deep_tensor]
    label = row[column_info.label]
    model_type = model_type.lower()
    if model_type == "wide_n_deep":
        feature = [wide_tensor] + deep_tensors
    elif model_type == "wide":
        feature = wide_tensor
    elif model_type == "deep":
        feature = deep_tensors
    else:
        raise TypeError("Unsupported model_type: %s" % model_type)
    return Sample.from_jtensor(feature, label)


def to_user_item_feature(row, column_info, model_type="wide_n_deep"):
    """
    convert a row to UserItemFeature given column feature information of a WideAndDeep model

    :param row: Row of userId, itemId, features and label
    :param column_info: ColumnFeatureInfo specify information of different features
    :return: UserItemFeature for recommender model
    """
    return UserItemFeature(row["userId"], row["itemId"],
                           row_to_sample(row, column_info, model_type))
