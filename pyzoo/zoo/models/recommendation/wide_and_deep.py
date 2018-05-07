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

import sys

from zoo.models.common.zoo_model import ZooModel
from zoo.models.recommendation.recommender import Recommender
from bigdl.util.common import callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class ColumnFeatureInfo(object):
    """
    The same data information shared by WideAndDeep model and its feature generation part.

    Each record should contain the following fields:
    """
    def __init__(self, wide_base_cols=None, wide_base_dims=None, wide_cross_cols=None,
                 wide_cross_dims=None, indicator_cols=None, indicator_dims=None,
                 embed_cols=None, embed_in_dims=None, embed_out_dims=None,
                 continuous_cols=None, label="label", bigdl_type="float"):
        self.wide_base_cols = [] if not wide_base_cols else wide_base_cols
        self.wide_base_dims = [] if not wide_base_dims else wide_base_dims
        self.wide_cross_cols = [] if not wide_cross_cols else wide_cross_cols
        self.wide_cross_dims = [] if not wide_cross_dims else wide_cross_dims
        self.indicator_cols = [] if not indicator_cols else indicator_cols
        self.indicator_dims = [] if not indicator_dims else indicator_dims
        self.embed_cols = [] if not embed_cols else embed_cols
        self.embed_in_dims = [] if not embed_in_dims else embed_in_dims
        self.embed_out_dims = [] if not embed_out_dims else embed_out_dims
        self.continuous_cols = [] if not continuous_cols else continuous_cols
        self.label = label
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return ColumnFeatureInfo, (self.wide_base_cols, self.wide_base_dims, self.wide_cross_cols,
                                   self.wide_cross_dims, self.indicator_cols, self.indicator_dims,
                                   self.embed_cols, self.embed_in_dims, self.embed_out_dims,
                                   self.continuous_cols, self.label)

    def __str__(self):
        return "ColumnFeatureInfo [wide_base_cols: %s, wide_base_dims: %s, wide_cross_cols: %s, " \
               "wide_cross_dims: %s, indicator_cols: %s, indicator_dims: %s, embed_cols: %s, " \
               "embed_cols: %s, embed_in_dims: %s, embed_out_dims: %s, continuous_cols: %s, label: %s]"\
               % (self.wide_base_cols, self.wide_base_dims, self.wide_cross_cols, self.wide_cross_dims,
                  self.indicator_cols, self.indicator_dims, self.embed_cols, self.embed_cols,
                  self.embed_in_dims, self.embed_out_dims, self.continuous_cols, self.label)


class WideAndDeep(Recommender):
    """
    The Wide and Deep model used for recommendation.

    # Arguments
    """
    def __init__(self, class_num, col_info, model_type="wide_n_deep",
                 hidden_layers=(40, 20, 10), bigdl_type="float"):
        super(WideAndDeep, self).__init__(None, bigdl_type,
                                          model_type,
                                          class_num,
                                          hidden_layers,
                                          col_info.wide_base_cols,
                                          col_info.wide_base_dims,
                                          col_info.wide_cross_cols,
                                          col_info.wide_cross_dims,
                                          col_info.indicator_cols,
                                          col_info.indicator_dims,
                                          col_info.embed_cols,
                                          col_info.embed_in_dims,
                                          col_info.embed_out_dims,
                                          col_info.continuous_cols,
                                          col_info.label)

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing WideAndDeep model (with weights).

        # Arguments
        path: The path to save the model. Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadWideAndDeep", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = WideAndDeep
        return model
