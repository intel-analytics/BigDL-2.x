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

from zoo.models.common import ZooModel
from zoo.models.recommendation import Recommender
from bigdl.util.common import callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class ColumnFeatureInfo(object):
    """
    The same data information shared by the WideAndDeep model and its feature generation part.

    Each instance could contain the following fields:
    wide_base_cols: Data of wide_base_cols together with wide_cross_cols will be fed
                    into the wide model. List of String. Default is an empty list.
    wide_base_dims: Dimensions of wide_base_cols. The dimensions of the data in
                    wide_base_cols should be within the range of wide_base_dims.
                    List of int. Default is an empty list.
    wide_cross_cols: Data of wide_cross_cols will be fed into the wide model.
                     List of String. Default is an empty list.
    wide_cross_dims: Dimensions of wide_cross_cols. The dimensions of the data in
                     wide_cross_cols should be within the range of wide_cross_dims.
                     List of int. Default is an empty list.
    indicator_cols: Data of indicator_cols will be fed into the deep model as multi-hot vectors.
                    List of String. Default is an empty list.
    indicator_dims: Dimensions of indicator_cols. The dimensions of the data in
                    indicator_cols should be within the range of indicator_dims.
                    List of int. Default is an empty list.
    embed_cols: Data of embed_cols will be fed into the deep model as embeddings.
                List of String. Default is an empty list.
    embed_in_dims: Input dimension of the data in embed_cols. The dimensions of the data in
                   embed_cols should be within the range of embed_in_dims.
                   List of int. Default is an empty list.
    embed_out_dims: The dimensions of embeddings. List of int. Default is an empty list.
    continuous_cols: Data of continuous_cols will be treated as continuous values for
                     the deep model. List of String. Default is an empty list.
    label: The name of the 'label' column. String. Default is 'label'.
    """
    def __init__(self, wide_base_cols=None, wide_base_dims=None, wide_cross_cols=None,
                 wide_cross_dims=None, indicator_cols=None, indicator_dims=None,
                 embed_cols=None, embed_in_dims=None, embed_out_dims=None,
                 continuous_cols=None, label="label", bigdl_type="float"):
        self.wide_base_cols = [] if not wide_base_cols else wide_base_cols
        self.wide_base_dims = [] if not wide_base_dims else [int(d) for d in wide_base_dims]
        self.wide_cross_cols = [] if not wide_cross_cols else wide_cross_cols
        self.wide_cross_dims = [] if not wide_cross_dims else [int(d) for d in wide_cross_dims]
        self.indicator_cols = [] if not indicator_cols else indicator_cols
        self.indicator_dims = [] if not indicator_dims else [int(d) for d in indicator_dims]
        self.embed_cols = [] if not embed_cols else embed_cols
        self.embed_in_dims = [] if not embed_in_dims else [int(d) for d in embed_in_dims]
        self.embed_out_dims = [] if not embed_out_dims else [int(d) for d in embed_out_dims]
        self.continuous_cols = [] if not continuous_cols else continuous_cols
        self.label = label
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return ColumnFeatureInfo, (self.wide_base_cols, self.wide_base_dims, self.wide_cross_cols,
                                   self.wide_cross_dims, self.indicator_cols, self.indicator_dims,
                                   self.embed_cols, self.embed_in_dims, self.embed_out_dims,
                                   self.continuous_cols, self.label)

    def __str__(self):
        return "ColumnFeatureInfo {wide_base_cols: %s, wide_base_dims: %s, wide_cross_cols: %s, " \
               "wide_cross_dims: %s, indicator_cols: %s, indicator_dims: %s, embed_cols: %s, " \
               "embed_cols: %s, embed_in_dims: %s, embed_out_dims: %s, continuous_cols: %s, " \
               "label: '%s'}" \
               % (self.wide_base_cols, self.wide_base_dims, self.wide_cross_cols,
                  self.wide_cross_dims, self.indicator_cols, self.indicator_dims,
                  self.embed_cols, self.embed_cols, self.embed_in_dims,
                  self.embed_out_dims, self.continuous_cols, self.label)


class WideAndDeep(Recommender):
    """
    The Wide and Deep model used for recommendation.

    # Arguments
    class_num: The number of classes. Positive int.
    column_info: An instance of ColumnFeatureInfo.
    model_type: String. 'wide', 'deep' and 'wide_n_deep' are supported. Default is 'wide_n_deep'.
    hidden_layers: Units of hidden layers for the deep model.
                   Tuple of positive int. Default is (40, 20, 10).
    """
    def __init__(self, class_num, column_info, model_type="wide_n_deep",
                 hidden_layers=(40, 20, 10), bigdl_type="float"):
        super(WideAndDeep, self).__init__(None, bigdl_type,
                                          model_type,
                                          int(class_num),
                                          [int(unit) for unit in hidden_layers],
                                          column_info.wide_base_cols,
                                          column_info.wide_base_dims,
                                          column_info.wide_cross_cols,
                                          column_info.wide_cross_dims,
                                          column_info.indicator_cols,
                                          column_info.indicator_dims,
                                          column_info.embed_cols,
                                          column_info.embed_in_dims,
                                          column_info.embed_out_dims,
                                          column_info.continuous_cols,
                                          column_info.label)

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing WideAndDeep model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadWideAndDeep", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = WideAndDeep
        return model
