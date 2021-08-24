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

from zoo.models.common import *
from zoo.models.recommendation import Recommender
from zoo.common.utils import callZooFunc
from zoo.pipeline.api.keras.layers import *

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

    @property
    def feature_cols(self):
        return self.wide_base_cols + self.wide_cross_cols +\
            self.indicator_cols + self.embed_cols + self.continuous_cols

    @property
    def label_cols(self):
        return [self.label]


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
                 hidden_layers=[40, 20, 10], bigdl_type="float"):
        assert len(column_info.wide_base_cols) == len(column_info.wide_base_dims), \
            "size of wide_base_columns should match"
        assert len(column_info.wide_cross_cols) == len(column_info.wide_cross_dims), \
            "size of wide_cross_columns should match"
        assert len(column_info.indicator_cols) == len(column_info.indicator_dims), \
            "size of wide_indicator_columns should match"
        assert len(column_info.embed_cols) == len(column_info.embed_in_dims) \
            == len(column_info.embed_out_dims), "size of wide_indicator_columns should match"

        self.class_num = int(class_num)
        self.wide_base_dims = column_info.wide_base_dims
        self.wide_cross_dims = column_info.wide_cross_dims
        self.indicator_dims = column_info.indicator_dims
        self.embed_in_dims = column_info.embed_in_dims
        self.embed_out_dims = column_info.embed_out_dims
        self.continuous_cols = column_info.continuous_cols
        self.model_type = model_type
        self.hidden_layers = [int(unit) for unit in hidden_layers]
        self.bigdl_type = bigdl_type
        self.model = self.build_model()
        super(WideAndDeep, self).__init__(None, self.bigdl_type,
                                          self.model_type,
                                          self.class_num,
                                          self.hidden_layers,
                                          self.wide_base_dims,
                                          self.wide_cross_dims,
                                          self.indicator_dims,
                                          self.embed_in_dims,
                                          self.embed_out_dims,
                                          self.continuous_cols,
                                          self.model)

    def build_model(self):
        wide_dims = sum(self.wide_base_dims) + sum(self.wide_cross_dims)
        input_wide = Input(shape=(wide_dims,))
        input_ind = Input(shape=(sum(self.indicator_dims),))
        input_emb = Input(shape=(len(self.embed_in_dims),))
        input_con = Input(shape=(len(self.continuous_cols),))

        wide_linear = SparseDense(self.class_num)(input_wide)

        if (self.model_type == "wide"):
            out = Activation("softmax")(wide_linear)
            model = Model(input_wide, out)
        elif (self.model_type == "deep"):
            (input_deep, merge_list) = self._deep_merge(input_ind, input_emb, input_con)
            deep_linear = self._deep_hidden(merge_list)
            out = Activation("softmax")(deep_linear)
            model = Model(input_deep, out)
        elif (self.model_type == "wide_n_deep"):
            (input_deep, merge_list) = self._deep_merge(input_ind, input_emb, input_con)
            deep_linear = self._deep_hidden(merge_list)
            merged = merge([wide_linear, deep_linear], "sum")
            out = Activation("softmax")(merged)
            model = Model([input_wide] + input_deep, out)

        else:
            raise TypeError("Unsupported model_type: %s" % self.model_type)

        return model

    def _deep_hidden(self, merge_list):
        if (len(merge_list) == 1):
            merged = merge_list[0]
        else:
            merged = merge(merge_list, "concat")
        linear = Dense(self.hidden_layers[0], activation="relu")(merged)

        for ilayer in range(1, len(self.hidden_layers)):
            linear_mid = Dense(self.hidden_layers[ilayer], activation="relu")(linear)
            linear = linear_mid
        last = Dense(self.class_num, activation="relu")(linear)
        return last

    def _deep_merge(self, input_ind, input_emb, input_cont):
        embed_width = 0
        embed = []
        for i in range(0, len(self.embed_in_dims)):
            flat_select = Flatten()(Select(1, embed_width)(input_emb))
            iembed = Embedding(self.embed_in_dims[i] + 1, self.embed_out_dims[i],
                               init="normal")(flat_select)
            flat_embed = Flatten()(iembed)
            embed.append(flat_embed)
            embed_width = embed_width + 1

        has_ind = len(self.indicator_dims) > 0
        has_emd = len(self.embed_in_dims) > 0
        has_cont = len(self.continuous_cols) > 0
        if (has_ind and has_emd and has_cont):
            input = [input_ind, input_emb, input_cont]
            merged_list = [input_ind] + embed + [input_cont]
        elif (not has_ind and has_emd and has_cont):
            input = [input_emb, input_cont]
            merged_list = embed + [input_cont]
        elif (has_ind and (not has_emd) and has_cont):
            input = [input_ind, input_cont]
            merged_list = [input_ind, input_cont]
        elif (has_ind and has_emd and (not has_cont)):
            input = [input_ind, input_emb]
            merged_list = [input_ind] + embed
        elif ((not has_ind) and (not has_emd) and has_cont):
            input = [input_cont]
            merged_list = [input_cont]
        elif ((not has_ind) and has_emd and (not has_cont)):
            input = [input_emb]
            merged_list = embed
        elif (has_ind and (not has_emd) and not (has_cont)):
            input = [input_ind]
            merged_list = [input_ind]
        else:
            raise TypeError("Empty deep model for: %s" % self.model_type)

        return (input, merged_list)

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
        jmodel = callZooFunc(bigdl_type, "loadWideAndDeep", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        labor_model = KerasZooModel._do_load(jmodel, bigdl_type)
        model.model = labor_model
        model.__class__ = WideAndDeep
        return model
