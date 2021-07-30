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

from zoo.models.common import KerasZooModel
from zoo.models.recommendation import Recommender
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
from zoo.common.utils import callZooFunc

if sys.version >= '3':
    long = int
    unicode = str


class SessionRecommender(Recommender):
    """
    The Session Recommender model used for recommendation.

    # Arguments
     item_ount: The number of distinct items. Positive integer.
     item_embed: The output size of embedding layer. Positive integer.
     rnn_hidden_layers: Units of hidden layers for the mlp model. Array of positive integers.
     session_length: The max number of items in the sequence of a session
     include_history: Whether to include purchase history. Boolean. Default is true.
     mlp_hidden_layers: Units of hidden layers for the mlp model. Array of positive integers.
     history_length: The max number of items in the sequence of historical purchase
     """

    def __init__(self, item_count, item_embed, rnn_hidden_layers=[40, 20], session_length=0,
                 include_history=False, mlp_hidden_layers=[40, 20], history_length=0,
                 bigdl_type="float"):
        assert session_length > 0, "session_length should align with input features"
        if include_history:
            assert history_length > 0, "history_length should align with input features"
        self.item_count = int(item_count)
        self.item_embed = int(item_embed)
        self.mlp_hidden_layers = [int(unit) for unit in mlp_hidden_layers]
        self.rnn_hidden_layers = [int(unit) for unit in rnn_hidden_layers]
        self.include_history = include_history
        self.session_length = int(session_length)
        self.history_length = int(history_length)
        self.bigdl_type = bigdl_type
        self.model = self.build_model()
        super(SessionRecommender, self).__init__(None, self.bigdl_type,
                                                 self.item_count,
                                                 self.item_embed,
                                                 self.rnn_hidden_layers,
                                                 self.session_length,
                                                 self.include_history,
                                                 self.mlp_hidden_layers,
                                                 self.history_length,
                                                 self.model)

    def build_model(self):
        input_rnn = Input(shape=(self.session_length,))
        session_table = Embedding(self.item_count + 1, self.item_embed, init="uniform")(input_rnn)

        gru = GRU(self.rnn_hidden_layers[0], return_sequences=True)(session_table)
        for hidden in range(1, len(self.rnn_hidden_layers) - 1):
            gru = GRU(self.rnn_hidden_layers[hidden], return_sequences=True)(gru)
        gru_last = GRU(self.rnn_hidden_layers[-1], return_sequences=False)(gru)
        rnn = Dense(self.item_count)(gru_last)

        if self.include_history:
            input_mlp = Input(shape=(self.history_length,))
            his_table = Embedding(self.item_count + 1, self.item_embed, init="uniform")(input_mlp)
            embedSum = KerasLayerWrapper(Sum(dimension=2))(his_table)
            flatten = Flatten()(embedSum)
            mlp = Dense(self.mlp_hidden_layers[0], activation="relu")(flatten)
            for hidden in range(1, len(self.mlp_hidden_layers)):
                mlp = Dense(self.mlp_hidden_layers[hidden], activation="relu")(mlp)
            mlp_last = Dense(self.item_count)(mlp)
            merged = merge(inputs=[rnn, mlp_last], mode="sum")
            out = Activation(activation="softmax")(merged)
            model = Model(input=[input_rnn, input_mlp], output=out)
        else:
            out = Activation(activation="softmax")(rnn)
            model = Model(input=input_rnn, output=out)
        return model

    def recommend_for_user(self, feature_rdd, max_items):
        raise Exception("recommend_for_user: Unsupported for SessionRecommender")

    def recommend_for_item(self, feature_rdd, max_users):
        raise Exception("recommend_for_item: Unsupported for SessionRecommender")

    def predict_user_item_pair(self, feature_rdd):
        raise Exception("predict_user_item_pair: Unsupported for SessionRecommender")

    def recommend_for_session(self, sessions, max_items, zero_based_label):
        """
        recommend for sessions given rdd of samples or list of samples.

        # Arguments
        sessions: rdd of samples or list of samples.
        max_items:   Number of items to be recommended to each user. Positive integer.
        zero_based_label: True if data starts from 0, False if data starts from 1
        :return rdd of list of list(item, probability),
        """
        if isinstance(sessions, list):
            sc = get_spark_context()
            sessions_rdd = sc.parallelize(sessions)
        elif (isinstance(sessions, RDD)):
            sessions_rdd = sessions
        else:
            raise TypeError("Unsupported training data type: %s" % type(sessions))
        results = callZooFunc(self.bigdl_type, "recommendForSession",
                              self.value,
                              sessions_rdd,
                              max_items,
                              zero_based_label)

        if isinstance(sessions, list):
            return results.collect()
        else:
            return results

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing SessionRecommender model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callZooFunc(bigdl_type, "loadSessionRecommender", path, weight_path)
        model = KerasZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = SessionRecommender
        return model
