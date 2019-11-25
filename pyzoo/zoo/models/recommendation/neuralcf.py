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


class NeuralCF(Recommender):
    """
    The neural collaborative filtering model used for recommendation.

    # Arguments
    user_count: The number of users. Positive int.
    item_count: The number of classes. Positive int.
    class_num: The number of classes. Positive int.
    user_embed: Units of user embedding. Positive int. Default is 20.
    item_embed: itemEmbed Units of item embedding. Positive int. Default is 20.
    hidden_layers: Units of hidden layers for MLP. Tuple of positive int. Default is (40, 20, 10).
    include_mf: Whether to include Matrix Factorization. Boolean. Default is True.
    mf_embed: Units of matrix factorization embedding. Positive int. Default is 20.
    """

    def __init__(self, user_count, item_count, class_num, user_embed=20,
                 item_embed=20, hidden_layers=[40, 20, 10], include_mf=True,
                 mf_embed=20, bigdl_type="float"):
        self.user_count = int(user_count)
        self.item_count = int(item_count)
        self.class_num = int(class_num)
        self.user_embed = int(user_embed)
        self.item_embed = int(item_embed)
        self.hidden_layers = [int(unit) for unit in hidden_layers]
        self.include_mf = include_mf
        self.mf_embed = int(mf_embed)
        self.bigdl_type = bigdl_type
        self.model = self.build_model()
        super(NeuralCF, self).__init__(None, self.bigdl_type,
                                       self.user_count,
                                       self.item_count,
                                       self.class_num,
                                       self.user_embed,
                                       self.item_embed,
                                       self.hidden_layers,
                                       self.include_mf,
                                       self.mf_embed,
                                       self.model)

    def build_model(self):
        input = Input(shape=(2,))
        user_flat = Flatten()(Select(1, 0)(input))
        item_flat = Flatten()(Select(1, 1)(input))
        mlp_user_embed = Embedding(self.user_count + 1, self.user_embed, init="uniform")(user_flat)
        mlp_item_embed = Embedding(self.item_count + 1, self.item_embed, init="uniform")(item_flat)
        mlp_user_flat = Flatten()(mlp_user_embed)
        mlp_item_flat = Flatten()(mlp_item_embed)
        mlp_latent = merge(inputs=[mlp_user_flat, mlp_item_flat], mode="concat")
        linear1 = Dense(self.hidden_layers[0], activation="relu")(mlp_latent)
        mlp_linear = linear1
        for ilayer in range(1, len(self.hidden_layers)):
            linear_mid = Dense(self.hidden_layers[ilayer], activation="relu")(mlp_linear)
            mlp_linear = linear_mid

        if (self.include_mf):
            assert (self.mf_embed > 0)
            mf_user_embed = Embedding(self.user_count + 1, self.mf_embed, init="uniform")(user_flat)
            mf_item_embed = Embedding(self.item_count + 1, self.mf_embed, init="uniform")(item_flat)
            mf_user_flatten = Flatten()(mf_user_embed)
            mf_item_flatten = Flatten()(mf_item_embed)
            mf_latent = merge(inputs=[mf_user_flatten, mf_item_flatten], mode="mul")
            concated_model = merge(inputs=[mlp_linear, mf_latent], mode="concat")
            linear_last = Dense(self.class_num, activation="softmax")(concated_model)
        else:
            linear_last = Dense(self.class_num, activation="softmax")(mlp_linear)
        model = Model(input, linear_last)
        return model

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing NeuralCF model (with weights).

        # Arguments
        path: The path for the pre-defined model.
              Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path for pre-trained weights if any. Default is None.
        """
        jmodel = callZooFunc(bigdl_type, "loadNeuralCF", path, weight_path)
        model = KerasZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = NeuralCF
        return model
