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
                 item_embed=20, hidden_layers=(40, 20, 10), include_mf=True,
                 mf_embed=20, bigdl_type="float"):
        super(NeuralCF, self).__init__(None, bigdl_type,
                                       int(user_count),
                                       int(item_count),
                                       int(class_num),
                                       int(user_embed),
                                       int(item_embed),
                                       [int(unit) for unit in hidden_layers],
                                       include_mf,
                                       int(mf_embed))

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
        jmodel = callBigDlFunc(bigdl_type, "loadNeuralCF", path, weight_path)
        model = ZooModel._do_load(jmodel, bigdl_type)
        model.__class__ = NeuralCF
        return model
