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

from pyspark import RDD

from zoo.models.common import *
from zoo.common.utils import callZooFunc

if sys.version >= '3':
    long = int
    unicode = str


class UserItemFeature(object):
    """
    Represent records of user-item with features.

    Each record should contain the following fields:
    user_id: Positive int.
    item_id: Positive int.
    sample: Sample which consists of feature(s) and label(s).
    """

    def __init__(self, user_id, item_id, sample, bigdl_type="float"):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.sample = sample
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return UserItemFeature, (self.user_id, self.item_id, self.sample)

    def __str__(self):
        return "UserItemFeature [user_id: %s, item_id: %s, %s]" % (
            self.user_id, self.item_id, self.sample)


class UserItemPrediction(object):
    """
    Represent the prediction results of user-item pairs.

    Each prediction record will contain the following information:
    user_id: Positive int.
    item_id: Positive int.
    prediction: The prediction (rating) for the user on the item.
    probability: The probability for the prediction.
    """

    def __init__(self, user_id, item_id, prediction, probability, bigdl_type="float"):
        self.user_id = user_id
        self.item_id = item_id
        self.prediction = prediction
        self.probability = probability
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return UserItemPrediction, (self.user_id, self.item_id, self.prediction, self.probability)

    def __str__(self):
        return "UserItemPrediction [user_id: %s, item_id: %s, prediction: %s, probability: %s]" % (
            self.user_id, self.item_id, self.prediction, self.probability)


class Recommender(KerasZooModel):
    """
    The base class for recommendation models in Analytics Zoo.
    """

    def predict_user_item_pair(self, feature_rdd):
        """
        Predict for user-item pairs.

        # Arguments
        feature_rdd: RDD of UserItemFeature.
        :return RDD of UserItemPrediction.
        """
        result_rdd = callZooFunc(self.bigdl_type, "predictUserItemPair",
                                 self.value,
                                 self._to_tuple_rdd(feature_rdd))
        return self._to_prediction_rdd(result_rdd)

    def recommend_for_user(self, feature_rdd, max_items):
        """
        Recommend a number of items for each user.

        # Arguments
        feature_rdd: RDD of UserItemFeature.
        max_items: The number of items to be recommended to each user. Positive int.
        :return RDD of UserItemPrediction.
        """
        result_rdd = callZooFunc(self.bigdl_type, "recommendForUser",
                                 self.value,
                                 self._to_tuple_rdd(feature_rdd),
                                 int(max_items))
        return self._to_prediction_rdd(result_rdd)

    def recommend_for_item(self, feature_rdd, max_users):
        """
        Recommend a number of users for each item.

        # Arguments
        feature_rdd: RDD of UserItemFeature.
        max_users: The number of users to be recommended to each item. Positive int.
        :return RDD of UserItemPrediction.
        """
        result_rdd = callZooFunc(self.bigdl_type, "recommendForItem",
                                 self.value,
                                 self._to_tuple_rdd(feature_rdd),
                                 int(max_users))
        return self._to_prediction_rdd(result_rdd)

    @staticmethod
    def _to_tuple_rdd(feature_rdd):
        assert isinstance(feature_rdd, RDD), "feature_rdd should be RDD of UserItemFeature"
        return feature_rdd.map(lambda x: (x.user_id, x.item_id, x.sample))

    @staticmethod
    def _to_prediction_rdd(result_rdd):
        return result_rdd.map(lambda y: UserItemPrediction(int(y[0]), int(y[1]), int(y[2]), y[3]))
