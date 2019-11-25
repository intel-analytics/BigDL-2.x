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

from bigdl.util.common import JavaValue
from zoo.common.utils import callZooFunc

if sys.version >= '3':
    long = int
    unicode = str


class Ranker(JavaValue):
    """
    Base class for Ranking models (e.g., TextMatcher and Ranker) that
    provides validation methods with different metrics.
    """

    def evaluate_ndcg(self, x, k, threshold=0.0):
        """
        Evaluate using normalized discounted cumulative gain on TextSet.

        :param x: TextSet. Each TextFeature should contain Sample with batch features and labels.
                  In other words, each Sample should be a batch of records having both positive
                  and negative labels.
        :param k: Positive int. Rank position.
        :param threshold: Float. If label > threshold, then it will be considered as
                          a positive record. Default is 0.0.

        :return: Float. NDCG result.
        """
        return callZooFunc(self.bigdl_type, "evaluateNDCG",
                           self.value, x, k, threshold)

    def evaluate_map(self, x, threshold=0.0):
        """
        Evaluate using mean average precision on TextSet.

        :param x: TextSet. Each TextFeature should contain Sample with batch features and labels.
                  In other words, each Sample should be a batch of records having both positive
                  and negative labels.
        :param threshold: Float. If label > threshold, then it will be considered as
                          a positive record. Default is 0.0.

        :return: Float. MAP result.
        """
        return callZooFunc(self.bigdl_type, "evaluateMAP",
                           self.value, x, threshold)
