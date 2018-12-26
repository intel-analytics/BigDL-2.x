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

from bigdl.util.common import JavaValue, callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class Ranker(JavaValue):
    def evaluate_ndcg(self, x, k, threshold=0.0):
        """
        """
        callBigDlFunc(self.bigdl_type, "evaluateNDCG",
                      self.value, x, k, threshold)

    def evaluate_map(self, x, threshold=0.0):
        """
        """
        callBigDlFunc(self.bigdl_type, "evaluateMAP",
                      self.value, x, threshold)
