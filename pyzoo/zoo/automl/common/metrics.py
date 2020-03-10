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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class Evaluator(object):
    """
    Evaluate metrics for y_true and y_pred
    """
    def __init__(self):
        self.metrics_func = {
            'mean_squared_error': mean_squared_error,
            'r_square': r2_score
        }

    def evaluate(self, metric, y_true, y_pred):
        if metric not in self.metrics_func.keys():
            raise ValueError("metric" + metric + "is not supported")
        return self.metrics_func[metric](y_true, y_pred, multioutput='raw_values')




