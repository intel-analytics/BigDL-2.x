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
import numpy as np


def check_input(y_true, y_pred, multioutput):
    if y_true is None or y_pred is None:
        raise ValueError("The input is None.")
    if not isinstance(y_true, (list, tuple, np.ndarray)):
        raise ValueError("Expected array-like input. Only list/tuple/ndarray are supported")
    if isinstance(y_true, (list, tuple)):
        y_true = np.array(y_true)
    if isinstance(y_pred, (list, tuple)):
        y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred have different number of samples "
                         "({0}!={1})".format(y_true.shape[0], y_pred.shape[0]))
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                              allowed_multioutput_str,
                              multioutput))

    return y_true, y_pred


def sMAPE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate Symmetric mean absolute percentage error (sMAPE).
    :param y_true: array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    :param multioutput: string in ['raw_values', 'uniform_average']
    :return:float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    y_true, y_pred = check_input(y_true, y_pred, multioutput)
    output_errors = np.mean(100 * 2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


class Evaluator(object):
    """
    Evaluate metrics for y_true and y_pred
    """

    metrics_func = {
        'mean_squared_error': mean_squared_error,
        'r_square': r2_score,
        'sMAPE': sMAPE
    }

    @staticmethod
    def evaluate(metric, y_true, y_pred):
        if not Evaluator.check_metric(metric):
            raise ValueError("metric " + metric + " is not supported")
        return Evaluator.metrics_func[metric](y_true, y_pred, multioutput='raw_values')

    @staticmethod
    def check_metric(metric):
        return True if metric in Evaluator.metrics_func.keys() else False
