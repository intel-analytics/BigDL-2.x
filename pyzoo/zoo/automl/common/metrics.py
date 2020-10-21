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

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_log_error as MSLE
import numpy as np


EPSILON = 1e-10


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
                             "You provided multioutput={!r}"
                             .format(allowed_multioutput_str, multioutput))

    return y_true, y_pred


def sMAPE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate Symmetric mean absolute percentage error (sMAPE).
    <math> \text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^n \frac{|F_t-A_t|}{|A_t|+|F_t|}</math>
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
    output_errors = np.mean(100 * np.abs(y_true - y_pred) /
                            (np.abs(y_true) + np.abs(y_pred) + EPSILON), axis=0,)
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


def MPE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate mean percentage error (MPE).
    <math> \text{MPE} = \frac{100\%}{n}\sum_{t=1}^n \frac{a_t-f_t}{a_t} </math>
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
    output_errors = np.mean(100 * (y_true - y_pred) / (y_true + EPSILON), axis=0,)
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


def MAPE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate mean absolute percentage error (MAPE).
    <math>\mbox{M} = \frac{100\%}{n}\sum_{t=1}^n  \left|\frac{A_t-F_t}{A_t}\right|, </math>
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
    output_errors = np.mean(100 * np.abs((y_true - y_pred) / (y_true + EPSILON)), axis=0,)
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


def MDAPE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate Median Absolute Percentage Error (MDAPE).
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
    output_errors = np.median(100 * np.abs((y_true - y_pred) / (y_true + EPSILON)), axis=0,)
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


def sMDAPE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate Symmetric Median Absolute Percentage Error (sMDAPE).
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
    output_errors = np.median(100 * np.abs(y_true - y_pred) /
                              (np.abs(y_true) + np.abs(y_pred) + EPSILON), axis=0, )
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


def ME(y_true, y_pred, multioutput='raw_values'):
    """
    calculate Mean Error (ME).
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
    output_errors = np.mean(y_true - y_pred, axis=0,)
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


def MSPE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate mean squared percentage error (MSPE).
    <math>\operatorname{MSPE}(L)=\operatorname{E}
    \left[\left( g(x_i)-\widehat{g}(x_i)\right)^2\right].</math>
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
    output_errors = np.mean(np.square(y_true - y_pred), axis=0,)
    if multioutput == 'raw_values':
        return output_errors
    return np.mean(output_errors)


def RMSE(y_true, y_pred, multioutput='raw_values'):
    """
    calculate square root of the mean squared error (RMSE).
    :param y_true: array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    :param multioutput: string in ['raw_values', 'uniform_average']
    :return:float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    return np.sqrt(MSE(y_true, y_pred, multioutput=multioutput))


def Accuracy(y_true, y_pred, multioutput=None):
    from sklearn.metrics._classification import accuracy_score
    return accuracy_score(y_true, y_pred)


class Evaluator(object):
    """
    Evaluate metrics for y_true and y_pred
    """

    metrics_func = {
        # Absolute
        'me': ME,
        'mae': MAE,
        'mse': MSE,
        'rmse': RMSE,
        'msle': MSLE,
        'r2': R2,
        # Relative
        'mpe': MPE,
        'mape': MAPE,
        'mspe': MSPE,
        'smape': sMAPE,
        'mdape': MDAPE,
        'smdape': sMDAPE,
        'accuracy': Accuracy,
    }

    max_mode_metrics = ('r2', 'accuracy')

    @staticmethod
    def evaluate(metric, y_true, y_pred, multioutput='raw_values'):
        Evaluator.check_metric(metric)
        return Evaluator.metrics_func[metric](y_true, y_pred, multioutput=multioutput)

    @staticmethod
    def check_metric(metric):
        if metric not in Evaluator.metrics_func.keys():
            raise ValueError("metric " + metric + " is not supported")

    @staticmethod
    def get_metric_mode(metric):
        if metric in Evaluator.max_mode_metrics:
            return "max"
        else:
            return "min"
