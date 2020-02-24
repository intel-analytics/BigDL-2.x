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

from zoo.automl.feature.base import PostRollTransformer
import numpy as np


class MinusFirst(PostRollTransformer):
    """
    minus the first data of a rolling window
    """
    def __init__(self):
        self.first_x = None

    def transform(self, inputs, is_train=False):
        """
        minus the values corresponding to the first datetime in a rolling window.
        :param inputs: a tuple of (x, y). y can be None in test mode.
        :param is_train: indicate whether in training mode
        :return: a tuple of (out_x, out_y).
            The shape of out_x is (x.shape[0], x.shape[1] - 1, x.shape[2])
            The shape of out_y is the same as input y.
        """
        x, y = inputs
        self.first_x = x[:, 0]
        first_extension_x = np.stack([self.first_x] * x.shape[1], axis=1)
        out_x = (x - first_extension_x)[:, 1:]
        if not y:
            return out_x, None
        else:
            first_extension_y = np.stack([self.first_x] * y.shape[1], axis=1)
            out_y = (y - first_extension_y)
            return out_x, out_y

    def inverse_transform(self, target):
        """
        inverse transform target value into origin magnitude
        :param target: ndarray of transformed y. Shape = (num_sample, horizon length, target num)
        :return: ndarray of inverse transformed y.
        """
        if not self.first_x:
            raise ValueError("You must call MinusFirst().transform first before calling"
                             " MinusFirst().inverse_transform!")
        first_extension_y = np.stack([self.first_x] * target.shape[1], axis=1)
        y = target + first_extension_y
        return y


class AutoEncoder(PostRollTransformer):
    """
    auto encoder
    """

    def transform(self, inputs, is_train=False):
        """
        Take a tuple of (x,y) as input for the interface uniformity. AutoEncoder only transform x
        into output_x, and the output y is the identity y.
        :param inputs: (x,y)
        :param is_train: indicate whether in training mode
        :return: (output_x,y). shape of output_x = [num_samples, num_features, 1], y is the same as input
        """
        pass

    def save(self, file_path):
        """
        save the status of Transformer
        :param file_path
        :return:
        """
        pass

    def restore(self, file_path):
        """
        restore the status saved
        :param file_path
        :return:
        """
        pass


class Aggregation(PostRollTransformer):
    """
    Aggregation for MLP
    """
    def transform(self, inputs, is_train=False):
        """
        fit data with the input
        :param inputs: (x,y)
        :param is_train: indicate whether in training mode
        :return: numpy array. shape = [num_samples, num_features, 1]
        """
        pass


POST_ROLL_ORDER = {"minus_first": 1,
                   "auto_encoder": 2,
                   "aggreg": 3}

POST_ROLL_TRANSFORMER_NAMES = set(POST_ROLL_ORDER.keys())

POST_ROLL_NAME2TRANSFORMER = {"minus_first": MinusFirst,
                              "auto_encoder": AutoEncoder,
                              "aggreg": Aggregation}

POST_ROLL_SAVE = AutoEncoder

POST_ROLL_INVERSE = MinusFirst
