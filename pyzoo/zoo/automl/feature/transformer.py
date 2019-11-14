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

from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    """
        Abstract Base class for basic feature transformers.
    """

    @abstractmethod
    def transform(self, inputs, is_train=False):
        """
        fit data with the input
        :param inputs: input to be fitted
        :param is_train: indicate whether in training mode
        :return:
        """

    def inverse_transform(self, target):
        """
        inverse transform target value into origin magnitude
        :param target: target to be inverse transformed
        :return:
        """
        return target

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


class PreRollTransformer(BaseTransformer):
    """
    base class for pre_roll transformers.
    """

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        fit data with the input
        :param inputs: input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        raise NotImplementedError()

    def inverse_transform(self, target):
        """
        inverse transform target value into origin magnitude
        :param target: target to be inverse transformed
        :return:
        """
        return target

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


class RollTransformer(BaseTransformer):
    """
    base class for roll transformers.
    """
    def __init__(self, horizon=1):
        """
        :param horizon: an int or a range or a list
        """
        self.horizon = horizon

    def transform(self, inputs, past_seq_len=2, is_train=False):
        """
        fit data with the input
        :param inputs: numpy array
        :param past_seq_len: the look back sequence length that need to unrolled
        :param is_train: indicate whether in training mode
        :return:
        """
        raise NotImplementedError()


class PostRollTransformer(BaseTransformer):
    """
    base class for post_roll transformers.
    """

    def transform(self, inputs, is_train=False):
        """
        fit data with the input
        :param inputs: (x, y)
        :param is_train: indicate whether in training mode
        :return: (x,y)
        """
        raise NotImplementedError()

    def inverse_transform(self, target):
        """
        inverse transform target value into origin magnitude
        :param target: target to be inverse transformed
        :return:
        """
        return target

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
