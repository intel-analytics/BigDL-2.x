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


class BaseImputation(ABC):
    """
    Abstract Base class for imputation.
    """

    @abstractmethod
    def impute(self, input_df):
        """
        fill na value
        :param input_df: input dataframe
        :return:
        """
        pass

    def save(self, file_path):
        """
        save the feature tools internal variables.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file_path : the file to be saved
        :param: config: the trial config
        :return:
        """
        pass

    def restore(self, **config):
        """
        Restore variables from file
        :param file_path: file contain saved parameters.
                          i.e. some parameters are obtained during training,
                          not in trial config, e.g. scaler fit params)
        :param config: the trial config
        :return:
        """
        pass
