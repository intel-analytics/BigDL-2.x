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


class BaseImpute(ABC):
    """
    base model for data imputation
    """
    
    @abstractmethod
    def impute(self, df):
        """
        fill in missing values in the dataframe
        :param df: dataframe containing missing values
        :return: dataframe without missing values
        """
        raise NotImplementError
        
    @abstractmethod
    def evaluate(self, df, drop_rate):
        """
        randomly drop some values and evaluate the data imputation method
        :param df: input dataframe (better without missing values)
        :param drop_rate: percentage value of randomly dropping data
        :return: MSE results
        """
        raise NotImplementError