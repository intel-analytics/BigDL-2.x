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

import pandas as pd
import numpy as np

class TSDataset:
    def __init__(self):
        '''
        TSDataset is an abstract of time series dataset.
        '''
    
    @staticmethod
    def from_pandas(df,
                    id_col,
                    datetime_col,
                    target_col,
                    extra_feature_col=None):
        '''
        Initialize a tsdataset from pandas dataframe.
        :param df: a pandas dataframe for your raw time series data.
        :param id_col: a str indicates the col name of dataframe id.
        :param datetime_col: a str indicates the col name of datetime 
               column in the input data frame.
        :param target_col: a str or list indicates the col name of target column
               in the input data frame.
        :param extra_feature_col: (optional) a str or list indicates the col name
               of extra feature columns that needs to predict the target column.
        Here is an df example:
        id        datetime      value   "extra feature 1"   "extra feature 2"
        00        2019-01-01    1.9     1                   2
        01        2019-01-01    2.3     0                   9
        00        2019-01-02    2.4     3                   4
        01        2019-01-02    2.6     0                   2
        ```python
        tsdataset = TSDataset.from_pandas(df, datetime_col="datetime",
                                          target_col="value", id_col="id",
                                          extra_feature_col=["extra feature 1",""extra feature 2"])
       ```
        '''
        pass

    @staticmethod
    def from_numpy(data,
                   target_idx,
                   extra_feature_idx=None):
        '''
        Initialize a tsdataset from numpy ndarray.
        We currently only support rolling operation on tsdataset initialized from numpy ndarray
        TODO: design id_idx API
        :param data: numpy ndarray with two dimension, the first dimension is time, the second 
               dimension is feature
        :param target_idx: int or list, target feature column index
        :param extra_feature_idx: int or list, extra feature column index
        Here is a ndarray example:
        [
        [1.9    1    2]
        [2.3    0    9]
        [2.4    3    4]
        [2.6    0    2]
        ]
        ```python
        tsdataset = TSDataset.from_numpy(data, target_idx=0,
                                         extra_feature_col=[1,2])
        ```
        '''
        pass

    def impute(self, mode="LastFillImpute", reindex=False):
        '''
        Impute the tsdataset
        :param mode: a str defaulted to "LastFillImpute" indicates imputing method.
               "FillZeroImpute" and "LastFillImpute" are supported for now. 
        :param reindex: indicates if we need to reindex the datetime to fill in the
               missing datetime.
        '''
        return self
    
    def deduplicate(self, mode="mean"):
        '''
        Merge those rows whose timestamp are seconds apart
        :param mode: str, One of "max", "min", "mean", "sum".
        '''
        return self
    
    def gen_dt_feature(self):
        '''
        Generate datetime feature for each row. 
        Currently we generate following features:
            "MINUTE", "DAY", "DAYOFYEAR", "HOUR", "WEEKDAY",
            "WEEKOFYEAR", "MONTH", "IS_AWAKE", "IS_BUSY_HOURS",
            "IS_WEEKEND"
        '''
        return self

    def rolling(self, lookback, horizon, feature_col=None, target_col=None, inplace=False):
        '''
        Sampling by rolling
        :param lookback: int, lookback value
        :param horizon: int or list, 
               if `horizon` is an int, we will sample `horizon` step  continuously after the forecasting point.
               if `horizon` is an list, we will sample discretely according to the input list.
        :param feature_col: str or list, indicate the feature col name. Default to None, where we will take all
               avaliable feature in rolling.
        :param target_col: str or list, indicate the target col name. Default to None, where we will take all
               target in rolling.
        :param inplace: bool,
               if True, we will save the rolling result for future to_numpy() calling.
               if False, we will return the rolling result directly.
        '''

    def to_numpy(self):
        '''
        export rolling result in form of a tuple of numpy ndarray (x, y)
        '''

    def to_pandas(self):
        '''
        export the pandas dataframe 
        '''
    


