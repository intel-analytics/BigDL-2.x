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

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from zoo.zouwu.preprocessing.abstract import BaseImpute


class LastFill(BaseImpute):
    """
    Impute missing data with last seen value
    """
    def __init__(self):
        """
        Construct model for last filling method
        """
        pass

    def impute(self, df):
        """
        impute data
        :params df: input dataframe
        :return: imputed dataframe
        """
        df.iloc[0] = df.iloc[0].fillna(0)
        return df.fillna(method='pad')

    def evaluate(self, df, drop_rate):
        """
        evaluate model by randomly drop some value
        :params df: input dataframe
        :params drop_rate: percentage value that will be randomly dropped
        :return: MSE results
        """
        missing = df.isna()*1
        df1 = self.impute(df)
        print("na", df.isna().sum().sum())
        missing = missing.to_numpy()
        mask = np.zeros(df.shape[0]*df.shape[1])
        idx = np.random.choice(mask.shape[0], int(mask.shape[0]*drop_rate), replace=False)
        mask[idx] = 1
        mask = np.reshape(mask, (df.shape[0], df.shape[1]))
        np_df = df.to_numpy()
        np_df[mask == 1] = None
        new_df = pd.DataFrame(np_df)
        impute_df = self.impute(new_df)
        print("NA", impute_df.isna().sum().sum())
        pred = impute_df.to_numpy()
        true = df1.to_numpy()
        print(pred)
        print("true", true)
        pred[missing == 1] = 0
        true[missing == 1] = 0
        a = metrics.mean_squared_error(true[:, 0], pred[:, 0])
        b = metrics.mean_squared_error(true[:, 1], pred[:, 1])
        return [a, b]
