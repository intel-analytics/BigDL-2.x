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

from zoo.zouwu.preprocessing.impute.abstract import BaseImpute


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
