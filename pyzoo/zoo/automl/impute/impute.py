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

from zoo.automl.common.util import save_config
from zoo.automl.impute.abstract import BaseImputation
from zoo.zouwu.preprocessing.impute.LastFill import LastFill


class LastFillImpute(BaseImputation):
    """
    LastFill imputation
    """
    def __init__(self):
        self.imputer = LastFill()

    def impute(self, input_df):
        assert self.imputer is not None
        df = self.imputer.impute(input_df)
        return df

    def restore(self, **config):
        self.imputer = LastFill()


class FillZeroImpute(BaseImputation):
    """
    FillZero imputation
    """
    def impute(self, input_df):
        input_df.fillna(0)
        return input_df
