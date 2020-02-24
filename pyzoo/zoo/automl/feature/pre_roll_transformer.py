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
import os

from zoo.automl.feature.base import BaseEstimator, BaseTransformer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from zoo.automl.feature.feature_utils import check_is_fitted


class MinMaxStandardizer(BaseEstimator):
    """
    min max scaler
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.scaler_filename = "scaler.save"

    def fit(self, inputs):
        """
        :param inputs: numpy array
        :return:
        """
        self.scaler.fit(inputs)
        return self

    def transform(self, inputs, transform_cols=None):
        """
        inplace standard scale transform_cols of input data frame.
        :param inputs: input numpy array
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        self.scaler.transform(inputs)
        if is_train:
            self.scaler.fit(inputs[transform_cols].values)
        else:
            self.scaler.transform(inputs[transform_cols].values)

    def inverse_transform(self, target, transform_cols=None):
        """
        since the scaler may include the transform of extra feature columns, and the inverse target
        only include target columns. Maybe need to extract the target scale information and inverse
        transform by hand.
        :param target: target to be inverse transformed
        :return:
        """
        self.scaler.inverse_transform(target)

    def save(self, file_path):
        """
        save scaler into file
        :param file_path
        :return:
        """
        joblib.dump(self.scaler, os.path.join(file_path, self.scaler_filename))

    def restore(self, file_path):
        """
        restore scaler from file
        :param file_path
        :return:
        """
        self.scaler = joblib.load(os.path.join(file_path, self.scaler_filename))


class StandardNormalizer(PreRollTransformer):
    """
    standard scalar
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_filename = "scaler.save"

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        inplace standard scale transform_cols of input data frame.
        :param inputs: input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        if is_train:
            self.scaler.fit(inputs[transform_cols].values)
        else:
            self.scaler.transform(inputs[transform_cols].values)

    def inverse_transform(self, target):
        """
        since the scaler may include the transform of extra feature columns, and the inverse target
        only include target columns. Maybe need to extract the target scale information and inverse
        transform by hand.
        :param target: target to be inverse transformed
        :return:
        """
        self.scaler.inverse_transform(target)

    def save(self, file_path):
        """
        save scaler into file
        :param file_path
        :return:
        """
        joblib.dump(self.scaler, os.path.join(file_path, self.scaler_filename))

    def restore(self, file_path):
        """
        restore scaler from file
        :param file_path
        :return:
        """
        self.scaler = joblib.load(os.path.join(file_path, self.scaler_filename))


class FeatureGenerator(PreRollTransformer):
    """
    generate features for input data frame
    """
    def __init__(self, generate_feature_names=None):
        """
        Constructor.
        :param generate_feature_names: a subset of
            {"month", "weekday", "day", "hour", "is_weekend", "IsAwake", "IsBusyHours"}
        """
        self.generated_feature_names = generate_feature_names

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        fit data with the input
        :param inputs: input data frame
        :param transform_cols: columns to be added into output
        :param is_train: indicate whether in training mode
        :return: numpy array. shape is (len(inputs), len(transform_cols) + len(features))
        """
        pass

    @staticmethod
    def get_allowed_features():
        return {"month", "weekday", "day", "hour", "is_weekend", "IsAwake", "IsBusyHours"}


class DeTrending(PreRollTransformer):

    def __init__(self):
        self.trend = None

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        inplace detrending transform_cols of input data frame.
        :param inputs: input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        if is_train:
            self.trend = "detrending output"
        detrended_values = inputs[transform_cols].values - self.trend
        output_df = pd.DataFrame(data=detrended_values, columns=transform_cols)
        return output_df

    def inverse_transform(self, target):
        """
        add trend for target data
        :param target: target to be inverse transformed
        :return:
        """
        pass


class Deseasonalizing(PreRollTransformer):

    def __init__(self):
        self.seasonality = None

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        inplace deseasonalizing transform_cols of input data frame.
        :param inputs: input input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        if is_train:
            self.seasonality = "deseasonalizing output"
        deseasonalized_values = inputs[transform_cols].values - self.seasonality
        output_df = pd.DataFrame(data=deseasonalized_values, columns=transform_cols)
        return output_df

    def inverse_transform(self, target):
        """
        add seasonality for target data
        :param target: target to be inverse transformed
        :return:
        """
        pass


class LogTransformer(PreRollTransformer):

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        inplace log transforming transform_cols of input data frame.
        :param inputs: input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        pass

    def inverse_transform(self, target):
        """
        inverse transform target value into origin magnitude
        :param target: target to be inverse transformed
        :return:
        """
        pass


PRE_ROLL_ORDER = {"min_max_norm": 1,
                  "standard_norm": 2,
                  "log_transfrom": 3,
                  "detrending": 4,
                  "deseasonalizing": 5,
                  "feature_generator": 6}
PRE_ROLL_TRANSFORMER_NAMES = set(PRE_ROLL_ORDER.keys())

PRE_ROLL_NAME2TRANSFORMER = {"min_max_norm": MinMaxNormalizer,
                             "standard_norm": StandardNormalizer,
                             "log_transfrom": LogTransformer,
                             "detrending": DeTrending,
                             "deseasonalizing": Deseasonalizing,
                             "feature_generator": FeatureGenerator}

PRE_ROLL_SAVE = (MinMaxNormalizer, StandardNormalizer)
PRE_ROLL_INVERSE = (MinMaxNormalizer, StandardNormalizer,
                    LogTransformer, DeTrending, Deseasonalizing)

TRANSFORM_TARGET_COL = (DeTrending, Deseasonalizing)
