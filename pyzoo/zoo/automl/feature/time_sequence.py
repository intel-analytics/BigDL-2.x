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
from featuretools import TransformFeature

from zoo.automl.common.util import save_config
from zoo.automl.feature.abstract import BaseFeatureTransformer

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import featuretools as ft
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Text, Numeric, DatetimeTimeIndex
import json


class TimeSequenceFeatureTransformer(BaseFeatureTransformer):
    """
    TimeSequence feature engineering
    """

    def __init__(self, future_seq_len=1, dt_col="datetime", target_col="value", extra_features_col=None, drop_missing=True):
        """
        Constructor.
        :param drop_missing: whether to drop missing values in the curve, if this is set to False, an error will be
        reported if missing values are found. If True, will drop the missing values and won't raise errors.
        """
        # self.scaler = MinMaxScaler()
        self.scaler = StandardScaler()
        self.config = None
        self.dt_col = dt_col
        self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.feature_data = None
        self.drop_missing = drop_missing
        self.generate_feature_list = None
        self.past_seq_len = None
        self.future_seq_len = future_seq_len

    def fit_transform(self, input_df, **config):
        """
        Fit data and transform the raw data to features. This is used in training for hyper parameter searching.
        This method will refresh the parameters (e.g. min and max of the MinMaxScaler) if any
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param config: tunable parameters
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
            length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        """
        self.config = self._get_feat_config(**config)
        self._check_input(input_df)
        # print(input_df.shape)
        feature_data = self._get_features(input_df, self.config)
        self.scaler.fit(feature_data)
        data_n = self._scale(feature_data)
        assert np.mean(data_n[0]) < 1e-5
        (x, y) = self._roll_train(data_n, past_seq_len=self.past_seq_len, future_seq_len=self.future_seq_len)

        return x, y

    def transform(self, input_df, is_train=True):
        """
        Transform data into features using the preset of configurations from fit_transform
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param is_train: If the input_df is for training.
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
            length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        """
        if self.config is None or self.past_seq_len is None:
            raise Exception("Needs to call fit_transform or restore first before calling transform")
        self._check_input(input_df)
        # generate features
        feature_data = self._get_features(input_df, self.config)
        # select and standardize data
        data_n = self._scale(feature_data)
        if is_train:
            (x, y) = self._roll_train(data_n, past_seq_len=self.past_seq_len, future_seq_len=self.future_seq_len)
            return x, y
        else:
            x = self._roll_test(data_n, past_seq_len=self.past_seq_len)
            self._get_y_pred_dt(input_df, self.past_seq_len)
            return x

    def post_processing(self, y_pred):
        """
        Used only in pipeline predict, after calling self.transform(input_df, is_train=False).
        Post_processing includes converting the predicted array to dataframe and scalar inverse transform.
        :param y_pred: Model prediction result (ndarray).
        :return: Un_scaled dataframe with datetime.
        """
        # for standard scalar
        value_mean = self.scaler.mean_[0]
        value_scale = self.scaler.scale_[0]
        y_unscale = y_pred * value_scale + value_mean
        self.y_pred_dt[self.target_col] = y_unscale
        return self.y_pred_dt

    def save(self, file_path, replace=False):
        """
        save the feature tools internal variables as well as the initialization args.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file : the file to be saved
        :return:
        """
        # for StandardScaler()
        data_to_save = {"mean": self.scaler.mean_.tolist(),
                        "scale": self.scaler.scale_.tolist(),
                        "future_seq_len": self.future_seq_len,
                        "dt_col": self.dt_col,
                        "target_col": self.target_col,
                        "extra_features_col": self.extra_features_col,
                        "drop_missing": self.drop_missing
                        }
        save_config(file_path, data_to_save, replace=replace)

        # with open(file_path, 'w') as output_file:
        #     # for StandardScaler()
        #     json.dump({"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist()}, output_file)
            # for minmaxScaler()
            # json.dump({"min": self.scaler.min_.tolist(), "scale": self.scaler.scale_.tolist()}, output_file)

    def restore(self, **config):
        """
        Restore variables from file
        :param file_path: the dumped variables file
        :return:
        """
#         with open(file_path, 'r') as input_file:
#             result = json.load(input_file)

        # for StandardScalar()
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.asarray(config["mean"])
        self.scaler.scale_ = np.asarray(config["scale"])

        self.config = self._get_feat_config(**config)

        self.future_seq_len = config["future_seq_len"]
        self.dt_col = config["dt_col"]
        self.target_col = config["target_col"]
        self.extra_features_col = config["extra_features_col"]
        self.drop_missing = config["drop_missing"]

        # for MinMaxScalar()
        # self.scaler = MinMaxScaler()
        # self.scaler.min_ = np.asarray(result["min"])
        # self.scaler.scale_ = np.asarray(result["scale"])
        # print(self.scaler.transform(input_data))

    def get_feature_list(self, input_df):
        feature_matrix, feature_defs = self._generate_features(input_df)
        return [feat.generate_name() for feat in feature_defs if isinstance(feat, TransformFeature)]

    def _get_feat_config(self, **config):
        """
        Get feature related arguments from global hyper parameter config and do necessary error checking
        :param config: the global config (usually from hyper parameter tuning)
        :return: config only for feature engineering
        """
        self._check_config(**config)
        feature_config_names = ["selected_features", "past_seq_len"]
        feat_config = {}
        for name in feature_config_names:
            if name not in config:
                continue
                # raise KeyError("Can not find " + name + " in config!")
            feat_config[name] = config[name]
        self.past_seq_len = feat_config.get("past_seq_len", 50)
        return feat_config

    def _check_input(self, input_df):
        """
        Check dataframe for integrity. Requires time sequence to come in uniform sampling intervals.
        :param input_df:
        :return:
        """
        # check NaT in datetime
        input_df = input_df.reset_index()
        dt = input_df[self.dt_col]
        if not np.issubdtype(dt, np.datetime64):
            raise ValueError("The dtype of datetime column is required to be np.datetime64!")
        is_nat = pd.isna(dt)
        if is_nat.any(axis=None):
            raise ValueError("Missing datetime in input dataframe!")

        # check uniform (is that necessary?)
        interval = dt[1] - dt[0]

        if not all([dt[i] - dt[i - 1] == interval for i in range(1, len(dt))]):
            raise ValueError("Input time sequence intervals are not uniform!")

        # check missing values
        if not self.drop_missing:
            is_nan = pd.isna(input_df)
            if is_nan.any(axis=None):
                raise ValueError("Missing values in input dataframe!")

        # check if the last datetime is large than current time. In that case, feature tools generate NaN.from
        last_datetime = dt.iloc[-1]
        current_time = np.datetime64('today', 's')
        if last_datetime > current_time:
            raise ValueError("Last date time is bigger than current time!")
        return input_df

    def _roll_data(self, data, seq_len):
        result = []
        mask = []
        for i in range(len(data) - seq_len + 1):
            result.append(data[i: i + seq_len])

            if pd.isna(data[i: i + seq_len]).any(axis=None):
                mask.append(0)
            else:
                mask.append(1)

        return np.asarray(result), np.asarray(mask)

    def _roll_train(self, dataframe, past_seq_len, future_seq_len):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        roll_train: split the whole dataset apart to build (x, y).
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seq_len: the length of the past sequence
        :param future_seq_len: the length of the future sequence
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future sequence
            length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence length = 1
        """
        max_past_seq_len = len(dataframe) - future_seq_len
        if past_seq_len > max_past_seq_len:
            raise ValueError("past_seq_len {} exceeds the maximum value {}".format(past_seq_len, future_seq_len))
        x = dataframe[0:-future_seq_len].values
        y = dataframe[past_seq_len:][0].values
        output_x, mask_x = self._roll_data(x, past_seq_len)
        output_y, mask_y = self._roll_data(y, future_seq_len)
        # assert output_x.shape[0] == output_y.shape[0], "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1) & (mask_y == 1)
        return output_x[mask], output_y[mask]

    def _roll_test(self, dataframe, past_seq_len):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        roll_test: the whole dataframe is regarded as x.
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seq_len: the length of the past sequence
        :return: x
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the last
            dimension, the 1st col is the time index (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
        """
        x = dataframe.values
        output_x, mask_x = self._roll_data(x, past_seq_len)
        # assert output_x.shape[0] == output_y.shape[0], "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1)
        return output_x[mask]

    def _get_y_pred_dt(self, input_df, past_seq_len):
        """
        :param dataframe:
        :return:
        """
        input_df = input_df.reset_index(drop=True)
        pre_pred_dt = input_df.loc[past_seq_len:, [self.dt_col]].copy()
        pre_pred_dt = pre_pred_dt.reset_index(drop=True)
        time_delta = pre_pred_dt.iloc[-1] - pre_pred_dt.iloc[-2]
        last_time = pre_pred_dt.iloc[-1] + time_delta
        last_df = pd.DataFrame({self.dt_col: last_time})
        self.y_pred_dt = pre_pred_dt.append(last_df, ignore_index=True)

    def _scale(self, data):
        """
        Scale the data
        :param data:
        :return:
        """
        np_scaled = self.scaler.transform(data)
        data_s = pd.DataFrame(np_scaled)
        return data_s

    def _rearrange_data(self, input_df):
        """
        change the input_df column order into [datetime, target, feature1, feature2, ...]
        :param input_df:
        :return:
        """
        cols = input_df.columns.tolist()
        new_cols = [self.dt_col, self.target_col] + [col for col in cols if col != self.dt_col and col != self.target_col]
        rearranged_data = input_df[new_cols].copy
        return rearranged_data

    def _generate_features(self, input_df):
        df = input_df.copy()
        df["id"] = df.index + 1

        es = ft.EntitySet(id="data")
        es = es.entity_from_dataframe(entity_id="time_seq",
                                      dataframe=df,
                                      index="id",
                                      time_index=self.dt_col)

        def is_awake(column):
            hour = column.dt.hour
            return (((hour >= 6) & (hour <= 23)) | (hour == 0)).astype(int)

        def is_busy_hours(column):
            hour = column.dt.hour
            return (((hour >= 7) & (hour <= 9)) | (hour >= 16) & (hour <= 19)).astype(int)

        IsAwake = make_trans_primitive(function=is_awake,
                                       input_types=[DatetimeTimeIndex],
                                       return_type=Numeric)
        IsBusyHours = make_trans_primitive(function=is_busy_hours,
                                           input_types=[DatetimeTimeIndex],
                                           return_type=Numeric)

        feature_matrix, feature_defs = ft.dfs(entityset=es,
                                              target_entity="time_seq",
                                              agg_primitives=["count"],
                                              trans_primitives=["month", "weekday", "day", "hour",
                                                                "is_weekend", IsAwake, IsBusyHours])
        return feature_matrix, feature_defs

    def _get_features(self, input_df, config):
        feature_matrix, feature_defs = self._generate_features(input_df)
        # self.write_generate_feature_list(feature_defs)
        feature_cols = np.asarray(config.get("selected_features"))
        target_cols = np.array([self.target_col])
        cols = np.concatenate([target_cols, feature_cols])
        target_feature_matrix = feature_matrix[cols]
        return target_feature_matrix.astype(float)

    def _get_optional_parameters(self):
        return set(["past_seq_len"])

    def _get_required_parameters(self):
        return set(["selected_features"])


# class DummyTimeSequenceFeatures(BaseFeatures):
#     """
#     A Dummy Feature Transformer that just load prepared data
#     use flag train=True or False in config to return train or test
#     """
#
#     def __init__(self, file_path):
#         """
#         the prepared data path saved by in numpy.savez
#         file contains 4 arrays: "x_train", "y_train", "x_test", "y_test"
#         :param file_path: the file_path of the npz
#         """
#         from zoo.automl.common.util import load_nytaxi_data
#         x_train, y_train, x_test, y_test = load_nytaxi_data(file_path)
#         self.train_data = (x_train, y_train)
#         self.test_data = (x_test, y_test)
#         self.is_train = False
#
#     def _get_data(self, train=True):
#         if train:
#             return self.train_data
#         else:
#             return self.test_data
#
#     def fit(self, input_df, **config):
#         """
#
#         :param input_df:
#         :param config:
#         :return:
#         """
#         self.is_train = True
#
#     def transform(self, input_df):
#         x, y = self._get_data(self.is_train)
#         if self.is_train is True:
#             self.is_train = False
#         return x, y
#
#     def _get_optional_parameters(self):
#         return set()
#
#     def _get_required_parameters(self):
#         return set()
#
#     def save(self, file_path, **config):
#         """
#         save nothing
#         :param file_path:
#         :param config:
#         :return:
#         """
#         pass
#
#     def restore(self, file_path, **config):
#         """
#         restore nothing
#         :param file_path:
#         :param config:
#         :return:
#         """
#         pass
