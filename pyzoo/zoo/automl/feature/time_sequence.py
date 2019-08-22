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

    def __init__(self, future_seq_len=1,
                 dt_col="datetime",
                 target_col="value",
                 extra_features_col=None,
                 drop_missing=True):
        """
        Constructor.
        :param drop_missing: whether to drop missing values in the curve, if this is set to False,
                             an error will be reported if missing values are found. If True, will
                             drop the missing values and won't raise errors.
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

    def _fit_transform(self, input_df):
        """
        Fit data and transform the raw data to features. This is used in training for hyper
        parameter searching.
        This method will refresh the parameters (e.g. min and max of the MinMaxScaler) if any
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        self._check_input(input_df, mode="train")
        # print(input_df.shape)
        feature_data = self._get_features(input_df, self.config)
        self.scaler.fit(feature_data)
        data_n = self._scale(feature_data)
        assert np.mean(data_n[0]) < 1e-5
        (x, y) = self._roll_train(data_n,
                                  past_seq_len=self.past_seq_len,
                                  future_seq_len=self.future_seq_len)

        return x, y

    def fit_transform(self, input_df, **config):
        """
        Fit data and transform the raw data to features. This is used in training for hyper
        parameter searching.
        This method will refresh the parameters (e.g. min and max of the MinMaxScaler) if any
        :param input_df: The input time series data frame, it can be a list of data frame or just
         one dataframe
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        self.config = self._get_feat_config(**config)

        if isinstance(input_df, list):
            train_x_list = []
            train_y_list = []
            for df in input_df:
                x, y = self._fit_transform(df)
                train_x_list.append(x)
                train_y_list.append(y)
            train_x = np.concatenate(train_x_list, axis=0)
            train_y = np.concatenate(train_y_list, axis=0)
        else:
            train_x, train_y = self._fit_transform(input_df)
        return train_x, train_y

    def _transform(self, input_df, mode):
        """
        Transform data into features using the preset of configurations from fit_transform
        :param input_df: The input time series data frame.
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01  1.9         1                       2
         2019-01-02  2.3         0                       2
        :param mode: 'val'/'test'.
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        self._check_input(input_df, mode)
        # generate features
        feature_data = self._get_features(input_df, self.config)
        # select and standardize data
        data_n = self._scale(feature_data)
        if mode == 'val':
            (x, y) = self._roll_train(data_n,
                                      past_seq_len=self.past_seq_len,
                                      future_seq_len=self.future_seq_len)
            return x, y
        else:
            x = self._roll_test(data_n, past_seq_len=self.past_seq_len)
            return x, None

    def transform(self, input_df, is_train=True):
        """
        Transform data into features using the preset of configurations from fit_transform
        :param input_df: The input time series data frame, input_df can be a list of data frame or
                         one data frame.
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01  1.9         1                       2
         2019-01-02  2.3         0                       2
        :param is_train: If the input_df is for training.
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        if self.config is None or self.past_seq_len is None:
            raise Exception("Needs to call fit_transform or restore first before calling transform")
        mode = "val" if is_train else "test"
        if isinstance(input_df, list):
            output_x_list = []
            output_y_list = []
            for df in input_df:
                if mode == 'val':
                    x, y = self._transform(df, mode)
                    output_x_list.append(x)
                    output_y_list.append(y)
                else:
                    x, _ = self._transform(df, mode)
                    output_x_list.append(x)
            output_x = np.concatenate(output_x_list, axis=0)
            if output_y_list:
                output_y = np.concatenate(output_y_list, axis=0)
            else:
                output_y = None
        else:
            output_x, output_y = self._transform(input_df, mode)
        return output_x, output_y

    def _unscale(self, y):
        # for standard scalar
        value_mean = self.scaler.mean_[0]
        value_scale = self.scaler.scale_[0]
        y_unscale = y * value_scale + value_mean
        return y_unscale

    def _get_y_pred_df(self, y_pred_dt_df, y_pred_unscale):
        """
        get prediction data frame with datetime column and target column.
        :param input_df:
        :return : prediction data frame. If future_seq_len is 1, the output data frame columns are
            datetime | {target_col}. Otherwise, the output data frame columns are
            datetime | {target_col}_0 | {target_col}_1 | ...
        """
        y_pred_df = y_pred_dt_df
        if self.future_seq_len > 1:
            columns = ["{}_{}".format(self.target_col, i) for i in range(self.future_seq_len)]
            y_pred_df[columns] = pd.DataFrame(y_pred_unscale)
        else:
            y_pred_df[self.target_col] = y_pred_unscale
        return y_pred_df

    def post_processing(self, input_df, y_pred, is_train):
        """
        Used only in pipeline predict, after calling self.transform(input_df, is_train=False).
        Post_processing includes converting the predicted array into data frame and scalar inverse
        transform.
        :param input_df: a list of data frames or one data frame.
        :param y_pred: Model prediction result (ndarray).
        :param is_train: indicate the output is used to evaluation or prediction.
        :return:
         In validation mode (is_train=True), return the unscaled y_pred and rolled input_y.
         In test mode (is_train=False) return unscaled data frame(s) in the format of
          {datetime_col} | {target_col(s)}.
        """
        y_pred_unscale = self._unscale(y_pred)
        if is_train:
            # return unscaled y_pred (ndarray) and y (ndarray).
            if isinstance(input_df, list):
                y_unscale_list = []
                for df in input_df:
                    _, y_unscale = self._roll_train(df[[self.target_col]],
                                                    self.past_seq_len,
                                                    self.future_seq_len)
                    y_unscale_list.append(y_unscale)
                output_y_unscale = np.concatenate(y_unscale_list, axis=0)
            else:
                _, output_y_unscale = self._roll_train(input_df[[self.target_col]],
                                                       self.past_seq_len,
                                                       self.future_seq_len)
            return output_y_unscale, y_pred_unscale

        else:
            # return data frame or a list of data frames.
            if isinstance(input_df, list):
                y_pred_dt_df_list = self._get_y_pred_dt_df(input_df, self.past_seq_len)
                y_pred_df_list = []
                y_pred_st_loc = 0
                for y_pred_dt_df in y_pred_dt_df_list:
                    df = self._get_y_pred_df(y_pred_dt_df,
                                             y_pred_unscale[y_pred_st_loc:
                                                            y_pred_st_loc + len(y_pred_dt_df)])
                    y_pred_st_loc = y_pred_st_loc + len(y_pred_dt_df)
                    y_pred_df_list.append(df)
                assert y_pred_st_loc == len(y_pred_unscale)
                return y_pred_df_list
            else:
                y_pred_dt_df = self._get_y_pred_dt_df(input_df, self.past_seq_len)
                y_pred_df = self._get_y_pred_df(y_pred_dt_df, y_pred_unscale)
                return y_pred_df

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

    def restore(self, **config):
        """
        Restore variables from file
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
        if isinstance(input_df, list):
            feature_matrix, feature_defs = self._generate_features(input_df[0])
        else:
            feature_matrix, feature_defs = self._generate_features(input_df)
    # return [feat.generate_name() for feat in feature_defs if isinstance(feat, TransformFeature)]
        feature_list = []
        for feat in feature_defs:
            feature_name = feat.generate_name()
            # todo: need to change if more than one target cols are supported
            if isinstance(feat, TransformFeature) \
                    or (self.extra_features_col and feat in self.extra_features_col):
            # if feature_name != self.target_col:
                feature_list.append(feature_name)
        return feature_list

    def _get_feat_config(self, **config):
        """
        Get feature related arguments from global hyper parameter config and do necessary error
        checking
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
        self.past_seq_len = feat_config.get("past_seq_len", 1)
        return feat_config

    def _check_input(self, input_df, mode="train"):
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

        # check if the last datetime is large than current time.
        # In that case, feature tools generate NaN.
        last_datetime = dt.iloc[-1]
        current_time = np.datetime64('today', 's')
        if last_datetime > current_time:
            raise ValueError("Last date time is bigger than current time!")

        # check the length of input data is smaller than requested.
        if mode == "test":
            min_input_len = self.past_seq_len
            error_msg = "Length of {} data should be larger than " \
                        "the past sequence length selected by automl.\n" \
                        "{} data length: {}\n" \
                        "past sequence length selected: {}\n" \
                .format(mode, mode, len(input_df), self.past_seq_len)
        else:
            min_input_len = self.past_seq_len + self.future_seq_len
            error_msg = "Length of {} data should be larger than " \
                        "the sequence length you want to predict " \
                        "plus the past sequence length selected by automl.\n"\
                        "{} data length: {}\n"\
                        "predict sequence length: {}\n"\
                        "past sequence length selected: {}\n"\
                .format(mode, mode, len(input_df), self.future_seq_len, self.past_seq_len)
        if len(input_df) < min_input_len:
            raise ValueError(error_msg)

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
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the
            last dimension, the 1st col is the time index (data type needs to be numpy datetime type
            , e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future
            sequence length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence
            length = 1
        """
        x = dataframe[0:-future_seq_len].values
        y = dataframe.iloc[past_seq_len:, 0].values
        output_x, mask_x = self._roll_data(x, past_seq_len)
        output_y, mask_y = self._roll_data(y, future_seq_len)
        # assert output_x.shape[0] == output_y.shape[0],
        # "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1) & (mask_y == 1)
        return output_x[mask], output_y[mask]

    def _roll_test(self, dataframe, past_seq_len):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        roll_test: the whole dataframe is regarded as x.
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seq_len: the length of the past sequence
        :return: x
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the
            last dimension, the 1st col is the time index (data type needs to be numpy datetime type
            , e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
        """
        x = dataframe.values
        output_x, mask_x = self._roll_data(x, past_seq_len)
        # assert output_x.shape[0] == output_y.shape[0],
        # "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1)
        return output_x[mask]

    def __get_y_pred_dt_df(self, input_df, past_seq_len):
        """
        :param input_df: one data frame
        :return: a data frame with prediction datetime
        """
        input_df = input_df.reset_index(drop=True)
        pre_pred_dt_df = input_df.loc[past_seq_len:, [self.dt_col]].copy()
        pre_pred_dt_df = pre_pred_dt_df.reset_index(drop=True)
        time_delta = pre_pred_dt_df.iloc[-1] - pre_pred_dt_df.iloc[-2]
        last_time = pre_pred_dt_df.iloc[-1] + time_delta
        last_df = pd.DataFrame({self.dt_col: last_time})
        y_pred_dt_df = pre_pred_dt_df.append(last_df, ignore_index=True)
        return y_pred_dt_df

    def _get_y_pred_dt_df(self, input_df, past_seq_len):
        """
        :param input_df: a data frame or a list of data frame
        :param past_seq_len:
        :return:
        """
        if isinstance(input_df, list):
            y_pred_dt_df_list = []
            for df in input_df:
                y_pred_dt_df = self.__get_y_pred_dt_df(df, past_seq_len)
                y_pred_dt_df_list.append(y_pred_dt_df)
            return y_pred_dt_df_list
        else:
            return self.__get_y_pred_dt_df(input_df, past_seq_len)

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
        new_cols = [self.dt_col,
                    self.target_col] + [col for col in cols
                                        if col != self.dt_col and col != self.target_col]
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
        # we do not include target col in candidates.
        # the first column is designed to be the default position of target column.
        target_col = np.array([self.target_col])
        cols = np.concatenate([target_col, feature_cols])
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
