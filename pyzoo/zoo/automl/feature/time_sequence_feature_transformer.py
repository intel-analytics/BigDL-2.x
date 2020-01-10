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

# from zoo.automl.feature.abstract import BaseFeatureTransformer
from zoo.automl.feature.pre_roll_transformer import *
from zoo.automl.feature.roll_transformer import *
from zoo.automl.feature.post_roll_transformer import *
from zoo.automl.common.util import save_config, load_config

import numpy as np
import pandas as pd


class TimeSequenceFeatureTransformer():
    """
    TimeSequence feature engineering
    """

    def __init__(self, horizon=1,
                 dt_col="datetime",
                 target_col="value",
                 extra_features_col=None,
                 generated_feature_names=None,
                 drop_missing=True,
                 check_uniform=True):
        """
        :param horizon: range/int/list/tuple. The horizon of the forecasting task.
            If input a range or a tuple, the feature transformer will transform a range of data for
            forecasting. eg. horizon=range(1, 3) or horizon=(1,3).
            If input a list of int or a int, the feature transformer will transform the specified
            horizon(s) of data for forecasting. eg. horizon=[6, 12] or horizon=6.
        :param dt_col: str. The name of datetime column in the input data frame.
        :param target_col: a list of str/str.
        :param extra_features_col: a list of str/str.
        :param generated_feature_names: a subset of allowed generated feature names. The field will
         be passed into FeatureGenerator only when FeatureGenerator is enabled. If
         generated_feature_names is None and FeatureGenerator is enabled, it will generate all the
         candidate features.
        :param drop_missing: whether to drop missing values in the curve. If it is set to False,
            an error will be reported if missing values are found. If True, the missing values will
            be dropped and won't raise errors.
        :param check_uniform: whether to check if time sequence comes in uniform sampling intervals.
        """
        self._check_input(horizon, dt_col, target_col, extra_features_col, generated_feature_names,
                          drop_missing, check_uniform)
        self.dt_col = dt_col
        self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.drop_missing = drop_missing
        self.check_uniform = check_uniform
        self.generate_feature_names = generated_feature_names

        self.past_seq_len = None
        self.transformer_names = None
        self.pre_roll_transformers = None
        self.post_roll_transformers = None
        self.config_file_name = "feature_transformer.json"

    def _check_input(self, horizon, dt_col, target_col, extra_features_col, generated_feature_names,
                     drop_missing, check_uniform):
        if not (isinstance(horizon, int) or all(isinstance(x, int) for x in horizon)):
            raise ValueError("Input horizon should be an int or a list/tuple/range of integers")
        if isinstance(horizon, tuple):
            self.horizon = range(horizon[0], horizon[1])
        else:
            self.horizon = horizon

        if not isinstance(dt_col, str):
            raise ValueError("Input dt_col should be a str")
        if not isinstance(target_col, str) or all(isinstance(x, str) for x in target_col):
            raise ValueError("Input target_col should be a str or a list of strs")
        if extra_features_col and not (isinstance(extra_features_col, str) or
                                       all(isinstance(x, str) for x in extra_features_col)):
            raise ValueError("Input extra_features_col should be a str or a list of strs")

        allowed_features = FeatureGenerator.get_allowed_features()
        if generated_feature_names:
            if not isinstance(generated_feature_names, (list, tuple, str)):
                raise ValueError("Input generated_feature_names should be a str or "
                                 "a list/tuple of strs")
            if not set(generated_feature_names).issubset(allowed_features):
                raise ValueError("Input generated_feature_names should be a subset of",
                                 allowed_features)
        else:
            self.generate_feature_names = allowed_features

        if not isinstance(drop_missing, bool):
            raise ValueError("Input drop_missing is not a bool!")
        if not isinstance(check_uniform, bool):
            raise ValueError("Input check_uniform is not a bool!")

        if not self._is_single_value(horizon) or not self._is_single_value(target_col):
            raise ValueError("setting multi-values on horizon and target column"
                             "simultaneously is not supported yet!")

    def _is_single_value(self, value):
        if isinstance(value, int):
            return True
        elif len(value) == 1:
            return True
        else:
            return False

    def train_transform(self, input_df, **config):
        """
        Fit data and transform the raw data to features. This is used in training for hyper
        parameter searching.
        This method will refresh the parameters (e.g. min and max of the MinMaxScaler) if any
        :param input_df: The input time series data frame
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: tuple (x,y)
            x: 3-d numpy array.
             If any of the transformers that drop time series information is not selected, then x is
             in format (no. of samples, past sequence length,
             target column num + extra feature column num + generated feature num),
             Else, x is in format (no. of samples, feature_length, 1).
            y: y is 2-d numpy array in format (no. of samples, horizon_num/target_col_num/1)
        """
        self._get_configs(config)
        self._check_transformers()
        self._check_input_df(input_df, mode="train")
        self._instantiate_transformers()

        train_x, train_y = self._transform(input_df, pre_roll_is_train=True, roll_is_train=True,
                                           post_roll_is_train=True)
        return train_x, train_y

    def val_transform(self, input_df):
        """
        Transform data into features using the preset of configurations from train_transform
        :param input_df: The input time series data frame
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: tuple (x,y)
             x: 3-d numpy array.
             If any of the transformers that drop time series information is not selected, then x is
             in format (no. of samples, past sequence length,
             target column num + extra feature column num + generated feature num),
             Else, x is in format (no. of samples, feature_length, 1)
            y: y is 2-d numpy array in format (no. of samples, horizon_num/target_col_num/1)
        """
        self._check_input_df(input_df, mode="val")
        self._check_train_transformed()
        val_x, val_y = self._transform(input_df, pre_roll_is_train=False, roll_is_train=True,
                                       post_roll_is_train=False)
        return val_x, val_y

    def test_transform(self, input_df):
        """
        Transform data into features using the preset of configurations from train_transform
        :param input_df: The input time series data frame
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: x
            x: 3-d numpy array.
             If any of the transformers that drop time series information is not selected, then x is
             in format (no. of samples, past sequence length,
             target column num + extra feature column num + generated feature num),
             Else, x is in format (no. of samples, feature_length, 1)
        """
        self._check_input_df(input_df, mode="test")
        self._check_train_transformed()
        test_x, _ = self._transform(input_df, pre_roll_is_train=False, roll_is_train=False,
                                    post_roll_is_train=False)
        return test_x

    def val_post_processing(self, input_df, y_pred):
        """
        inverse transform y_pred and roll input_df for evaluation. Always used after val_transform.
        :param input_df:
        :param y_pred:
        :return: a tuple of two numpy arrays: (y_true, y_pred_untransformed)
            y_true is the rolled target value converted from input_df.
            y_pred_untransformed is the inverse transformed y_pred
        """
        self._check_train_transformed()
        y_pred_untransformed = self._inverse_transform(y_pred)
        y_true = self._roll_transform(input_df[[self.target_col]], is_train=True)
        return y_true, y_pred_untransformed

    def test_post_processing(self, input_df, y_pred):
        """
        inverse transform y_pred and output as a data frame (with datetime column) for prediction.
        Always used after test_transform.
        :param input_df:
        :param y_pred:
        :return: a data frame with date time column and untransformed y_pred with the name of
            target columns/. If there is only one target column and multi horizons, the column names
            will be {target_column_name}_{horizon_index}. Eg, [value_1, value_2] if target column
            name is value, and horizon ranges from 1 to 2.
        """
        self._check_train_transformed()
        y_pred_untransformed = self._inverse_transform(y_pred)
        pred_df = self._get_pred_df(input_df, y_pred_untransformed)
        return pred_df

    def save(self, file_path):
        """
        1. save initialization args
        2. save the intermediate state of transformers
        :param file_path:
        :return:
        """
        self._save_ft_configs(file_path)

        for transformer in self.pre_roll_transformers:
            if isinstance(transformer, PRE_ROLL_SAVE):
                transformer.save(file_path)
        for transformer in self.post_roll_transformers:
            if isinstance(transformer, POST_ROLL_SAVE):
                transformer.save(file_path)

    def _save_ft_configs(self, file_path):
        init_info = {"horizon": self.horizon,
                     "dt_col": self.dt_col,
                     "target_col": self.target_col,
                     "extra_features_col": self.extra_features_col,
                     "drop_missing": self.drop_missing,
                     "check_uniform": self.check_uniform,
                     "generate_feature_names": self.generate_feature_names,
                     }
        info_to_save = init_info.copy()
        searched_configs = {"past_seq_len": self.past_seq_len,
                            "transformers": self.transformer_names}
        info_to_save.update(searched_configs)
        config_file_path = os.path.join(file_path, self.config_file_name)
        save_config(config_file_path, info_to_save)

    def restore(self, file_path):
        """
        1. restore initialization args
        2. instantiate transformers according to config and set other searched parameters
         (eg. past_seq_len)
        3. restore transformers
        :param file_path:
        :return:
        """
        self._restore_ft_configs(file_path)
        for transformer in self.pre_roll_transformers:
            if isinstance(transformer, PRE_ROLL_SAVE):
                transformer.restore(file_path)
        for transformer in self.post_roll_transformers:
            if isinstance(transformer, POST_ROLL_SAVE):
                transformer.restore(file_path)

    def _restore_ft_configs(self, file_path):
        config_file_path = os.path.join(file_path, self.config_file_name)
        config = load_config(config_file_path)

        self.horizon = config["horizon"]
        self.dt_col = config["dt_col"]
        self.target_col = config["target_col"]
        self.extra_features_col = config["extra_features_col"]
        self.drop_missing = config["drop_missing"]
        self.check_uniform = config["check_uniform"]
        self.generate_feature_names = config["generate_feature_names"]

        self._get_configs(config)
        self._instantiate_transformers()

    def _transform(self, input_df, pre_roll_is_train, roll_is_train, post_roll_is_train):
        pre_roll_output = self._pre_roll_transform(input_df, pre_roll_is_train)
        roll_input = self._get_roll_input(pre_roll_output)
        roll_output = self._roll_transform(roll_input, roll_is_train)
        x, y = self._post_roll_transform(roll_output, post_roll_is_train)
        return x, y

    def _pre_roll_transform(self, input_df, is_train):
        """
        transform with the selected PreRollTransformers.
        1. set transform_cols for different PreRollTransformers
        2. transform with the selected PreRollTransforms. FeatureGenerator will output a ndarray
        while other PreRollTransformers take a data frame as an input and also output a data frame.
        If none of the PreRollTransformers is selected, it will just output the input_df.
        :param input_df:
        :param is_train:
        :return:
        """
        def _get_transform_cols(transformer):
            if not self.extra_features_col:
                transform_cols = self.target_col
            elif isinstance(transformer, TRANSFORM_TARGET_COL):
                transform_cols = self.target_col
            else:
                transform_cols = self.target_col + self.extra_features_col
            return transform_cols

        tmp_input_df = input_df.copy()
        for transformer in self.pre_roll_transformers:
            transform_cols = _get_transform_cols(transformer)
            tmp_input_df = transformer.transform(tmp_input_df, transform_cols=transform_cols,
                                                 is_train=is_train)
        output = tmp_input_df
        return output

    def _get_roll_input(self, pre_roll_output):
        """
        Since FeatureGenerator or any PreRollTransformers may not be selected, in which case,
        pre_roll_output is a data frame, we need to extract the valid columns and feed the
        RollTransformer with a ndarray.
        :param pre_roll_output:
        :return:
        """
        if isinstance(pre_roll_output, pd.DataFrame):
            if self.extra_features_col:
                valid_cols = self.target_col + self.extra_features_col
            else:
                valid_cols = self.target_col
            roll_input = pre_roll_output[valid_cols].copy()
            return roll_input
        else:
            return pre_roll_output

    def _roll_transform(self, roll_input, is_train):
        """
        transform with RollTransformer.
        :param roll_input:
        :param is_train:
        :return: roll_output which is a tuple of (x,y). y is None if roll_is_train is false
        """
        if self.past_seq_len is None:
            raise ValueError("Call transform before reading feature configs "
                             "(specially past_seq_len)!")
        roll_output = self.roll_transformer.transform(roll_input, past_seq_len=self.past_seq_len,
                                                      is_train=is_train)
        return roll_output

    def _post_roll_transform(self, post_roll_input, is_train):
        """
        transform with the selected PostRollTransformers.
        :param post_roll_input:
        :param is_train:
        :return: a tuple of (x, y). If y in post_roll_input is None, output y is also None.
        """
        tmp_input = post_roll_input
        for transformer in self.post_roll_transformers:
            tmp_input = transformer.transform(tmp_input, is_train=is_train)
        x, y = tmp_input
        return x, y

    def _inverse_transform(self, y_pred):
        """
        inverse transform y_pred
        :param y_pred: numpy array
        :return: numpy array. Inverse transformed result of y_pred, which is of the same shape of
            input y_pred
        """
        tmp = y_pred
        for transformer in reversed(self.post_roll_transformers):
            if isinstance(transformer, POST_ROLL_INVERSE):
                tmp = transformer.inverse_transform(tmp)
        for transformer in reversed(self.pre_roll_transformers):
            if isinstance(transformer, PRE_ROLL_INVERSE):
                tmp = transformer.inverse_transform(tmp)
        y_out = tmp
        return y_out

    def _get_pred_df(self, input_df, pred_array):
        """
        generate output data frame.
        1. Calculate prediction datetime by input_df and self.past_seq_len.
        2. generate output data frame based on the datetime and pred array
        :param input_df:
        :param pred_array:
        :return: a data frame
        """
        pass

    def _check_transformers(self):
        """
        1. check if all transformer name is allowed
        2. check pre_roll transformer rules
        3. check post_roll transformer rules
        4. check inter pre_roll and poster_roll transformer config rules
        :return:
        """
        allowed_transformer_names = PRE_ROLL_TRANSFORMER_NAMES.union(PRE_ROLL_TRANSFORMER_NAMES)
        diff = self.transformer_names.difference(allowed_transformer_names)
        if diff:
            raise ValueError("{} are not legal transformer names. Allowed transformer names are {}"
                             .format(diff, allowed_transformer_names))
        self._check_pre_roll_rules()
        self._check_post_roll_rules()
        self._check_inter_rules()

    def _check_pre_roll_rules(self):
        assert not ("min_max_norm" in self.transformer_names and "standard_norm" in self.transformer_names)

    def _check_post_roll_rules(self):
        pass

    def _check_inter_rules(self):
        sequence_model_transformers = set("feature_generator")
        mlp_transformers = set("auto_encoder", "aggreg")
        assert not (self.transformer_names & sequence_model_transformers
                    and self.transformer_names & mlp_transformers)

    def _instantiate_transformers(self):
        self._instantiate_pre_roll_transformers()
        self.roll_transformer = BasicRollTransformer()
        self._instantiate_post_roll_transformers()

    def _instantiate_pre_roll_transformers(self):
        pre_roll_transformer_names = self.transformer_names & PRE_ROLL_TRANSFORMER_NAMES
        if pre_roll_transformer_names:
            sorted_pre_roll_transformer_names = sorted(pre_roll_transformer_names,
                                                       key=lambda x: PRE_ROLL_ORDER[x])
            self.pre_roll_transformers = []
            for name in sorted_pre_roll_transformer_names:
                if name != "feature_generator":
                    self.pre_roll_transformers.append(PRE_ROLL_NAME2TRANSFORMER[name]())
                else:
                    self.pre_roll_transformers.append(
                        PRE_ROLL_NAME2TRANSFORMER[name](self.generate_feature_names))

    def _instantiate_post_roll_transformers(self):
        post_roll_transformer_names = self.transformer_names & POST_ROLL_TRANSFORMER_NAMES
        if post_roll_transformer_names:
            sorted_post_roll_transformer_names = sorted(post_roll_transformer_names,
                                                        key=lambda x: POST_ROLL_ORDER[x])
            self.post_roll_transformers = [POST_ROLL_NAME2TRANSFORMER[name]
                                           for name in sorted_post_roll_transformer_names]

    def _check_input_df(self, input_df, mode="train"):
        """
        1. check NaT in datetime
        2. check input datetime format. Requires to be datetime64
        3. if check_uniform, check if time sequence intervals are uniform
        4. if not drop_missing, report error for missing values
        5. check if input datetime is larger than current time.
            In that case, feature tools will generate NaN.
        6. check input data length.
            If mode is "train"/"val", input data length >= (past sequence length + max(horizon))
            else, input data length >= past sequence length
        :param input_df:
        :param mode:
        :return:
        """
        pass

    def _get_configs(self, config):
        self.past_seq_len = config.get('past_seq_len', 2)
        self.transformer_names = set(config.get('transformers',
                                                ["min_max_norm", "feature_generator"]))

    def _check_train_transformed(self):
        """
        used in val_transform and test_transform.
        :return:
        """
        assert self.pre_roll_transformers is not None
        assert self.post_roll_transformers is not None
        assert self.past_seq_len is not None
        assert self.transformer_names is not None

