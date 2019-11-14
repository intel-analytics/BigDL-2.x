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

from zoo.automl.feature.transformer import RollTransformer

# import numpy as np
# import pandas as pd


# def _rolling(data, seq_len):
#     """
#     roll data and drop missing
#     :param data: numpy array
#     :param seq_len: sequence length
#     :return: (rolled ndarray, mask ndarray).
#         The shape of rolled ndarray is (len(data) - seq_len + 1, seq_len)
#         The shape of mask ndarray is (len(data) - seq_len + 1)
#     """
#     result = []
#     mask = []
#     for i in range(len(data) - seq_len + 1):
#         result.append(data[i: i + seq_len])
#
#         if pd.isna(data[i: i + seq_len]).any(axis=None):
#             mask.append(0)
#         else:
#             mask.append(1)
#     return np.asarray(result), np.asarray(mask)


class BasicRollTransformer(RollTransformer):

    def transform(self, inputs, past_seq_len=2, is_train=False):
        """
        fit data with the input
        :param inputs: numpy array
        :param past_seq_len: the look back sequence length that need to unrolled
        :param is_train: indicate whether in training mode
        :return: (x, y).
            In training mode: (x, y)
                x rolls the inputs from begin to max(horizon) with a rolling length of past_seq_len.
                x shape : (number of samples, past_seq_len, target_col_num + feature_num)
                y rolls the inputs from (past sequence length + min(horizon)) to the last with a
                rolling length of len(horizon)
                y shape:  (number of samples, horizon length, target column num)
            In test mode:
                x rolls the inputs with a rolling length of past_seq_len.
                y is None
        """
        pass
