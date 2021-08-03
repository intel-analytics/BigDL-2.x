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
from torch.utils.data import Dataset
import torch


def get_roll_start_idx(df, id_col, window_size):
    import itertools
    id_start_idxes = df.index[df[id_col] != df[id_col].shift(1)].tolist()
    roll_start_idx_iter = ((range(id_start_idxes[i], id_start_idxes[i+1] - window_size + 1))
                           for i in range(len(id_start_idxes) - 1))
    roll_start_idxes = list(itertools.chain.from_iterable(roll_start_idx_iter))
    return roll_start_idxes


class RollDataset(Dataset):
    def __init__(self, df, lookback, horizon, feature_col, target_col, id_col):
        """
        todo: add check for df
        pre-request for df:
        1. all the values are not nan
        2. if contains multiple ids, rows of same id should be consecutive
        3. dataframe has been ordered by timestamp for each id.
        """
        self.arr = df.loc[:, target_col + feature_col].to_numpy()
        max_horizon = horizon if isinstance(horizon, int) else max(horizon)
        window_size = lookback + max_horizon
        self.roll_start_idxes = get_roll_start_idx(df, id_col, window_size=window_size)
        self.lookback = lookback
        self.horizon = horizon
        self.target_num = len(target_col)

    def __len__(self):
        return len(self.roll_start_idxes)

    def __getitem__(self, idx):
        start_idx = self.roll_start_idxes[idx]
        x = self.arr[start_idx: start_idx + self.lookback]
        arr_target_only = self.arr[:, :self.target_num]
        if isinstance(self.horizon, int):
            y = arr_target_only[start_idx + self.lookback: start_idx + self.lookback + self.horizon]
        else:
            horizons = np.array(self.horizon)
            y = np.take(arr_target_only, horizons + start_idx + self.lookback - 1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y

