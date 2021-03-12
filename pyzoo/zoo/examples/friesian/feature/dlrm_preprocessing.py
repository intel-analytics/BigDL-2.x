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

from zoo.friesian.feature import FeatureTable

import os

from argparse import ArgumentParser
from time import time

from zoo.orca import init_orca_context, stop_orca_context

LABEL_COL = 0
INT_COLS = ["_c{}".format(i) for i in list(range(1, 14))]
CAT_COLS = ["_c{}".format(i) for i in list(range(14, 40))]


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--days', required=True)
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder')
    parser.add_argument('--model_size_file')
    parser.add_argument('--model_folder', required=True)
    parser.add_argument(
        '--write_mode',
        choices=['overwrite', 'errorifexists'],
        default='errorifexists')

    parser.add_argument('--frequency_limit')

    args = parser.parse_args()

    start, end = args.days.split('-')
    args.day_range = list(range(int(start), int(end) + 1))
    args.days = len(args.day_range)

    return args


if __name__ == '__main__':
    args = _parse_args()
    init_orca_context("local", cores=36, memory="50g")

    time_start = time()
    paths = [os.path.join(args.input_folder, 'day_%d.parquet' % i) for i in args.day_range]
    df = FeatureTable.read_parquet(paths)
    df_id_list = df.categorify(CAT_COLS, freq_limit=15)

    df_all_data = FeatureTable.read_parquet(paths[:-1])
    df_all_data = df_all_data.encode_string(CAT_COLS, df_id_list)\
        .fillna(0, INT_COLS + CAT_COLS).clip(INT_COLS).log(INT_COLS)
    df_all_data = df_all_data.merge(INT_COLS, "X_int").merge(CAT_COLS, "X_cat")
    df_all_data.compute()
    time_end = time()
    print("Train data loading and preprocessing time: ", time_end - time_start)
    df_all_data.df.show(5)
    print("Finished")
    stop_orca_context()
