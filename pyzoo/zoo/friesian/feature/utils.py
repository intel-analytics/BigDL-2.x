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

from zoo.common.utils import callZooFunc


def compute(df):
    return callZooFunc("float", "compute", df)


def log_with_clip(df, columns):
    return callZooFunc("float", "log", df, columns)


def assign_string_idx(df_list):
    return callZooFunc("float", "assignStringIdx", df_list)


def assign_string_idx2(df, columns, freq_limit):
    return callZooFunc("float", "assignStringIdx2", df, columns, freq_limit)


def fill_na(df, fill_val, columns):
    return callZooFunc("float", "fillNa", df, fill_val, columns)


def fill_na_int(df, fill_val, columns):
    return callZooFunc("float", "fillNaInt", df, fill_val, columns)


def read_parquet(paths):
    return callZooFunc("float", "readParquet", paths)


def dlrm_preprocess(paths, CAT_columns, INT_columns, freq_limit):
    return callZooFunc("float", "dlrmPreprocess", paths, CAT_columns, INT_columns, freq_limit)


def dlrm_preprocess_returndf(paths, CAT_columns, INT_columns, freq_limit):
    return callZooFunc("float", "dlrmPreprocessReturnDF", paths, CAT_columns, INT_columns,
                       freq_limit)


def dlrm_preprocess_rdd(paths, CAT_columns, INT_columns, freq_limit):
    return callZooFunc("float", "dlrmPreprocessRDD", paths, CAT_columns, INT_columns,
                       freq_limit)


def dlrm_preprocess_returndf_compute(paths, CAT_columns, INT_columns, freq_limit):
    return callZooFunc("float", "dlrmPreprocessReturnDFCompute", paths, CAT_columns, INT_columns,
                       freq_limit)
