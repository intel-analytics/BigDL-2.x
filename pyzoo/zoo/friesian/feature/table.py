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
import copy
import os

from pyspark.sql.functions import col, array, broadcast
from zoo.orca import OrcaContext
from zoo.friesian.feature.utils import generate_string_idx, fill_na, \
    fill_na_int, compute, log_with_clip, clip_min

JAVA_INT_MIN = -2147483648
JAVA_INT_MAX = 2147483647


class Table:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def _read_parquet(paths):
        if not isinstance(paths, list):
            paths = [paths]
        spark = OrcaContext.get_spark_session()
        df = spark.read.parquet(*paths)
        return df

    def compute(self):
        compute(self.df)
        return self

    def to_spark_df(self):
        return self.df

    def count(self):
        cnt = self.df.count()
        return cnt

    def broadcast(self):
        self.df = broadcast(self.df)

    def fillna(self, value, columns):
        copy_tbl = copy.copy(self)
        if isinstance(value, int) and JAVA_INT_MIN <= value <= JAVA_INT_MAX:
            copy_tbl.df = fill_na_int(self.df, value, columns)
        else:
            copy_tbl.df = fill_na(self.df, value, columns)
        return copy_tbl

    def clip(self, columns, min=0):
        if not isinstance(columns, list):
            columns = [columns]
        copy_tbl = copy.copy(self)
        copy_tbl.df = clip_min(self.df, columns, min)
        return copy_tbl

    def log(self, columns, clipping=True):
        if not isinstance(columns, list):
            columns = [columns]
        copy_tbl = copy.copy(self)
        copy_tbl.df = log_with_clip(self.df, columns, clipping)
        return copy_tbl

    # Merge column values as a list to a new col
    def merge(self, merge_cols, merged_col_name):
        copy_tbl = copy.copy(self)
        assert isinstance(merge_cols, list)
        copy_tbl.df = self.df.withColumn(merged_col_name, array(merge_cols)).drop(*merge_cols)
        return copy_tbl

    # May not need the below methods if IndexTable only has two columns with the desired names:
    # col_name and id.
    def drop(self, *cols):
        copy_tbl = copy.copy(self)
        copy_tbl.df = self.df.drop(*cols)
        return copy_tbl

    def rename(self, columns):
        assert isinstance(columns, dict), "columns should be a dictionary of {'old_name1': " \
                                          "'new_name1', 'old_name2': 'new_name2'}"
        copy_tbl = copy.copy(self)
        new_df = self.df
        for old_name, new_name in columns.items():
            new_df = new_df.withColumnRenamed(old_name, new_name)
        copy_tbl.df = new_df
        return copy_tbl

    def show(self, n=20, truncate=True):
        self.df.show(n, truncate)


class FeatureTable(Table):
    @classmethod
    def read_parquet(cls, paths):
        return cls(Table._read_parquet(paths))

    def encode_string(self, columns, indices):
        if not isinstance(columns, list):
            columns = [columns]
        if not isinstance(indices, list):
            indices = [indices]
        assert len(columns) == len(indices)
        data_df = self.df
        for i in range(len(columns)):
            index_tbl = indices[i]
            col_name = columns[i]
            index_tbl.broadcast()
            data_df = data_df.join(index_tbl.df, col_name, how="left") \
                .drop(col_name).withColumnRenamed("id", col_name)
        return FeatureTable(data_df)

    def gen_string_idx(self, columns, freq_limit):
        df_id_list = generate_string_idx(self.df, columns, freq_limit)
        string_idx_list = list(map(lambda x: StringIndex(x[0], x[1]),
                                   zip(df_id_list, columns)))
        return string_idx_list


# Assume this table only has two columns: col_name and id
class StringIndex(Table):
    def __init__(self, df, col_name):
        super().__init__(df)
        cols = df.columns
        assert len(cols) >= 2, "StringIndex should have >= 2 columns: col_name, id and other " \
                               "columns"
        assert "id" in cols, "id should be a column of the DataFrame"
        assert col_name in cols, col_name + " should be a column of the DataFrame"
        self.col_name = col_name

    @classmethod
    def read_parquet(cls, paths, col_name=None):
        if not isinstance(paths, list):
            paths = [paths]
        if col_name is None and len(paths) >= 1:
            col_name = os.path.basename(paths[0]).split(".")[0]
        return cls(Table._read_parquet(paths), col_name)

    def write_parquet(self, path, mode="overwrite"):
        path = path + "/" + self.col_name + ".parquet"
        self.df.write.parquet(path, mode=mode)
