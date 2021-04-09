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

from pyspark import SparkContext
from pyspark.sql.functions import col, udf, array, broadcast, explode, struct, collect_list
from zoo.orca import OrcaContext
from zoo.friesian.feature.utils import generate_string_idx, fill_na, \
    fill_na_int, compute, log_with_clip, clip_min
import random
from pyspark.sql.types import ArrayType, IntegerType, Row, StructType, StructField
from zoo.common.utils import callZooFunc

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

    @staticmethod
    def _read_json(paths, cols):
        if not isinstance(paths, list):
            paths = [paths]
        spark = OrcaContext.get_spark_session()
        df = spark.read.json(paths)
        if cols:
            if isinstance(cols, list):
                df = df.select(*cols)
            elif isinstance(cols, str):
                df = df.select(cols)
            else:
                raise Exception("cols should be a column name or list of column names")
        return df

    def _clone(self, df):
        return Table(df)

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

    def drop(self, *cols):
        return self._clone(self.df.drop(*cols))

    def fillna(self, value, columns):
        if isinstance(value, int) and JAVA_INT_MIN <= value <= JAVA_INT_MAX:
            return self._clone(fill_na_int(self.df, value, columns))
        else:
            return self._clone(fill_na(self.df, value, columns))

    def clip(self, columns, min=0):
        if not isinstance(columns, list):
            columns = [columns]
        return self._clone(clip_min(self.df, columns, min))

    def log(self, columns, clipping=True):
        if not isinstance(columns, list):
            columns = [columns]
        return self._clone(log_with_clip(self.df, columns, clipping))

    # Merge column values as a list to a new col
    def merge_cols(self, columns, target):
        assert isinstance(columns, list)
        return self._clone(self.df.withColumn(target, array(columns)).drop(*columns))

    def rename(self, columns):
        assert isinstance(columns, dict), "columns should be a dictionary of {'old_name1': " \
                                          "'new_name1', 'old_name2': 'new_name2'}"
        new_df = self.df
        for old_name, new_name in columns.items():
            new_df = new_df.withColumnRenamed(old_name, new_name)
        return self._clone(new_df)

    def show(self, n=20, truncate=True):
        self.df.show(n, truncate)

    def write_parquet(self, path, mode="overwrite"):
        self.df.write.mode(mode).parquet(path)

class FeatureTable(Table):
    @classmethod
    def read_parquet(cls, paths):
        return cls(Table._read_parquet(paths))

    @classmethod
    def read_json(cls, paths, cols=None):
        return cls(Table._read_json(paths, cols))

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

    def gen_ind2ind(self, cols, indices):
        df = self.encode_string(cols, indices).df.select(*cols).distinct()
        return FeatureTable(df)

    def _clone(self, df):
        return FeatureTable(df)

    def gen_negtive_samples(self, item_size, item_col="item", label_col="label", neg_num=1):
        df = callZooFunc("float", "genNegSamples", self.df, item_size, item_col, label_col, neg_num)
        return FeatureTable(df)

    def gen_hist_seq(self, user_col, cols, sort_col='time', min_len=1, max_len=100):
        df = callZooFunc("float", "genHistSeq", self.df, user_col, cols, sort_col, min_len, max_len)
        return FeatureTable(df)

    def gen_neg_hist_seq(self, item_size, item2cat, item_history_col, neg_num):
        df = callZooFunc("float", "genNegHisSeq", self.df, item_size, item2cat, item_history_col,
                           neg_num)
        return  FeatureTable(df)

    def pad(self, padding_cols, seq_len=100):
        df = callZooFunc("float", "postPad", self.df, padding_cols, seq_len)
        return FeatureTable(df)

    def mask(self, mask_cols, seq_len=100):
        df = callZooFunc("float", "mask", self.df, mask_cols, seq_len)
        return FeatureTable(df)

    def gen_length(self, col_name):
        df = callZooFunc("float", "genLength", self.df, col_name)
        return FeatureTable(df)

    def mask_pad(self, padding_cols, mask_cols, seq_len=100):
        table = self.mask(mask_cols, seq_len)
        return table.pad(padding_cols, seq_len)

    def transform_python_udf(self, in_col, out_col, udf_func):
        df = self.df.withColumn(out_col, udf_func(col(in_col)))
        return FeatureTable(df)

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

    def _clone(self, df):
        return StringIndex(df, self.col_name)

    def write_parquet(self, path, mode="overwrite"):
        path = path + "/" + self.col_name + ".parquet"
        self.df.write.parquet(path, mode=mode)
