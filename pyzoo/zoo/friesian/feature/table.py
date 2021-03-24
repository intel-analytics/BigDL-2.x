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

from pyspark.sql.functions import col, array, broadcast
from pyspark.sql import DataFrame
from zoo.orca import OrcaContext
from zoo.friesian.feature.utils import assign_string_idx, fill_na, \
    fill_na_int, compute, log_with_clip, friesian_clip


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

    def count(self):
        cnt = self.df.count()
        print("Table size: ", cnt)
        return cnt

    def broadcast(self):
        return broadcast(self.df)

    def fillna(self, value, columns):
        return Table(fill_na(self.df, value, columns))

    def fillna_int(self, value, columns):
        return Table(fill_na_int(self.df, value, columns))

    def clip(self, columns, min=0):
        if not isinstance(columns, list):
            columns = [columns]
        return Table(friesian_clip(self.df, columns, min))

    def log(self, columns, clip=True):
        if not isinstance(columns, list):
            columns = [columns]
        return Table(log_with_clip(self.df, columns, clip))

    # Merge column values as a list to a new col
    def merge(self, columns, merged_col_name):
        assert isinstance(columns, list)
        return Table(self.df.withColumn(merged_col_name, array(columns)).drop(*columns))

    # May not need the below methods if IndexTable only has two columns with the desired names:
    # col_name and id.
    def drop(self, *cols):
        return Table(self.df.drop(*cols))

    def rename(self, columns):
        assert isinstance(columns, dict), "columns should be a dictionary of {'old_name1': " \
                                          "'new_name1', 'old_name2': 'new_name2'}"
        new_df = self.df
        for old_name, new_name in columns.items():
            new_df = new_df.withColumnRenamed(old_name, new_name)
        return Table(new_df)

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
            index_df = indices[i]
            col_name = columns[i]
            data_df = data_df.join(index_df.broadcast(), col_name, how="left") \
                .drop(col_name).withColumnRenamed("id", col_name)
        return FeatureTable(data_df)

    def gen_string_idx(self, columns, freq_limit):
        df_id_list = assign_string_idx(self.df, columns, freq_limit)
        string_idx_list = list(map(lambda x: StringIndex(x), df_id_list))
        return string_idx_list


# Assume this table only has two columns: col_name and id
class StringIndex(Table):
    def __init__(self, df):
        super().__init__(df)
        cols = df.columns
        if len(cols) != 2:
            raise ValueError("StringIndex should have only 2 columns, col_name and id")
        assert "id" in cols, "id should in the DataFrame"
        for c in cols:
            if c != "id":
                self.col_name = c
                break

    @classmethod
    def read_parquet(cls, paths):
        return cls(Table._read_parquet(paths))

    def write_parquet(self, path, mode="overwrite"):
        path = path + "/" + self.col_name + ".parquet"
        self.df.write.parquet(path, mode=mode)
