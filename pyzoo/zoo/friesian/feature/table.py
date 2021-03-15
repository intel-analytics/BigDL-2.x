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

from pyspark.sql.functions import col, udf, array, broadcast, log
from pyspark.sql import DataFrame
from zoo.orca import OrcaContext
from zoo.friesian.feature.utils import assign_string_idx, fill_na


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
        print("Table size: ", self.df.count())
        return self

    def broadcast(self):
        return broadcast(self.df)

    def fillna(self, value, columns):
        return Table(fill_na(self.df, value, columns))

    def clip(self, columns, min=0):
        df = self.df
        if not isinstance(columns, list):
            columns = [columns]
        clip_udf = udf(lambda data: max(min, data))
        for col_name in columns:
            df = df.withColumn(col_name, clip_udf(col(col_name)))
        return Table(df)

    # Assume all entries of columns are non-negative
    def log(self, columns):
        df = self.df
        if not isinstance(columns, list):
            columns = [columns]
        for col_name in columns:
            df = df.withColumn(col_name, log(col(col_name) + 1))
        return Table(df)

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

    def show(self, n=20, truncate=True, vertical=False):
        self.df.show(n, truncate, vertical)


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
        data_df = self.df
        frequency_dict = {}
        default_limit = None
        if freq_limit:
            frequency_limit = freq_limit.split(",")
            for fl in frequency_limit:
                frequency_pair = fl.split(":")
                if len(frequency_pair) == 1:
                    default_limit = int(frequency_pair[0])
                elif len(frequency_pair) == 2:
                    frequency_dict[frequency_pair[0]] = frequency_pair[1]

        df_count_filtered_list = []

        for col_n in columns:
            df_col = data_df.select(col_n).filter(col_n + ' is not null').groupBy(col_n).count()
            if col_n in frequency_dict:
                df_col = df_col.filter(col('count') >= int(frequency_dict[col_n]))
            elif default_limit:
                df_col = df_col.filter(col('count') >= default_limit)

            df_count_filtered_list.append(df_col)

        spark = OrcaContext.get_spark_session()
        df_id_list = assign_string_idx(df_count_filtered_list)
        string_idx_list = list(map(lambda x: StringIndex(DataFrame(x, spark)), df_id_list))
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
