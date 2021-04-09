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
    def _read_json(sc, paths, cols):
        if sc:
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
    def read_json(cls, paths, cols=None, sc=None):
        return cls(Table._read_json(sc, paths, cols))

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

    def add_negtive_samples(self, item_size, item_col="item", label_col="label", neg_num=1):
        def gen_neg(item):
            result = []
            for i in range(neg_num):
                while True:
                    neg_item = random.randint(0, item_size - 1)
                    if neg_item == item:
                        continue
                    else:
                        result.append((neg_item, 0))
                        break
            result.append((item, 1))
            return result
        structure = StructType().add(item_col, IntegerType()).add(label_col, IntegerType())
        neg_udf = udf(gen_neg, ArrayType(structure))
        df = self.df.withColumn("item_label", neg_udf(col(item_col))) \
                .withColumn("item_label", explode(col("item_label"))).drop(item_col)
        df = df.withColumn(item_col, col("item_label.item")) \
                .withColumn(label_col, col("item_label.label")).drop("item_label")
        return FeatureTable(df)

    def transform_python_udf(self, in_col, out_col, udf_func):
        df = self.df.withColumn(out_col, udf_func(col(in_col)))
        return FeatureTable(df)

    def gen_his_seq(self, user_col, cols, sort_col='time', min_len=1, max_len=100):
        def gen_his(row_list):
            if sort_col:
                row_list.sort(key=lambda row: row[sort_col])

            def gen_his_one_row(rows, i):
                histories = [[row[col] for row in rows[:i]] for col in cols]
                return (*[rows[i][col] for col in cols], *histories)

            if len(row_list) <= min_len:
                return None
            if len(row_list) > max_len:
                row_list = row_list[-max_len:]
            result = [gen_his_one_row(row_list, i) for i in range(min_len, len(row_list))]
            return result

        structure = StructType()
        for c in cols:
            structure = structure.add(c, IntegerType())
        for c in cols:
            structure = structure.add(c + "_history", ArrayType(IntegerType()))
        schema = ArrayType(structure)
        gen_his_udf = udf(lambda x: gen_his(x), schema)
        df = self.df.groupBy(user_col) \
            .agg(collect_list(struct(*[col(name) for name in (cols + [sort_col])])).alias("his_collect"))
        df = df.withColumn("history", gen_his_udf(col("his_collect"))).dropna(subset=['history'])
        # df.collect()
        df = df.withColumn("history", explode(col("history"))) \
                .drop("his_collect")
        for c in cols:
            df = df.withColumn(c, col("history." + c)) \
                    .withColumn(c + "_history", col("history." + c + '_history'))
        df = df.drop("history")
        return FeatureTable(df)

    def gen_neg_hist_seq(self, item_size, item2cat, item_history_col='item_history', neg_num=5):
        sc = SparkContext.getOrCreate()

        def gen_neg_his(item_size, item_cat_map_b):
            def gen(row):
                item_cat_map = item_cat_map_b.value
                noclk_item_list = []
                noclk_cat_list = []
                for pos_item in row[item_history_col]:
                    noclk_tmp_item = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    # add 5 noclk item/cat
                    while True:
                        neg_item = random.randint(0, item_size - 1)
                        if neg_item == pos_item or neg_item not in item_cat_map:
                            continue
                        noclk_tmp_item.append(neg_item)
                        noclk_tmp_cat.append(item_cat_map[neg_item])
                        noclk_index += 1
                        if noclk_index >= neg_num:
                            break
                    noclk_item_list.append(noclk_tmp_item)
                    noclk_cat_list.append(noclk_tmp_cat)
                row = Row(*([row[col] for col in row.__fields__]
                            + [noclk_item_list, noclk_cat_list]))
                return row

            return gen

        item2cat_tuples = item2cat.df.rdd.map(lambda row: (row[0], row[1])).collect()
        item2cat_map = dict(item2cat_tuples)
        rdd = self.df.rdd.map(gen_neg_his(item_size, sc.broadcast(item2cat_map)))
        schema = StructType(self.df.schema.fields
                    + [StructField('noclk_item_list', ArrayType(ArrayType(IntegerType())))]
                    + [StructField('noclk_cat_list', ArrayType(ArrayType(IntegerType())))])
        df = rdd.toDF(schema)
        return FeatureTable(df)

    def pad(self, padding_cols, seq_len=100):
        def pad(seq):
            length = len(seq)
            if len(seq) < seq_len:
                if isinstance(seq[0], list):
                    return seq + [[0] * len(seq[0]) for i in range(seq_len - length)]
                else:
                    return seq + [0] * (seq_len - length)
            else:
                return seq[:seq_len]

        df = self.df
        for c in padding_cols:
            col_type = df.schema[c].dataType
            pad_udf = udf(pad, col_type)
            df = df.withColumn(c, pad_udf(col(c)))
        return FeatureTable(df)

    def mask(self, mask_cols, seq_len=100):
        def add_mask(seq):
            length = len(seq)
            if len(seq) < seq_len:
                return [1] * length + [0] * (seq_len - length)
            else:
                return [1] * seq_len

        mask_udf = udf(add_mask, ArrayType(IntegerType()))
        df = self.df
        for c in mask_cols:
            df = df.withColumn(c + "_mask", mask_udf(col(c)))
        return FeatureTable(df)

    def mask_pad(self, padding_cols, mask_cols, seq_len=100):
        table = self.mask(mask_cols, seq_len)
        return table.pad(padding_cols, seq_len)

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
