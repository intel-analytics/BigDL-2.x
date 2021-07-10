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

import shutil
import os.path
import pytest
import hashlib
import operator
from unittest import TestCase

from pyspark.sql.functions import col, concat, max, min, array
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, \
    DoubleType

from zoo.orca import OrcaContext
from zoo.friesian.feature import FeatureTable, StringIndex
from zoo.common.nncontext import *


class TestTable(TestCase):
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")

    def test_fillna_int(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filled_tbl = feature_tbl.fillna(5, ["col_2", "col_3"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_2 is null").count() != 0 and feature_tbl \
            .df.filter("col_3 is null").count() != 0, "feature_tbl should not be changed"
        assert filled_tbl.df.filter("col_2 == 5").count() == 1, "col_2 null values should be " \
                                                                "filled with 5"
        assert filled_tbl.df.filter("col_3 == 5").count() == 1, "col_3 null values should be " \
                                                                "filled with 5"
        filled_tbl = feature_tbl.fillna(5, None)
        assert filled_tbl.df.filter("col_2 == 5").count() == 1, "col_2 null values should be " \
                                                                "filled with 5"
        assert filled_tbl.df.filter("col_3 == 5").count() == 1, "col_3 null values should be " \
                                                                "filled with 5"
        with self.assertRaises(Exception) as context:
            feature_tbl.fillna(0, ["col_2", "col_3", "col_8"])
        self.assertTrue('do not exist in this Table' in str(context.exception))

    def test_fillna_double(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filled_tbl = feature_tbl.fillna(3.2, ["col_2", "col_3"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_2 is null").count() != 0 and feature_tbl \
            .df.filter("col_3 is null").count() != 0, "feature_tbl should not be changed"
        assert filled_tbl.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_3 is null").count() == 0, "col_3 null values should be " \
                                                                   "filled"
        filled_tbl = feature_tbl.fillna(5, ["col_2", "col_3"])
        assert filled_tbl.df.filter("col_2 == 5").count() == 1, "col_2 null values should be " \
                                                                "filled with 5"
        assert filled_tbl.df.filter("col_3 == 5").count() == 1, "col_3 null values should be " \
                                                                "filled with 5"

    def test_fillna_long(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filled_tbl = feature_tbl.fillna(3, ["col_1", "col_2", "col_3"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_2 is null").count() != 0 and feature_tbl \
            .df.filter("col_3 is null").count() != 0, "feature_tbl should not be changed"
        assert filled_tbl.df.filter("col_1 is null").count() == 0, "col_1 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_3 is null").count() == 0, "col_3 null values should be " \
                                                                   "filled"

    def test_fillna_string(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        with self.assertRaises(Exception) as context:
            feature_tbl.fillna(3.2, ["col_4", "col_5"])
        self.assertTrue('numeric does not match the type of column col_4' in str(context.exception))

        filled_tbl = feature_tbl.fillna("bb", ["col_4", "col_5"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert filled_tbl.df.filter("col_4 is null").count() == 0, "col_4 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_5 is null").count() == 0, "col_5 null values should be " \
                                                                   "filled"

    def test_filter_by_frequency(self):
        data = [("a", "b", 1),
                ("b", "a", 2),
                ("a", "bc", 3),
                ("c", "c", 2),
                ("b", "a", 2),
                ("ab", "c", 1),
                ("c", "b", 1),
                ("a", "b", 1)]
        schema = StructType([StructField("A", StringType(), True),
                             StructField("B", StringType(), True),
                             StructField("C", IntegerType(), True)])
        spark = OrcaContext.get_spark_session()
        df = spark.createDataFrame(data, schema)
        tbl = FeatureTable(df).filter_by_frequency(["A", "B", "C"])
        assert tbl.to_spark_df().count() == 2, "the count of frequency >=2 should be 2"

    def test_hash_encode(self):
        spark = OrcaContext.get_spark_session()
        data = [("a", "b", 1),
                ("b", "a", 2),
                ("a", "c", 3),
                ("c", "c", 2),
                ("b", "a", 1),
                ("a", "d", 1)]
        schema = StructType([StructField("A", StringType(), True),
                             StructField("B", StringType(), True),
                             StructField("C", IntegerType(), True)])
        df = spark.createDataFrame(data, schema)
        tbl = FeatureTable(df)
        hash_str = lambda x: hashlib.md5(str(x).encode('utf-8', 'strict')).hexdigest()
        hash_int = lambda x: int(hash_str(x), 16) % 100
        hash_value = []
        for row in df.collect():
            hash_value.append(hash_int(row[0]))
        tbl_hash = []
        for record in tbl.hash_encode(["A"], 100).to_spark_df().collect():
            tbl_hash.append(int(record[0]))
        assert(operator.eq(hash_value, tbl_hash)), "the hash encoded value should be equal"

    def test_cross_hash_encode(self):
        spark = OrcaContext.get_spark_session()
        data = [("a", "b", "c", 1),
                ("b", "a", "d", 2),
                ("a", "c", "e", 3),
                ("c", "c", "c", 2),
                ("b", "a", "d", 1),
                ("a", "d", "e", 1)]
        schema = StructType([StructField("A", StringType(), True),
                             StructField("B", StringType(), True),
                             StructField("C", StringType(), True),
                             StructField("D", IntegerType(), True)])
        df = spark.createDataFrame(data, schema)
        cross_hash_df = df.withColumn("A_B_C", concat("A", "B", "C"))
        tbl = FeatureTable(df)
        cross_hash_str = lambda x: hashlib.md5(str(x).encode('utf-8', 'strict')).hexdigest()
        cross_hash_int = lambda x: int(cross_hash_str(x), 16) % 100
        cross_hash_value = []
        for row in cross_hash_df.collect():
            cross_hash_value.append(cross_hash_int(row[4]))
        tbl_cross_hash = []
        for record in tbl.cross_hash_encode(["A", "B", "C"], 100).to_spark_df().collect():
            tbl_cross_hash.append(int(record[4]))
        assert(operator.eq(cross_hash_value, tbl_cross_hash)), "the crossed hash encoded value" \
                                                               "should be equal"

    def test_gen_string_idx(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit=1)
        assert string_idx_list[0].size() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].size() == 2, "col_5 should have 2 indices"
        with tempfile.TemporaryDirectory() as local_path:
            for str_idx in string_idx_list:
                str_idx.write_parquet(local_path)
                str_idx_log = str_idx.log(["id"])
                assert str_idx.df.filter("id == 1").count() == 1, "id in str_idx should = 1"
                assert str_idx_log.df.filter("id == 1").count() == 0, "id in str_idx_log should " \
                                                                      "!= 1"
            assert os.path.isdir(local_path + "/col_4.parquet")
            assert os.path.isdir(local_path + "/col_5.parquet")
            new_col_4_idx = StringIndex.read_parquet(local_path + "/col_4.parquet")
            assert "col_4" in new_col_4_idx.df.columns, "col_4 should be a column of new_col_4_idx"
            with self.assertRaises(Exception) as context:
                StringIndex.read_parquet(local_path + "/col_5.parquet", "col_4")
            self.assertTrue('col_4 should be a column of the DataFrame' in str(context.exception))

    def test_gen_string_idx_dict(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit={"col_4": 1,
                                                                                     "col_5": 3})
        with self.assertRaises(Exception) as context:
            feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit="col_4:1,col_5:3")
        self.assertTrue('freq_limit only supports int, dict or None, but get str' in str(
            context.exception))
        assert string_idx_list[0].size() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].size() == 1, "col_5 should have 1 indices"

    def test_gen_string_idx_none(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"], freq_limit=None)
        assert string_idx_list[0].size() == 3, "col_4 should have 3 indices"
        assert string_idx_list[1].size() == 2, "col_5 should have 2 indices"

    def test_clip(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        clip_tbl = feature_tbl.clip(["col_1", "col_2", "col_3"], min=2, max=None)
        assert isinstance(clip_tbl, FeatureTable), "clip_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 < 2").count() != 0 and feature_tbl \
            .df.filter("col_2 < 2").count() != 0, "feature_tbl should not be changed"
        assert clip_tbl.df.filter("col_1 < 2").count() == 0, "col_1 should >= 2"
        assert clip_tbl.df.filter("col_2 < 2").count() == 0, "col_2 should >= 2"
        assert clip_tbl.df.filter("col_3 < 2").count() == 0, "col_3 should >= 2"
        with self.assertRaises(Exception) as context:
            feature_tbl.clip(None, 2)
        self.assertTrue('columns should be str or a list of str, but got None.'
                        in str(context.exception))

        feature_tbl = FeatureTable.read_parquet(file_path)
        clip_tbl = feature_tbl.clip(["col_1", "col_2", "col_3"], min=None, max=1)
        assert isinstance(clip_tbl, FeatureTable), "clip_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 > 1").count() != 0 and feature_tbl \
            .df.filter("col_2 > 1").count() != 0, "feature_tbl should not be changed"
        assert clip_tbl.df.filter("col_1 > 1").count() == 0, "col_1 should <= 1"
        assert clip_tbl.df.filter("col_2 > 1").count() == 0, "col_2 should <= 1"
        assert clip_tbl.df.filter("col_3 > 1").count() == 0, "col_3 should <= 1"

        feature_tbl = FeatureTable.read_parquet(file_path)
        clip_tbl = feature_tbl.clip(["col_1", "col_2", "col_3"], min=0, max=1)
        assert isinstance(clip_tbl, FeatureTable), "clip_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 > 1 or col_1 < 0").count() != 0 and feature_tbl \
            .df.filter("col_2 > 1 or col_2 < 0").count() != 0, "feature_tbl should not be changed"
        assert clip_tbl.df.filter("col_1 < 0").count() == 0, "col_1 should >= 0"
        assert clip_tbl.df.filter("col_2 > 1").count() == 0, "col_2 should <= 1"
        assert clip_tbl.df.filter("col_3 < 0 or col_3 > 1").count() == 0, "col_3 should >=0 " \
                                                                          "and <= 1"

    def test_dropna(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        dropped_tbl = feature_tbl.dropna(["col_1", "col_4"])
        assert isinstance(dropped_tbl, FeatureTable), "dropped_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 is null").count() != 0 and feature_tbl\
            .df.filter("col_4 is null").count() != 0, "feature_tbl should not be changed"
        assert dropped_tbl.df.filter("col_1 is null").count() == 0, "col_1 null values should " \
                                                                    "be dropped"
        assert dropped_tbl.df.filter("col_4 is null").count() == 0, "col_4 null values should " \
                                                                    "be dropped"
        assert 0 < dropped_tbl.df.count() < feature_tbl.df.count(), "the number of rows should " \
                                                                    "be decreased"

        dropped_tbl = feature_tbl.dropna(["col_1", "col_4"], how="all")
        assert dropped_tbl.df.filter("col_1 is null and col_4 is null").count() == 0, \
            "col_1 and col_4 should not both have null values"
        dropped_tbl = feature_tbl.dropna(["col_2", "col_4"], how="all")
        assert dropped_tbl.df.filter("col_2 is null").count() > 0, \
            "col_2 should still have null values after dropna with how=all"

        dropped_tbl = feature_tbl.dropna(["col_2", "col_3", "col_5"], thresh=2)
        assert dropped_tbl.df.filter("col_2 is null").count() > 0, \
            "col_2 should still have null values after dropna with thresh=2"
        assert dropped_tbl.df.filter("col_3 is null and col_5 is null").count() == 0, \
            "col_3 and col_5 should not both have null values"

    def test_fill_median(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        with self.assertRaises(Exception) as context:
            feature_tbl.fill_median(["col_4", "col_5"])
        self.assertTrue('col_4 with data type StringType is not supported' in
                        str(context.exception))

        filled_tbl = feature_tbl.fill_median(["col_1", "col_2"])
        assert isinstance(filled_tbl, FeatureTable), "filled_tbl should be a FeatureTable"
        assert filled_tbl.df.filter("col_1 is null").count() == 0, "col_1 null values should be " \
                                                                   "filled"
        assert filled_tbl.df.filter("col_2 is null").count() == 0, "col_2 null values should be " \
                                                                   "filled"

    def test_filter(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        filtered_tbl = feature_tbl.filter(feature_tbl.col_1 == 1)
        assert filtered_tbl.size() == 3, "Only 3 out of 5 rows has value 1 for col_1"
        filtered_tbl2 = feature_tbl.filter(
            (feature_tbl.col("col_1") == 1) & (feature_tbl.col_2 == 1))
        assert filtered_tbl2.size() == 1, "Only 1 out of 5 rows has value 1 for col_1 and col_2"

    def test_rename(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        name_dict = {"col_1": "new_col1", "col_4": "new_col4"}
        rename_tbl = feature_tbl.rename(name_dict)
        cols = rename_tbl.df.columns
        assert isinstance(rename_tbl, FeatureTable), "rename_tbl should be a FeatureTable"
        assert "col_1" in feature_tbl.df.columns, "feature_tbl should not be changed"
        assert "new_col1" in cols, "new_col1 should be a column of the renamed tbl."
        assert "new_col4" in cols, "new_col4 should be a column of the renamed tbl."

    def test_log(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        log_tbl = feature_tbl.log(["col_1", "col_2", "col_3"])
        assert isinstance(log_tbl, FeatureTable), "log_tbl should be a FeatureTable"
        assert feature_tbl.df.filter("col_1 == 1").count() != 0 and feature_tbl \
            .df.filter("col_2 == 1").count() != 0, "feature_tbl should not be changed"
        assert log_tbl.df.filter("col_1 == 1").count() == 0, "col_1 should != 1"
        assert log_tbl.df.filter("col_2 == 1").count() == 0, "col_2 should != 1"
        assert log_tbl.df.filter("col_3 == 1").count() == 0, "col_3 should != 1"

    def test_merge(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        merged_tbl = feature_tbl.merge_cols(["col_1", "col_2", "col_3"], "int_cols")
        assert "col_1" not in merged_tbl.df.columns, "col_1 shouldn't be a column of merged_tbl"
        assert "int_cols" in merged_tbl.df.columns, "int_cols should be a column of merged_tbl"
        assert "col_1" in feature_tbl.df.columns, "col_1 should be a column of feature_tbl"

    def test_norm(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path).fillna(0, ["col_2", "col_3"])
        normalized_tbl, min_max = feature_tbl.min_max_scale(["col_2"])
        max_value = normalized_tbl.df.select("col_2") \
            .agg(max(col("col_2")).alias("max")) \
            .rdd.map(lambda row: row['max']).collect()[0]
        min_value = normalized_tbl.df.select("col_2") \
            .agg(min(col("col_2")).alias("min")) \
            .rdd.map(lambda row: row['min']).collect()[0]

        assert max_value <= 1, "col_2 shouldn't be more than 1 after normalization"
        assert min_value >= 0, "col_2 shouldn't be less than 0 after normalization"

        tbl2 = FeatureTable(feature_tbl.df.withColumn("col2-col3", array(["col_2", "col_3"])))
        normalized_tbl2, min_max = tbl2.min_max_scale(["col_2", "col2-col3"])
        normalized_tbl2.compute()

    def test_cross(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path).fillna(0, ["col_2", "col_3"])
        crossed_tbl = feature_tbl.cross_columns([["col_2", "col_3"]], [100])
        assert "col_2_col_3" in crossed_tbl.df.columns, "crossed column is not created"
        max_value = crossed_tbl.df.select("col_2_col_3") \
            .agg(max(col("col_2_col_3")).alias("max")) \
            .rdd.map(lambda row: row['max']).collect()[0]
        min_value = crossed_tbl.df.select("col_2_col_3") \
            .agg(min(col("col_2_col_3")).alias("min")) \
            .rdd.map(lambda row: row['min']).collect()[0]

        assert max_value <= 100, "cross value shouldn't be more than 100 after cross"
        assert min_value > 0, "cross value shouldn't be less than 0 after cross"

    def test_add_negative_items(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", 1, "2019-07-01 12:01:19.000"),
                ("jack", 2, "2019-08-01 12:01:19.000"),
                ("jack", 3, "2019-09-01 12:01:19.000"),
                ("alice", 4, "2019-09-01 12:01:19.000"),
                ("alice", 5, "2019-10-01 12:01:19.000"),
                ("alice", 6, "2019-11-01 12:01:19.000")]
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("item", IntegerType(), True),
            StructField("time", StringType(), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)
        tbl = FeatureTable(df).add_negative_samples(10)
        dft = tbl.df
        assert tbl.size() == 12
        assert dft.filter("label == 1").count() == 6
        assert dft.filter("label == 0").count() == 6

    def test_add_hist_seq(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", 1, "2019-07-01 12:01:19.000"),
                ("jack", 2, "2019-08-01 12:01:19.000"),
                ("jack", 3, "2019-09-01 12:01:19.000"),
                ("jack", 4, "2019-07-02 12:01:19.000"),
                ("jack", 5, "2019-08-03 12:01:19.000"),
                ("jack", 6, "2019-07-04 12:01:19.000"),
                ("jack", 7, "2019-08-05 12:01:19.000"),
                ("alice", 4, "2019-09-01 12:01:19.000"),
                ("alice", 5, "2019-10-01 12:01:19.000"),
                ("alice", 6, "2019-11-01 12:01:19.000")]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("item", IntegerType(), True),
                             StructField("time", StringType(), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.withColumn("ts", col("time").cast("timestamp").cast("long"))
        tbl = FeatureTable(df.select("name", "item", "ts")) \
            .add_hist_seq(["item"], "name", "ts", 1, 4)
        assert tbl.size() == 8
        assert tbl.df.filter(col("name") == "alice").count() == 2
        assert tbl.df.filter("name like '%jack'").count() == 6
        assert "item_hist_seq" in tbl.df.columns

    def test_gen_neg_hist_seq(self):
        spark = OrcaContext.get_spark_session()
        sc = OrcaContext.get_spark_context()
        data = [
            ("jack", [1, 2, 3, 4, 5]),
            ("alice", [4, 5, 6, 7, 8]),
            ("rose", [1, 2])]
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("item_hist_seq", ArrayType(IntegerType()), True)])
        df = spark.createDataFrame(data, schema)
        df2 = sc \
            .parallelize([(1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 2), (8, 2), (9, 2)]) \
            .toDF(["item", "category"]).withColumn("item", col("item").cast("Integer")) \
            .withColumn("category", col("category").cast("Integer"))
        tbl = FeatureTable(df)
        tbl = tbl.add_neg_hist_seq(9, "item_hist_seq", 4)
        assert tbl.df.select("neg_item_hist_seq").count() == 3

    def test_add_value_features(self):
        spark = OrcaContext.get_spark_session()
        sc = OrcaContext.get_spark_context()
        data = [
            ("jack", [1, 2, 3, 4, 5]),
            ("alice", [4, 5, 6, 7, 8]),
            ("rose", [1, 2])]
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("item_hist_seq", ArrayType(IntegerType()), True)])

        df = spark.createDataFrame(data, schema)
        df.filter("name like '%alice%'").show()

        df2 = sc \
            .parallelize([(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (8, 2), (9, 2)]) \
            .toDF(["item", "category"]).withColumn("item", col("item").cast("Integer")) \
            .withColumn("category", col("category").cast("Integer"))
        tbl = FeatureTable(df)
        tbl2 = tbl.add_neg_hist_seq(9, "item_hist_seq", 4)
        tbl3 = tbl2.add_value_features(["item_hist_seq", "neg_item_hist_seq"],
                                       FeatureTable(df2), "item", "category")
        assert tbl3.df.select("category_hist_seq").count() == 3
        assert tbl3.df.select("neg_category_hist_seq").count() == 3
        assert tbl3.df.filter("name like '%alice%'").select("neg_category_hist_seq").count() == 1
        assert tbl3.df.filter("name == 'rose'").select("neg_category_hist_seq").count() == 1

    def test_pad(self):
        spark = OrcaContext.get_spark_session()
        data = [
            ("jack", [1, 2, 3, 4, 5], [[1, 2, 3], [1, 2, 3]]),
            ("alice", [4, 5, 6, 7, 8], [[1, 2, 3], [1, 2, 3]]),
            ("rose", [1, 2], [[1, 2, 3]])]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("list", ArrayType(IntegerType()), True),
                             StructField("matrix", ArrayType(ArrayType(IntegerType())))])
        df = spark.createDataFrame(data, schema)
        tbl1 = FeatureTable(df).pad(["list", "matrix"], seq_len=4)
        dft1 = tbl1.df
        tbl2 = FeatureTable(df).pad(cols=["list", "matrix"], mask_cols=["list"], seq_len=4)
        assert dft1.filter("size(matrix) = 4").count() == 3
        assert dft1.filter("size(list) = 4").count() == 3
        assert tbl2.df.filter("size(list_mask) = 4").count() == 3
        assert tbl2.df.filter("size(list_mask) = 2").count() == 0
        assert "list_mask" in tbl2.df.columns

    def test_median(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        with self.assertRaises(Exception) as context:
            feature_tbl.median(["col_4", "col_5"])
        self.assertTrue('col_4 with data type StringType is not supported' in
                        str(context.exception))

        median_tbl = feature_tbl.median(["col_1", "col_2", "col_3"])
        assert isinstance(median_tbl, FeatureTable), "median_tbl should be a FeatureTable"
        assert median_tbl.df.count() == 3, "the number of rows of median_tbl should be equal to " \
                                           "the number of specified columns"
        assert median_tbl.df.filter("column == 'col_1'").count() == 1, "col_1 should exist in " \
                                                                       "'column' of median_tbl"
        assert median_tbl.df.filter("column == 'col_2'").filter("median == 1.0").count() == 1, \
            "the median of col_2 should be 1.0"

    def test_cast(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", "123", 14, 8),
                ("alice", "34", 25, 9),
                ("rose", "25344", 23, 10)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("a", StringType(), True),
                             StructField("b", IntegerType(), True),
                             StructField("c", IntegerType(), True)])
        df = spark.createDataFrame(data, schema)
        tbl = FeatureTable(df)
        tbl = tbl.cast("a", "int")
        assert dict(tbl.df.dtypes)['a'] == "int", "column a should be now be cast to integer type"
        tbl = tbl.cast("a", "float")
        assert dict(tbl.df.dtypes)['a'] == "float", "column a should be now be cast to float type"
        tbl = tbl.cast(["b", "c"], "double")
        assert dict(tbl.df.dtypes)['b'] == dict(tbl.df.dtypes)['c'] == "double", \
            "column b and c should be now be cast to double type"
        tbl = tbl.cast(None, "float")
        assert dict(tbl.df.dtypes)['name'] == dict(tbl.df.dtypes)['a'] == dict(tbl.df.dtypes)['b'] \
            == dict(tbl.df.dtypes)['c'] == "float", \
            "all the columns should now be cast to float type"
        with self.assertRaises(Exception) as context:
            tbl = tbl.cast("a", "notvalid")
        self.assertTrue(
            "type should be string, boolean, int, long, short, float, double."
            in str(context.exception))

    def test_select(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        select_tbl = feature_tbl.select("col_1", "col_2")
        assert "col_1" in select_tbl.df.columns, "col_1 shoul be selected"
        assert "col_2" in select_tbl.df.columns, "col_2 shoud be selected"
        assert "col_3" not in select_tbl.df.columns, "col_3 shoud not be selected"
        assert feature_tbl.size() == select_tbl.size(), \
            "the selected table should have the same rows"
        with self.assertRaises(Exception) as context:
            feature_tbl.select()
        self.assertTrue("cols should be str or a list of str, but got None."
                        in str(context.exception))

    def test_create_from_dict(self):
        indices = {'a': 1, 'b': 2, 'c': 3}
        col_name = 'letter'
        tbl = StringIndex.from_dict(indices, col_name)
        assert 'id' in tbl.df.columns, "id should be one column in the stringindex"
        assert 'letter' in tbl.df.columns, "letter should be one column in the stringindex"
        assert tbl.size() == 3, "the StringIndex should have three rows"
        with self.assertRaises(Exception) as context:
            StringIndex.from_dict(indices, None)
        self.assertTrue("col_name should be str, but get None"
                        in str(context.exception))
        with self.assertRaises(Exception) as context:
            StringIndex.from_dict(indices, 12)
        self.assertTrue("col_name should be str, but get int"
                        in str(context.exception))
        with self.assertRaises(Exception) as context:
            StringIndex.from_dict([indices], col_name)
        self.assertTrue("indices should be dict, but get list"
                        in str(context.exception))

    def test_encode_string_from_dict(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", "123", 14, 8),
                ("alice", "34", 25, 9),
                ("rose", "25344", 23, 10)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", StringType(), True),
                             StructField("age", IntegerType(), True),
                             StructField("height", IntegerType(), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema))
        columns = ["name", "num"]
        indices = []
        indices.append({"jack": 1, "alice": 2, "rose": 3})
        indices.append({"123": 3, "34": 1, "25344": 2})
        tbl = tbl.encode_string(columns, indices)
        assert 'name' in tbl.df.columns, "name should be still in the columns"
        assert 'num' in tbl.df.columns, "num should be still in the columns"
        assert tbl.df.where(tbl.df.age == 14).select("name").collect()[0]["name"] == 1, \
            "the first row of name should be 1"
        assert tbl.df.where(tbl.df.height == 10).select("num").collect()[0]["num"] == 2, \
            "the third row of num should be 2"

    def test_columns(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        col_names = feature_tbl.columns
        assert isinstance(col_names, list), "col_names should be a list of strings"
        assert col_names == ["col_1", "col_2", "col_3", "col_4", "col_5"], \
            "column names are incorrenct"

    def test_get_stats(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", "123", 14, 8.5),
                ("alice", "34", 25, 9.7),
                ("rose", "25344", 23, 10.0)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", StringType(), True),
                             StructField("age", IntegerType(), True),
                             StructField("height", DoubleType(), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema))
        columns = ["age", "height"]
        # test str
        statistics = tbl.get_stats(columns, "min")
        assert len(statistics) == 2, "the dict should contain two statistics"
        assert statistics["age"] == 14, "the min value of age is not correct"
        assert statistics["height"] == 8.5, "the min value of height is not correct"
        columns = ["age", "height"]
        # test dict
        statistics = tbl.get_stats(columns, {"age": "max", "height": "avg"})
        assert len(statistics) == 2, "the dict should contain two statistics"
        assert statistics["age"] == 25, "the max value of age is not correct"
        assert statistics["height"] == 9.4, "the avg value of height is not correct"
        # test list
        statistics = tbl.get_stats(columns, ["min", "max"])
        assert len(statistics) == 2, "the dict should contain two statistics"
        assert statistics["age"][0] == 14, "the min value of age is not correct"
        assert statistics["age"][1] == 25, "the max value of age is not correct"
        assert statistics["height"][0] == 8.5, "the min value of height is not correct"
        assert statistics["height"][1] == 10.0, "the max value of height is not correct"
        # test dict of list
        statistics = tbl.get_stats(columns, {"age": ["min", "max"], "height": ["min", "avg"]})
        assert len(statistics) == 2, "the dict should contain two statistics"
        assert statistics["age"][0] == 14, "the min value of age is not correct"
        assert statistics["age"][1] == 25, "the max value of age is not correct"
        assert statistics["height"][0] == 8.5, "the min value of height is not correct"
        assert statistics["height"][1] == 9.4, "the max value of height is not correct"
        statistics = tbl.get_stats(None, "min")
        assert len(statistics) == 2, "the dict should contain two statistics"
        assert statistics["age"] == 14, "the min value of age is not correct"
        assert statistics["height"] == 8.5, "the min value of height is not correct"

    def test_min(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", "123", 14, 8.5),
                ("alice", "34", 25, 9.7),
                ("rose", "25344", 23, 10.0)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", StringType(), True),
                             StructField("age", IntegerType(), True),
                             StructField("height", DoubleType(), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema))
        columns = ["age", "height"]
        min_result = tbl.min(columns)
        assert min_result.to_list("min") == [14, 8.5], \
            "the min value for age and height is not correct"

    def test_max(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", "123", 14, 8.5),
                ("alice", "34", 25, 9.7),
                ("rose", "25344", 23, 10.0)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", StringType(), True),
                             StructField("age", IntegerType(), True),
                             StructField("height", DoubleType(), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema))
        columns = ["age", "height"]
        min_result = tbl.max(columns)
        assert min_result.to_list("max") == [25, 10.0], \
            "the maximum value for age and height is not correct"

    def test_to_list(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", "123", 14, 8.5, [0, 0]),
                ("alice", "34", 25, 9.6, [1, 1]),
                ("rose", "25344", 23, 10.0, [2, 2])]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", StringType(), True),
                             StructField("age", IntegerType(), True),
                             StructField("height", DoubleType(), True),
                             StructField("array", ArrayType(IntegerType()), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema))
        list1 = tbl.to_list("name")
        list2 = tbl.to_list("num")
        list3 = tbl.to_list("age")
        list4 = tbl.to_list("height")
        list5 = tbl.to_list("array")
        assert list1 == ["jack", "alice", "rose"], "the result of name is not correct"
        assert list2 == ["123", "34", "25344"], "the result of num is not correct"
        assert list3 == [14, 25, 23], "the result of age is not correct"
        assert list4 == [8.5, 9.6, 10.0], "the result of height is not correct"
        assert list5 == [[0, 0], [1, 1], [2, 2]], "the result of array is not correct"

    def test_to_dict(self):
        spark = OrcaContext.get_spark_session()
        # test the case the column of key is unique
        data = [("jack", "123", 14),
                ("alice", "34", 25),
                ("rose", "25344", 23)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", StringType(), True),
                             StructField("age", IntegerType(), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema))
        dictionary = tbl.to_dict()
        print(dictionary)
        assert dictionary["name"] == ['jack', 'alice', 'rose']

    def test_add(self):
        spark = OrcaContext.get_spark_session()
        data = [("jack", "123", 14, 8.5),
                ("alice", "34", 25, 9.6),
                ("rose", "25344", 23, 10.0)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", StringType(), True),
                             StructField("age", IntegerType(), True),
                             StructField("height", DoubleType(), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema))
        columns = ["age", "height"]
        new_tbl = tbl.add(columns, 1.5)
        new_list = new_tbl.df.take(3)
        assert len(new_list) == 3, "new_tbl should have 3 rows"
        assert new_list[0]['age'] == 15.5, "the age of jack should increase 1.5"
        assert new_list[0]['height'] == 10, "the height of jack should increase 1.5"
        assert new_list[1]['age'] == 26.5, "the age of alice should increase 1.5"
        assert new_list[1]['height'] == 11.1, "the height of alice should increase 1.5"
        assert new_list[2]['age'] == 24.5, "the age of rose should increase 1.5"
        assert new_list[2]['height'] == 11.5, "the height of rose should increase 1.5"
        new_tbl = tbl.add(columns, -1)
        new_list = new_tbl.df.take(3)
        assert len(new_list) == 3, "new_tbl should have 3 rows"
        assert new_list[0]['age'] == 13, "the age of jack should decrease 1"
        assert new_list[0]['height'] == 7.5, "the height of jack should decrease 1"
        assert new_list[1]['age'] == 24, "the age of alice should decrease 1"
        assert new_list[1]['height'] == 8.6, "the height of alice should decrease 1"
        assert new_list[2]['age'] == 22, "the age of rose should decrease 1"
        assert new_list[2]['height'] == 9.0, "the height of rose should decrease 1"

    def test_sample(self):
        spark = OrcaContext.get_spark_session()
        df = spark.range(1000)
        feature_tbl = FeatureTable(df)
        total_line_1 = feature_tbl.size()
        feature_tbl2 = feature_tbl.sample(0.5)
        total_line_2 = feature_tbl2.size()
        assert int(total_line_1/2) - 100 < total_line_2 < int(total_line_1/2) + 100, \
            "the number of rows should be half"
        total_distinct_line = feature_tbl2.distinct().size()
        assert total_line_2 == total_distinct_line, "all rows should be distinct"

    def test_group_by(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data2.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)

        groupby_tbl1 = feature_tbl.group_by("col_4", agg={"col_1": ["sum", "count"]})
        assert groupby_tbl1.df.filter("col_4 == 'a' and sum(col_1) == 3").count() == 1, \
            "the sum of col_1 with col_4 = 'a' should be 3"
        assert groupby_tbl1.df.filter("col_4 == 'b' and `count(col_1)` == 5").count() == 1, \
            "the count of col_1 with col_4 = 'b' should be 5"

        groupby_tbl2 = feature_tbl.group_by(agg={"target": "avg", "col_2": "last"})
        assert groupby_tbl2.df.collect()[0]["avg(target)"] == 0.9, \
            "the mean of target should be 0.9"

        groupby_tbl3 = feature_tbl.group_by("col_5", agg=["max", "min"], join=True)
        assert len(groupby_tbl3.df.columns) == len(feature_tbl.df.columns) + 10, \
            "groupby_tbl3 should have (#df.columns - #columns)*len(agg)=10 more columns"
        assert groupby_tbl3.df.filter("col_5 == 'cc' and `max(col_2)` == 9").count() == \
            feature_tbl.df.filter("col_5 == 'cc'").count(), \
            "max of col_2 should 9 for all col_5 = 'cc' in groupby_tbl3"
        assert groupby_tbl3.df.filter("col_5 == 'aa' and `min(col_3)` == 1.0").count() == \
            feature_tbl.df.filter("col_5 == 'aa'").count(), \
            "min of col_3 should 1.0 for all col_5 = 'aa' in groupby_tbl3"

        groupby_tbl4 = feature_tbl.group_by(["col_4", "col_5"], agg="first", join=True)
        assert groupby_tbl4.df.filter("col_4 == 'b' and col_5 == 'dd' and `first(col_1)` == 0") \
            .count() == feature_tbl.df.filter("col_4 == 'b' and col_5 == 'dd'").count(), \
            "first of col_1 should be 0 for all col_4 = 'b' and col_5 = 'dd' in groupby_tbl4"

    def test_append_column(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/")
        df = FeatureTable.read_csv(file_path+"data.csv", header=True)
        df = df.append_column("z", 0)
        assert df.select("z").size() == 4
        assert df.filter("z == 0").size() == 4
        df = df.append_column("str", "a")
        assert df.select("str").size() == 4
        assert df.filter("str == 'a'").size() == 4
        df = df.append_column("float", 1.2)
        assert df.select("float").size() == 4
        assert df.filter("float == 1.2").size() == 4

    def test_ordinal_shuffle(self):
        spark = OrcaContext.get_spark_session()
        data = [("a", 14), ("b", 25), ("c", 23), ("d", 2), ("e", 1)]
        schema = StructType([StructField("name", StringType(), True),
                             StructField("num", IntegerType(), True)])
        tbl = FeatureTable(spark.createDataFrame(data, schema).repartition(1))
        shuffled_tbl = tbl.ordinal_shuffle_partition()
        rows = tbl.df.collect()
        shuffled_rows = shuffled_tbl.df.collect()
        rows.sort(key=lambda x: x[1])
        shuffled_rows.sort(key=lambda x: x[1])
        assert rows == shuffled_rows

    def test_write_parquet(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
        feature_tbl = FeatureTable.read_parquet(file_path)
        feature_tbl.write_parquet("saved.parquet")
        loaded_tbl = FeatureTable.read_parquet("saved.parquet")
        if os.path.exists("saved.parquet"):
            shutil.rmtree("saved.parquet")

    def test_read_csv(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/data.csv")
        feature_tbl = FeatureTable.read_csv(file_path, header=True)
        assert feature_tbl.size() == 4
        columns = feature_tbl.columns
        assert columns == ["col1", "col2", "col3"]
        records = feature_tbl.df.collect()
        assert isinstance(records[0][0], float)
        assert isinstance(records[0][1], str) and isinstance(records[0][1], str)
        file_path2 = os.path.join(self.resource_path, "friesian/feature/data_no_header.csv")
        feature_tbl2 = FeatureTable.read_csv(file_path2, names=["col1", "_col2", "col3"],
                                             dtype={"col1": "int"})
        assert feature_tbl2.size() == 4
        columns2 = feature_tbl2.columns
        assert columns2 == ["col1", "_col2", "col3"]
        records2 = feature_tbl2.df.collect()
        assert isinstance(records2[0][0], int)
        assert isinstance(records2[0][1], str) and isinstance(records2[0][1], str)
        feature_tbl3 = FeatureTable.read_csv(file_path, header=True, dtype=["int", "str", "str"])
        records3 = feature_tbl3.df.collect()
        assert isinstance(records3[0][0], int)
        assert isinstance(records3[0][1], str) and isinstance(records3[0][1], str)

    def test_category_encode_and_one_hot_encode(self):
        file_path = os.path.join(self.resource_path, "friesian/feature/data.csv")
        feature_tbl = FeatureTable.read_csv(file_path, header=True)
        feature_tbl, indices = feature_tbl.category_encode(columns=["col2", "col3"])
        assert isinstance(indices, list) and len(indices) == 2
        assert isinstance(indices[0], StringIndex) and isinstance(indices[1], StringIndex)
        assert indices[0].size() == 3 and indices[1].size() == 4
        dict1 = indices[0].to_dict()
        dict2 = indices[1].to_dict()
        records = feature_tbl.df.collect()
        assert records[0][1] == dict1["x"] and records[0][2] == dict2["abc"]
        assert records[3][1] == dict1["z"] and records[2][2] == dict2["aaa"]
        feature_tbl = feature_tbl.one_hot_encode(columns=["col2", "col3"], prefix=["o1", "o2"])
        feature_tbl.show()
        columns = feature_tbl.columns
        assert columns == ["col1", "o1_0", "o1_1", "o1_2", "o1_3", "o2_0",
                           "o2_1", "o2_2", "o2_3", "o2_4"]
        records = feature_tbl.df.collect()
        record = records[0]
        value1 = dict1["x"]
        value2 = dict2["abc"]
        for i in range(1, 4):
            if i == value1:
                assert record[i+1] == 1
            else:
                assert record[i+1] == 0
        for i in range(1, 5):
            if i == value2:
                assert record[i+5] == 1
            else:
                assert record[i+5] == 0

    def test_split(self):
        file_path = os.path.join(self.resource_path, "orca/learn/ncf.csv")
        feature_tbl = FeatureTable.read_csv(file_path, header=True, dtype="int")
        tbl1, tbl2 = feature_tbl.split([0.8, 0.2], seed=1128)
        total_size = feature_tbl.size()
        size1 = tbl1.size()
        size2 = tbl2.size()
        assert size1 + size2 == total_size


if __name__ == "__main__":
    pytest.main([__file__])
