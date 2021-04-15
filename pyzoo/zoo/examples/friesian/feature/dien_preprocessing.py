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

import sys
import os
import time
from pyspark import StorageLevel

try:
    import cPickle as pkl
except ModuleNotFoundError:
    import pickle as pkl

from optparse import OptionParser
from zoo.orca import init_orca_context, stop_orca_context, OrcaContext
from pyspark.sql.functions import udf
from zoo.friesian.feature import FeatureTable
from pyspark.sql.types import StringType, IntegerType, ArrayType, FloatType

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--meta", dest="meta_file")
    parser.add_option("--review", dest="review_file")
    parser.add_option("--output", dest="output")
    (options, args) = parser.parse_args(sys.argv)
    begin = time.time()
    sc = init_orca_context("local")
    spark = OrcaContext.get_spark_session()

    # read review datavi run.sh
    review_df = spark.read.json(options.review_file).select(
        ['reviewerID', 'asin', 'unixReviewTime']) \
        .withColumnRenamed('reviewerID', 'user') \
        .withColumnRenamed('asin', 'item') \
        .withColumnRenamed('unixReviewTime', 'time')\
        .dropna("any").persist(storageLevel=StorageLevel.DISK_ONLY)
    print("review_df, ", review_df.count())

    # read meta data
    def get_category(x):
        cat = x[0][-1] if x[0][-1] is not None else "default"
        return cat.strip().lower()
    spark.udf.register("get_category", get_category, StringType())
    meta_df = spark.read.json(options.meta_file).select(['asin', 'categories'])\
        .dropna(subset=['asin', 'categories']) \
        .selectExpr("*", "get_category(categories) as category") \
        .withColumnRenamed("asin", "item").drop("categories").distinct()\
        .persist(storageLevel=StorageLevel.DISK_ONLY)
    print("meta_data, ", meta_df.count())

    meta_tbl = FeatureTable(meta_df)
    review_tbl = FeatureTable(review_df)
    full_tbl = review_tbl.join(meta_tbl, on="item", how="left").fillna("default", ["category"])
    item_size = full_tbl.df.select("item").distinct().count()

    print("full data after join,", full_tbl.count())
    indices = full_tbl.gen_string_idx(['user', 'item', 'category'], 1)
    item2cat = full_tbl.gen_ind2ind(['item', 'category'], [indices[1], indices[2]])
    item2cat_map = dict(item2cat.df.rdd.map(lambda row: (row[0], row[1])).collect())

    for i in range(len(indices)):
        indices[i].write_parquet(options.output)
    item2cat.write_parquet(options.output+"item2cat")

    get_label = udf(lambda x: [float(x), 1 - float(x)], ArrayType(FloatType()))
    reset_cat = udf(lambda item: item2cat_map[item], returnType=IntegerType())

    full_tbl = full_tbl\
        .encode_string(['user', 'item', 'category'], [indices[0], indices[1], indices[2]]) \
        .gen_hist_seq(user_col="user", cols=['item', 'category'],
                      sort_col='time', min_len=1, max_len=100)\
        .gen_length("item_history")\
        .gen_negative_samples(item_size, item_col='item', neg_num=1) \
        .transform_python_udf("item", "category", reset_cat) \
        .gen_neg_hist_seq(item_size, item2cat.df, 'item_history', neg_num=5) \
        .mask_pad(
            padding_cols=['item_history', 'category_history', 'noclk_item_list', 'noclk_cat_list'],
            mask_cols=['item_history'],
            seq_len=100)\
        .transform_python_udf("label", "label", get_label)
    full_tbl.write_parquet(options.output + "data")

    print("final output count, ", full_tbl.count())
    stop_orca_context()
    end = time.time()
    print(f"perf preprocessing time: {(end - begin):.2f}s")
