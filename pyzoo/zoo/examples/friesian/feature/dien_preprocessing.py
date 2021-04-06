import json
import random
import sys
import os

from pyspark import StorageLevel

try:
    import cPickle as pkl
except ModuleNotFoundError:
    import pickle as pkl

from optparse import OptionParser

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from zoo import init_nncontext
from zoo.orca.data.pandas import read_json
from zoo.orca.data.shard import SharedValue
from zoo.friesian.feature import FeatureTable
from pyspark.sql.types import StringType, IntegerType, ArrayType, FloatType

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--meta", dest="meta_file", default='/Users/guoqiong/intelWork/git/friesian/data/book_review/meta_Books.json')
    parser.add_option("--review", dest="review_file",default="/Users/guoqiong/intelWork/git/friesian/data/book_review/reviews_Books.json")
    parser.add_option("--output", dest="output", default='/Users/guoqiong/intelWork/git/friesian/data/book_review/reprocessed_small')
    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext()
    spark = SparkSession(sc)

    # read review datavi run.sh
    review_df = spark.read.json(options.review_file).select(
        ['reviewerID', 'asin', 'unixReviewTime']) \
        .withColumnRenamed('reviewerID', 'user') \
        .withColumnRenamed('asin', 'item') \
        .withColumnRenamed('unixReviewTime', 'time')\
        .dropna("any").sample(0.0001).persist(storageLevel=StorageLevel.DISK_ONLY)

    # read meta data
    def get_category(x):
        cat = x[0][-1] if x[0][-1] is not None else "default"
        return cat.strip().lower()
    spark.udf.register("get_category", get_category, StringType())
    meta_df = spark.read.json(options.meta_file).select(['asin', 'categories'])\
        .dropna(subset=['asin', 'categories']) \
        .selectExpr("*", "get_category(categories) as category") \
        .withColumnRenamed("asin", "item").drop("categories").persist(storageLevel=StorageLevel.DISK_ONLY)

    full_df = review_df.join(meta_df, on="item", how="left").fillna("default", ["item"])

    full_df.printSchema()
    sys.exit()

    item_size = full_df.select("item").distinct().count()

    full_df.persist(StorageLevel.DISK_ONLY)
    full_tbl = FeatureTable(full_df)
    indices = full_tbl.gen_string_idx(['user', 'item', 'category'], '1')
    item2cat = full_tbl.gen_ind2ind(['item', 'category'], [indices[1], indices[2]])

    for i in range(len(indices)):
        indices[i].write_parquet(options.output)
    item2cat.write_parquet(options.output+"item2cat")

    get_length = udf(lambda x: len(x), IntegerType())
    get_label = udf(lambda x: [float(x), 1 - float(x)], ArrayType(FloatType()))

    full_tbl = full_tbl\
        .encode_string(['user', 'item', 'category'], [indices[0], indices[1], indices[2]])\
        .gen_his_seq(user_col="user", cols=['item', 'category'], sort_col='time', min_len=1, max_len=100)\
        .transform_python_udf("item_history", "length", get_length)\
        .add_negtive_samples(item_size, item_col='item', neg_num=1) \
        .gen_neg_hist_seq(item_size, item2cat, neg_num=5)\
        .mask_pad(
            padding_cols=['item_history', 'category_history', 'noclk_item_list', 'noclk_cat_list'],
            mask_cols=['item_history'],
            seq_len=100)\
        .transform_python_udf("label", "label", get_label)
    full_tbl.write_parquet(options.output + "data")
    sc.stop()
