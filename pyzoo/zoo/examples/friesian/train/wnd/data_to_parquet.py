#!/usr/bin/env python
# coding: utf-8

import os
from time import time
from argparse import ArgumentParser

from pyspark.sql.types import *
from zoo.orca import init_orca_context, stop_orca_context, OrcaContext
from zoo.friesian.feature import FeatureTable

conf = {"spark.network.timeout": "10000000",
        "spark.sql.broadcastTimeout": "7200",
        "spark.sql.shuffle.partitions": "2000",
        "spark.locality.wait": "0s",
        "spark.sql.crossJoin.enabled": "true",
        "spark.task.cpus": "1",
        "spark.executor.heartbeatInterval": "200s",
        "spark.driver.maxResultSize": "40G",
        "spark.eventLog.enabled": "true",
        "spark.eventLog.dir": "hdfs://172.16.0.105:8020/sparkHistoryLogs",
        "spark.app.name": "recsys-kai",
        "spark.executor.memoryOverhead": "64g"}

class RecsysSchema:
    def __init__(self):
        self.string_cols1 = [
            'text_tokens',
            'hashtags',  # Tweet Features
            'tweet_id',       #
            'present_media',          #
            'present_links',          #
            'present_domains',        #
            'tweet_type',     #
            'language',       #
        ]
        self.int_cols1 = [
            'tweet_timestamp',
        ]
        self.string_cols2 = [
            'engaged_with_user_id',
        ]
        self.int_cols2 = [
            'engaged_with_user_follower_count',  # Engaged With User Features
            'engaged_with_user_following_count',      #
        ]
        self.bool_cols1 = [
            'engaged_with_user_is_verified',          #
        ]
        self.int_cols3 = [
            'engaged_with_user_account_creation',
        ]
        self.string_cols3 = [
            'enaging_user_id',
        ]
        self.int_cols4 = [
            'enaging_user_follower_count',  # Engaging User Features
            'enaging_user_following_count',      #
        ]
        self.bool_cols2 = [
            'enaging_user_is_verified',
        ]
        self.int_cols5 = [
            'enaging_user_account_creation',
        ]
        self.bool_cols3 = [
            'engagee_follows_engager',  # Engagement Features

        ]
        self.float_cols = [
            'reply_timestamp',  # Target Reply
            'retweet_timestamp',  # Target Retweet
            'retweet_with_comment_timestamp',  # Target Retweet with comment
            'like_timestamp',  # Target Like
        ]

        # After some conversion
        self.int_cols6 = [
            'tweet_timestamp',
            'engaged_with_user_follower_count',  # Engaged With User Features
            'engaged_with_user_following_count',      #
            'engaged_with_user_account_creation',
            'enaging_user_follower_count',  # Engaging User Features
            'enaging_user_following_count',           #
            'enaging_user_account_creation',
        ]

    def toDtype(self):
        str_fields1 = ["str" for i in self.string_cols1]
        int_fields1 = ["int" for i in self.int_cols1]
        str_fields2 = ["str" for i in self.string_cols2]
        int_fields2 = ["int" for i in self.int_cols2]
        bool_fields1 = ["bool" for i in self.bool_cols1]
        int_fields3 = ["int" for i in self.int_cols3]
        str_fields3 = ["str" for i in self.string_cols3]
        int_fields4 = ["int" for i in self.int_cols4]
        bool_fields2 = ["bool" for i in self.bool_cols2]
        int_fields5 = ["int" for i in self.int_cols5]
        bool_fields3 = ["bool" for i in self.bool_cols3]
        float_fields = ["float" for i in self.float_cols]
        return str_fields1 \
               + int_fields1 \
               + str_fields2 \
               + int_fields2 \
               + bool_fields1 \
               + int_fields3 \
               + str_fields3 \
               + int_fields4 \
               + bool_fields2 \
               + int_fields5 \
               + bool_fields3 \
               + float_fields




def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=44,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="130",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the folder of parquet files.")
    parser.add_argument('--output_folder', type=str, default=".",
                        help="The path to save the preprocessed data to parquet files. ")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master,
                          cores=args.executor_cores, num_nodes=args.num_executor,
                          memory=args.executor_memory,
                          driver_cores=args.driver_cores,
                          driver_memory=args.driver_memory, conf=conf)
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.executor_cores,
                          num_nodes=args.num_executor, memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          conf=conf)

    start = time()
    train_tbl = FeatureTable.read_csv(os.path.join(args.input_folder,"train"),
                                      delimiter="\x01",
                                      dtype=RecsysSchema().toDtype()
                                      )
    train_tbl.write_parquet(os.path.join(args.output_folder, "spark_parquet"))

    val_tbl = FeatureTable.read_csv(os.path.join(args.input_folder, "val"),
                                      delimiter="\x01",
                                      dtype=RecsysSchema().toDtype())

    val_tbl.write_parquet(os.path.join(args.output_folder, "test_spark_parquet"))

    end = time()
    print("Convert to parquet time: ", end - start)
    stop_orca_context()



