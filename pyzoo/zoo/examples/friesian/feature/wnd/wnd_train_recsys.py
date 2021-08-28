from __future__ import division, print_function, unicode_literals

import sys
import os
import math
from optparse import OptionParser
from time import time
import subprocess
import tempfile
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.data.utils import extract_one_path

from zoo.models.recommendation.wide_and_deep import ColumnFeatureInfo
from zoo.orca import OrcaContext
from zoo.orca.learn.tf2.estimator import Estimator
from zoo.orca.data.file import open_text, exists, write_text, makedirs
from zoo.friesian.feature import FeatureTable
from zoo.common.utils import get_remote_file_to_local

from wnd_model import *

cross_cols = [['present_media', 'tweet_type']]

embedding_cols = [
                  'tweet_id',
                  'engaged_with_user_id',
                  'enaging_user_id',
                  'present_media',  #
                  'present_links',  #
                  'present_domains',  #
                  ]

indicator_cols = ['present_media',  #
                  'tweet_type',  #
                  'language',  #
                  'engaged_with_user_is_verified',
                  'enaging_user_is_verified',
                  ]

cont_cols = ['engaged_with_user_follower_count',  # Engaged With User Features
             'engaged_with_user_following_count',  #
             'enaging_user_follower_count',  # Engaging User Features
             'enaging_user_following_count',  #
             'len_hashtags',
             'len_domains',
             'len_links'
             ]


_SHUFFLE_BUFFER = 1500

conf = {"spark.network.timeout": "10000000",
        "spark.sql.broadcastTimeout": "7200",
        "spark.sql.shuffle.partitions": "2000",
        "spark.locality.wait": "0s",
        "spark.sql.hive.filesourcePartitionFileCacheSize": "4096000000",
        "spark.sql.crossJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.unsafe": "true",
        "spark.kryoserializer.buffer.max": "1024m",
        "spark.task.cpus": "1",
        "spark.executor.heartbeatInterval": "200s",
        "spark.driver.maxResultSize": "40G",
        "spark.driver.memoryOverhead": "100G",
        "spark.executor.memoryOverhead": "100G"}

def get_size(data_dir):
    spark = OrcaContext.get_spark_session()
    if data_dir.split("://")[0] == data_dir:  # no prefix
        data_dir_with_prefix = "file://" + data_dir
    else:
        data_dir_with_prefix = data_dir

    if exists(os.path.join(data_dir, "train_parquet")) and exists(os.path.join(data_dir, "test_parquet")):
        train_df = spark.read.parquet(os.path.join(data_dir_with_prefix, "train_parquet"))
        test_df = spark.read.parquet(os.path.join(data_dir_with_prefix, "test_parquet"))

    # get cat sizes
    with tempfile.TemporaryDirectory() as local_path:
        get_remote_file_to_local(os.path.join(data_dir, "meta/categorical_sizes.pkl"),
                                 os.path.join(local_path, "categorical_sizes.pkl"))
        with open(os.path.join(local_path, "categorical_sizes.pkl"), 'rb') as f:
            cat_sizes_dic = pickle.load(f)

    indicator_sizes = [cat_sizes_dic[c] for c in indicator_cols] + [1] * 2
    print("indicator sizes: ", indicator_sizes)
    embedding_sizes = [cat_sizes_dic[c] for c in embedding_cols]  # tweet_id, engaged_user, engaging_user
    print("embedding sizes: ", embedding_sizes)

    with tempfile.TemporaryDirectory() as local_path:
        get_remote_file_to_local(os.path.join(data_dir, "meta/cross_sizes.pkl"),
                                 os.path.join(local_path, "cross_sizes.pkl"))
        with open(os.path.join(local_path, "cross_sizes.pkl"), 'rb') as f:
            cross_sizes_dic = pickle.load(f)

    cross_sizes = [cross_sizes_dic[c] for c in ["_".join(cross_names) for cross_names in cross_cols]]

    return train_df, test_df, indicator_sizes, embedding_sizes, cross_sizes


def model_creator(config):
    model = build_model(class_num=config["class_num"],
                        column_info=config["column_info"],
                        model_type=config["model_type"],
                        hidden_units=config["hidden_units"])
    optimizer = tf.keras.optimizers.Adam(config["lr"])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The cluster mode, such as local, yarn or standalone.')
    parser.add_option('--master', type=str, default=None,
                      help='The master url, only used when cluster mode is standalone.')
    parser.add_option('--executor_cores', type=int, default=48,
                      help='The executor core number.')
    parser.add_option('--executor_memory', type=str, default="160g",
                      help='The executor memory.')
    parser.add_option('--num_executor', type=int, default=8,
                      help='The number of executor.')
    parser.add_option('--driver_cores', type=int, default=4,
                      help='The driver core number.')
    parser.add_option('--driver_memory', type=str, default="36g",
                      help='The driver memory.')
    parser.add_option("--data_dir", dest="data_dir")
    parser.add_option("--model_dir", dest="model_dir")
    parser.add_option("--model_type", dest="model_type", default="wnd")
    parser.add_option("--batch_size", "-b", dest="batch_size", default=400, type=int)
    parser.add_option("--epoch", "-e", dest="epochs", default=2, type=int)
    parser.add_option("--learning_rate", "-l", dest="learning_rate", default=1e-4, type=float)
    parser.add_option("--class_num", dest="class_num", default=2, type=int)
    parser.add_option('--mode', type=str, default="train", dest="mode")
    parser.add_option('--early_stopping', type=int, default=3, dest="early_stopping")
    parser.add_option('--hidden_units', dest="hidden_units", type=str,
                      help='hidden units for deep mlp', default="1024, 1024")

    (options, args) = parser.parse_args(sys.argv)
    options.hidden_units = [int(x) for x in options.hidden_units.split(',')]

    model_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wnd_model.py")

    if options.cluster_mode == "local":
        init_orca_context("local", cores=options.executor_cores, memory=options.executor_memory,
                          init_ray_on_spark=True)
    elif options.cluster_mode == "standalone":
        init_orca_context("standalone", master=options.master,
                          cores=options.executor_cores, num_nodes=options.num_executor,
                          memory=options.executor_memory,
                          driver_cores=options.driver_cores, driver_memory=options.driver_memory, conf=conf,
                          init_ray_on_spark=True,
                          extra_python_lib=model_file)
    elif options.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=options.executor_cores,
                          num_nodes=options.num_executor, memory=options.executor_memory,
                          driver_cores=options.driver_cores, driver_memory=options.driver_memory,
                          conf=conf,
                          init_ray_on_spark=True,
                          extra_python_lib=model_file)
    elif options.cluster_mode == "spark-submit":
        init_orca_context("spark-submit")

    train_df, test_df, indicator_sizes, embedding_sizes, cross_sizes = get_size(options.data_dir)

    column_info = ColumnFeatureInfo(wide_base_cols=[],
                                    wide_base_dims=[],
                                    wide_cross_cols=["_".join(cross_names) for cross_names in cross_cols],
                                    wide_cross_dims=cross_sizes,
                                    indicator_cols=indicator_cols,
                                    indicator_dims=indicator_sizes,
                                    embed_cols=embedding_cols,
                                    embed_in_dims=embedding_sizes,
                                    embed_out_dims=[16] * len(embedding_cols),
                                    continuous_cols=cont_cols,
                                    label="label"
                                    )

    config = {
        "class_num": options.class_num,
        "model_type": options.model_type,
        "lr": options.learning_rate,
        "hidden_units": options.hidden_units,
        "column_info": column_info,
        "inter_op_parallelism": 1,
        "intra_op_parallelism": 40
    }

    est = Estimator.from_keras(
        model_creator=model_creator,
        verbose=True,
        config=config,
        backend="tf2")

    if options.mode == "train":
        train_count = train_df.count()
        print("train size: ", train_count)
        steps = math.ceil(train_count / options.batch_size)
        test_count = test_df.count()
        print("test size: ", test_count)
        val_steps = math.ceil(test_count / options.batch_size)

        if not exists(options.model_dir):
            makedirs(options.model_dir)

        callbacks = []

        # early stopping
        earlystopping = options.early_stopping
        if earlystopping:
            from tensorflow.keras.callbacks import EarlyStopping

            callbacks.append(EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=earlystopping))

        start = time()
        est.fit(data=train_df,
                epochs=options.epochs,
                batch_size=options.batch_size,
                callbacks=callbacks,
                steps_per_epoch=steps,
                feature_cols=column_info.wide_base_cols +
                             column_info.wide_cross_cols +
                             column_info.indicator_cols +
                             column_info.embed_cols +
                             column_info.continuous_cols,
                label_cols=['label'])
        end = time()
        print("Training time is: ", end - start)
        est.save(os.path.join(options.model_dir, "model-%d.ckpt" % options.epochs))
        model = est.get_model()
        model.save_weights(os.path.join(options.model_dir, "model.h5"))

        prediction_df = est.predict(test_df, batch_size=options.batch_size,
                                    feature_cols=column_info.wide_base_cols +
                                                 column_info.wide_cross_cols +
                                                 column_info.indicator_cols +
                                                 column_info.embed_cols +
                                                 column_info.continuous_cols
                                    )
        prediction_df.cache()

        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                                  labelCol="label",
                                                  metricName="areaUnderROC")
        auc = evaluator.evaluate(prediction_df)
        print("AUC score is: ", auc)
    stop_orca_context()
