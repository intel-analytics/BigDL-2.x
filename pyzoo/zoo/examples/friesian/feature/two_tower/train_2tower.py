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

import pickle
from pyspark import StorageLevel
import argparse

from pyspark.sql.functions import col, array

from zoo.orca import init_orca_context
from zoo.orca import OrcaContext
from zoo.orca.learn.tf2.estimator import Estimator
from zoo.friesian.feature import FeatureTable
from two_tower import *

spark_conf = {"spark.network.timeout": "10000000",
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
              "spark.eventLog.enabled": "true",
              "spark.eventLog.dir": "hdfs://172.16.0.105:8020/sparkHistoryLogs",
              "spark.app.name": "recsys-2tower",
              "spark.executor.memoryOverhead": "120g"}


def train(config, train_data, test_data, epochs=1, batch_size=128, model_dir='.'):

    def model_creator(config):
        model = build_model(config["user_col_info"], config["item_col_info"])
        print(model.summary())
        optimizer = tf.keras.optimizers.Adam(config["lr"])
        model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'Recall', 'AUC'])
        return model

    estimator = Estimator.from_keras(model_creator=model_creator,
                                     verbose=False,
                                     config=config)

    callbacks = []
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=5))

    train_count = train_data.count()
    test_count = test_data.count()
    steps_per_epoch = math.ceil(train_count / batch_size)
    val_steps = math.ceil(test_count / batch_size)
    feature_cols = user_col_info.get_name_list() + item_col_info.get_name_list()
    print("Total number of train records: {}".format(train_count))
    print("Total number of val records: {}".format(test_count))

    estimator.fit(train_data, epochs=epochs, batch_size=batch_size,
                  feature_cols=feature_cols,
                  label_cols=['label'],
                  callbacks=callbacks,
                  validation_data=test_data,
                  steps_per_epoch=steps_per_epoch,
                  validation_steps=val_steps)

    model = estimator.get_model()
    user_model = get_1tower_model(model, user_col_info)
    item_model = get_1tower_model(model, item_col_info)
    tf.saved_model.save(model, os.path.join(model_dir, "twotower-model"))
    tf.saved_model.save(user_model, os.path.join(model_dir, "user-model"))
    tf.saved_model.save(item_model, os.path.join(model_dir, "item-model"))
    model.save(model_dir + "/twotower_model.h5")
    user_model.save(model_dir + "/user_model.h5")
    item_model.save(model_dir + "/item_model.h5")
    estimator.save(os.path.join(model_dir, "twotower_model.ckpt"))
    print("saved models")
    return estimator


def load_data(data_path):
    def gen_label(x):
        # ts = [int(e) for e in x[:1] if e > 0]
        ts = [e for e in x[:] if e > 0]
        out = 1 if len(ts) > 0 else 0
        return out
    useful_cols = num_cols + cat_cols + embed_cols + ts_cols
    tbl = FeatureTable.read_parquet(data_path)

    # tbl = tbl.rename({"enaging_user_id": "user_id", "tweet_id": "item_id"})
    tbl = tbl.rename({"user": "user_id", "item": "item_id"})

    tbl = tbl.fillna(0, useful_cols)
    tbl = tbl.apply(in_col=ts_cols, out_col='label', func=gen_label, dtype='bigint')

    return tbl

def gen_stats_dicts(tbl, model_dir, frequency_limit=25):
    stats = {}
    for c in embed_cols + cat_cols + num_cols:
        min = tbl.select(c).get_stats(c, "min")[c]
        max = tbl.select(c).get_stats(c, "max")[c]
        total_num = tbl.select(c).size()
        dist_num = tbl.select(c).distinct().size()
        null_num = tbl.filter(col(c).isNull()).size()
        stats[c + "_min"] = min
        stats[c + "_max"] = max
        stats[c + "_total_num"] = total_num
        stats[c + "_ditinct_num"] = dist_num
        stats[c + "_null_num"] = null_num
        stats[c + "_null_ratio"] = null_num / float(total_num)

    for c in ts_cols + ["label"]:
        label0_cnt = tbl.select(c).filter(col(c) == 0).size()
        label1_cnt = tbl.select(c).filter(col(c) > 0).size()
        stats[c + "_label0_cnt"] = label0_cnt
        stats[c + "_label1_cnt"] = label1_cnt

    # for item in stats.items():
    #     print(item)
    stats_dir = model_dir + "/stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    with open(os.path.join(stats_dir, "feature_stats"), 'wb') as f:
        pickle.dump(stats, f)

    index_dicts=[]
    for c in embed_cols:
        c_count = tbl.select(c).group_by(c, agg={c: "count"}).rename({"count(" + c + ")": "count"})
        c_count.df.printSchema()
        c_count = c_count.filter(col("count") >= frequency_limit).order_by("count", ascending=False)
        c_count_pd = c_count.to_pandas()
        c_count_pd.reindex()
        c_count_pd[c + "_new"] = c_count_pd.index + 1
        index_dict = dict(zip(c_count_pd[c], c_count_pd[c + "_new"]))
        index_dict_reverse = dict(zip(c_count_pd[c + "_new"], c_count_pd[c]))
        index_dicts.append(index_dict)
        with open(os.path.join(stats_dir, c + "_index_dict.txt"), 'w') as text_file:
            c_count_pd.to_csv(text_file, index=False)
        with open(os.path.join(stats_dir, c + "_index_dict"), 'wb') as f:
            pickle.dump(index_dict, f)
        with open(os.path.join(stats_dir, c + "_index_dict_reverse"), 'wb') as f:
            pickle.dump(index_dict_reverse, f)
    return stats, index_dicts

def prepare_features(train_tbl, test_tbl, embed_index_dicts):
    def reindex_embeds(tbl):
        embed_in_dims = {}
        for i, c in enumerate(embed_cols):
            index_dict = embed_index_dicts[i]
            embed_in_dims[c] = max(index_dict.values()) + 1
            spark = OrcaContext.get_spark_session()
            index_dict_bc = spark.sparkContext.broadcast(index_dict)
            index_lookup = lambda x: index_dict_bc.value.get(x, 0)
            tbl = tbl.apply(c, c, index_lookup, "int")
        return tbl, embed_in_dims

    def add_ratio_features(tbl):
        cal_ratio = (lambda x: x[1] / x[0] if x[0] > 0 else 0.0)
        tbl = tbl.apply(["engaged_with_user_follower_count","engaged_with_user_following_count"],
                        'engaged_with_user_follower_following_ratio', cal_ratio, "float")\
            .apply(["enaging_user_follower_count","enaging_user_following_count"],
                    "enaging_user_follower_following_ratio", cal_ratio, "float")
        return tbl

    def organize_cols(tbl):
        tbl = tbl.select(array("enaging_user_follower_count", "enaging_user_following_count",
                             "enaging_user_follower_following_ratio").alias("user_num"),
                       array("len_hashtags", "len_domains", "len_links",
                             "engaged_with_user_follower_count",
                             "engaged_with_user_following_count",
                             "engaged_with_user_follower_following_ratio").alias("item_num"),
                       *cat_cols, *embed_cols, "label")
        return tbl

    print("reindexing embedding cols")
    train_tbl, embed_in_dims = reindex_embeds(train_tbl)
    test_tbl, _ = reindex_embeds(test_tbl)

    print("add ratio features")
    train_tbl = add_ratio_features(train_tbl)
    test_tbl = add_ratio_features(test_tbl)

    train_tbl, min_max_dic = train_tbl.min_max_scale(num_cols + ratio_cols)
    for column, (col_min, col_max) in min_max_dic.items():
        normalize = lambda x: (x - col_min) / (col_max - col_min)
        test_tbl = test_tbl.apply(column, column, normalize, "float")
    with open(os.path.join(args.model_dir, "stats/min_max.pkl"), 'wb') as f:
        pickle.dump(min_max_dic, f)

    user_col_info = ColumnInfo2Tower(indicator_cols=["enaging_user_is_verified"],
                                     indicator_dims=[2],
                                     embed_cols=["user_id"],
                                     embed_in_dims=[embed_in_dims["user_id"]],
                                     embed_out_dims=[16],
                                     numerical_cols=["user_num"],
                                     numerical_dims=[3],
                                     name="user")
    item_col_info = ColumnInfo2Tower(indicator_cols=["engaged_with_user_is_verified",
                                                     "present_media", "tweet_type", "language"],
                                     indicator_dims=[2, 13, 3, 67],  # max + 1
                                     embed_cols=["engaged_with_user_id", "hashtags",
                                                 "present_links", "present_domains"],
                                     embed_in_dims=[embed_in_dims["engaged_with_user_id"],
                                                    embed_in_dims["hashtags"],
                                                    embed_in_dims["present_links"],
                                                    embed_in_dims["present_domains"]],
                                     embed_out_dims=[16, 16, 16, 16],
                                     numerical_cols=["item_num"],
                                     numerical_dims=[6],
                                     name="item")

    print("organize columns and specify user_col_info and item_col_info")
    train_tbl = organize_cols(train_tbl)
    test_tbl = organize_cols(test_tbl)

    return train_tbl.df, test_tbl.df, user_col_info, item_col_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two Tower Training/Inference')
    parser.add_argument('--cluster_mode', type=str, default="spark-submit",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--frequency_limit', type=int, default = 25, help='frequency limit')

    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", init_ray_on_spark=True)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executor,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf,
                               init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf, extra_python_lib="two_tower.py", object_store_memory="80g",
                               init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")

    num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "len_hashtags", "len_domains", "len_links", "hashtags", "present_links",
                "present_domains"]
    cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified",
                "present_media", "tweet_type", "language"]
    ratio_cols = ["engaged_with_user_follower_following_ratio",
                  "enaging_user_follower_following_ratio"]
    ts_cols = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp',
               'like_timestamp']
    embed_cols = ["user_id", "engaged_with_user_id", "hashtags", "present_links", "present_domains"]

    #train_tbl, n_uid, n_mid = load_data(args.data_dir + "/saved.parquet")
    #test_tbl, _, _ = load_data(args.data_dir +"/saved_test.parquet")
    train_tbl = load_data(args.data_dir)
    test_tbl = load_data(args.data_dir)
    stats, embed_dicts = gen_stats_dicts((train_tbl.concat(test_tbl, "outer")),
                                         args.model_dir,
                                         frequency_limit=args.frequency_limit)
    train_df, test_df, user_col_info, item_col_info = prepare_features(train_tbl, test_tbl, embed_dicts)

    num_partitions=args.num_executor * args.executor_cores * 2
    train_df = train_df.repartition(num_partitions)
    test_df = test_df.repartition(num_partitions)
    train_df.persist(StorageLevel.DISK_ONLY)
    test_df.persist(StorageLevel.DISK_ONLY)

    train_config = {"lr": 1e-3,
                    "user_col_info": user_col_info,
                    "item_col_info": item_col_info,
                    "inter_op_parallelism": 4,
                    "intra_op_parallelism": args.executor_cores}

    train(train_config, train_df, test_df, epochs=args.epochs, batch_size=args.batch_size,
                      model_dir=args.model_dir)
