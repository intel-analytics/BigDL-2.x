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

from zoo.friesian.feature import FeatureTable

import os

from argparse import ArgumentParser
from time import time
import numpy as np

from zoo.orca import init_orca_context, stop_orca_context
from zoo.util.utils import get_node_ip

LABEL_COL = 0
INT_COLS = ["_c{}".format(i) for i in list(range(1, 14))]
CAT_COLS = ["_c{}".format(i) for i in list(range(14, 40))]

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
        "spark.driver.maxResultSize": "40G"}


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=48,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="240g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--days', type=str, required=True,
                        help="Day range for preprocessing, such as 0-23, 0-1.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the folder of parquet files.")
    parser.add_argument('--output_folder')
    parser.add_argument(
        '--write_mode',
        choices=['overwrite', 'errorifexists'],
        default='errorifexists')

    parser.add_argument('--frequency_limit')

    args = parser.parse_args()
    start, end = args.days.split('-')
    args.day_range = list(range(int(start), int(end) + 1))
    args.days = len(args.day_range)

    return args


def save_file(folder_path):
    def f(index, iterator):
        ip = get_node_ip()
        part_size = 1000000
        buffer = []
        sub_index = 0
        for record in iterator:
            if len(buffer) == part_size:
                import random
                random.shuffle(buffer)
                X_int_buffer = np.array([record[1] for record in buffer], dtype=np.float32)
                X_cat_buffer = np.array([record[2] for record in buffer], dtype=np.int32)
                y_buffer = np.array([record[0] for record in buffer], dtype=np.int32)
                file_name = "partition{}_{}_{}.npz".format(index, sub_index, part_size)
                np.savez_compressed(
                    folder_path + file_name,
                    X_int=X_int_buffer,
                    X_cat=X_cat_buffer,
                    y=y_buffer,
                    size=part_size
                )
                sub_index += 1
                buffer = [[record[0], record[1], record[2]]]
                yield index, part_size, file_name, ip
            else:
                buffer.append([record[0], record[1], record[2]])
        remain_size = len(buffer)
        if remain_size > 0:
            import random
            random.shuffle(buffer)
            X_int_buffer = np.array([record[1] for record in buffer], dtype=np.float32)
            X_cat_buffer = np.array([record[2] for record in buffer], dtype=np.int32)
            y_buffer = np.array([record[0] for record in buffer], dtype=np.int32)
            file_name = "partition{}_{}_{}.npz".format(index, sub_index, remain_size)
            np.savez_compressed(
                folder_path + file_name,
                X_int=X_int_buffer,
                X_cat=X_cat_buffer,
                y=y_buffer,
                size=remain_size
            )
            yield index, remain_size, file_name, ip

    return f


def preprocess_and_save(data_tbl, models, mode, partitions=None):
    data_tbl = data_tbl.encode_string(CAT_COLS, models) \
        .fillna(0, INT_COLS + CAT_COLS).log(INT_COLS)
    data_tbl = data_tbl.merge_cols(INT_COLS, "X_int").merge_cols(CAT_COLS, "X_cat")
    rdd = data_tbl.to_spark_df().rdd
    if mode == "train":
        save_path = "/disk1/saved_data/"
    elif mode == "test":
        assert partitions
        rdd = rdd.repartition(partitions)
        save_path = "/disk1/saved_data_test/"
    else:
        raise ValueError("mode should be either train or test")
    print("Saving data files to disk")
    save_res = rdd.mapPartitionsWithIndex(
        save_file(save_path)).collect()
    return save_res


def process_save_res(save_res):
    size_map = {}
    for partition_id, subpartition_size, file_name, ip in save_res:
        if ip not in size_map:
            size_map[ip] = {}
        if partition_id not in size_map[ip]:
            size_map[ip][partition_id] = []
        size_map[ip][partition_id].append(subpartition_size)
    size = 0
    count = 0
    for node, data in size_map.items():
        for partition_id, subpartition_size in data.items():
            size += sum(subpartition_size)
            count += len(subpartition_size)
        print("Node {} has {} subpartitions and {} records".format(node, count, size))
        size = 0
        count = 0


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
    time_start = time()
    paths = [os.path.join(args.input_folder, "day_%d.parquet" % i) for i in args.day_range]
    tbl = FeatureTable.read_parquet(paths)
    idx_list = tbl.gen_string_idx(CAT_COLS, freq_limit=args.frequency_limit)

    train_data = FeatureTable.read_parquet(paths[:-1])
    train_save_res = preprocess_and_save(train_data, idx_list, "train")

    test_data = FeatureTable.read_parquet(os.path.join(args.input_folder, "day_23_test.parquet"))
    test_save_res = preprocess_and_save(test_data, idx_list, "test",
                                        partitions=args.executor_cores*args.num_executor)

    time_end = time()
    print("Total time: ", time_end - time_start)
    train_data.show(5)
    print("Train data distribution: ")
    process_save_res(train_save_res)
    print("Test data distribution: ")
    process_save_res(test_save_res)
    print("Finished")
    stop_orca_context()
