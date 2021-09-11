# Convert original parquet to be compatible with Spark.
# Original data can be loaded with pandas or pyarrow, but will throw error with Spark.
# Spark can't handle column name with space.
import pandas as pd
import os
from os import listdir
from argparse import ArgumentParser
import pandas as pd

from zoo.orca import init_orca_context, stop_orca_context, OrcaContext
from zoo.friesian.feature import FeatureTable

# conf = {"spark.network.timeout": "10000000",
#         "spark.sql.broadcastTimeout": "7200",
#         "spark.sql.shuffle.partitions": "2000",
#         "spark.locality.wait": "0s",
#         "spark.sql.crossJoin.enabled": "true",
#         "spark.task.cpus": "1",
#         "spark.executor.heartbeatInterval": "200s",
#         "spark.driver.maxResultSize": "40G",
#         "spark.app.name": "recsys-val-parquet",
#         "spark.executor.memoryOverhead": "30g"}

def _parse_args():
    parser = ArgumentParser()

    # parser.add_argument('--cluster_mode', type=str, default="local",
    #                     help='The cluster mode, such as local, yarn or standalone.')
    # parser.add_argument('--master', type=str, default=None,
    #                     help='The master url, only used when cluster mode is standalone.')
    # parser.add_argument('--executor_cores', type=int, default=44,
    #                     help='The executor core number.')
    # parser.add_argument('--executor_memory', type=str, default="130",
    #                     help='The executor memory.')
    # parser.add_argument('--num_executor', type=int, default=8,
    #                     help='The number of executor.')
    # parser.add_argument('--driver_cores', type=int, default=4,
    #                     help='The driver core number.')
    # parser.add_argument('--driver_memory', type=str, default="36g",
    #                     help='The driver memory.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the folder of parquet files.")
    parser.add_argument('--output_folder', type=str, default=".",
                        help="The path to save the preprocessed data to parquet files. ")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    # if args.cluster_mode == "local":
    #     init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    # elif args.cluster_mode == "standalone":
    #     init_orca_context("standalone", master=args.master,
    #                       cores=args.executor_cores, num_nodes=args.num_executor,
    #                       memory=args.executor_memory,
    #                       driver_cores=args.driver_cores,
    #                       driver_memory=args.driver_memory, conf=conf)
    # elif args.cluster_mode == "yarn":
    #     init_orca_context("yarn-client", cores=args.executor_cores,
    #                       num_nodes=args.num_executor, memory=args.executor_memory,
    #                       driver_cores=args.driver_cores, driver_memory=args.driver_memory,
    #                       conf=conf)


    input_files = [f for f in listdir(args.input_folder) if f.endswith(".parquet")]
    for f in input_files:
        df = pd.read_parquet(os.path.join(args.input_folder, f))
        df = df.rename(columns={"text_ tokens": "text_tokens"})
        # This is a typo. Other typos include enaging...
        df = df.rename(columns={"retweet_timestampe": "retweet_timestamp"})
        df.to_parquet(os.path.join(args.output_folder, f))
