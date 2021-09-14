# Convert original parquet to be compatible with Spark.
# Original data can be loaded with pandas or pyarrow, but will throw error with Spark.
# Spark can't handle column name with space.

import os
from os import listdir
from argparse import ArgumentParser
import pandas as pd

def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the folder of parquet files.")
    parser.add_argument('--output_folder', type=str, default=".",
                        help="The path to save the preprocessed data to parquet files. ")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()

    input_files = [f for f in listdir(args.input_folder) if f.endswith(".parquet")]
    for f in input_files:
        df = pd.read_parquet(os.path.join(args.input_folder, f))
        df = df.rename(columns={"text_ tokens": "text_tokens"})
        # This is a typo. Other typos include enaging...
        df = df.rename(columns={"retweet_timestampe": "retweet_timestamp"})
        df.to_parquet(os.path.join(args.output_folder, "spark_parquet/%s" % f))
