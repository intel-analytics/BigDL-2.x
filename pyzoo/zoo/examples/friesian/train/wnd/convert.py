# Convert original parquet to be compatible with Spark.
# Original data can be loaded with pandas or pyarrow, but will throw error with Spark.
# Spark can't handle column name with space.
import pandas as pd
import os

pd.read_csv(os.path.join(args.input_folder, "valid"),
                                    delimiter="\x01",

df = pd.read_parquet("original/part-00000.parquet")
df = df.rename(columns={"text_ tokens": "text_tokens"})
# This is a typo. Other typos include enaging...
df = df.rename(columns={"retweet_timestampe": "retweet_timestamp"})
df.to_parquet("part-00000.parquet")
