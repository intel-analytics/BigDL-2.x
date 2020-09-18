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

from bigdl.util.common import get_node_and_core_number
from zoo import init_nncontext
from zoo.orca import OrcaContext
from zoo.orca.data import SparkXShards
from zoo.orca.data.utils import *


def read_csv(file_path, **kwargs):
    """
    Read csv files to SparkXShards of pandas DataFrames.

    :param file_path: A csv file path, a list of multiple csv file paths, or a directory
    containing csv files. Local file system, HDFS, and AWS S3 are supported.
    :param kwargs: You can specify read_csv options supported by pandas.
    :return: An instance of SparkXShards.
    """
    return read_file_spark(file_path, "csv", **kwargs)


def read_json(file_path, **kwargs):
    """
    Read json files to SparkXShards of pandas DataFrames.

    :param file_path: A json file path, a list of multiple json file paths, or a directory
    containing json files. Local file system, HDFS, and AWS S3 are supported.
    :param kwargs: You can specify read_json options supported by pandas.
    :return: An instance of SparkXShards.
    """
    return read_file_spark(file_path, "json", **kwargs)


def read_file_spark(file_path, file_type, **kwargs):
    sc = init_nncontext()
    node_num, core_num = get_node_and_core_number()
    backend = OrcaContext.pandas_read_backend

    if backend == "pandas":
        file_url_splits = file_path.split("://")
        prefix = file_url_splits[0]

        file_paths = []
        if isinstance(file_path, list):
            [file_paths.extend(extract_one_path(path, os.environ)) for path in file_path]
        else:
            file_paths = extract_one_path(file_path, os.environ)

        if not file_paths:
            raise Exception("The file path is invalid or empty, please check your data")

        num_files = len(file_paths)
        total_cores = node_num * core_num
        num_partitions = num_files if num_files < total_cores else total_cores
        rdd = sc.parallelize(file_paths, num_partitions)

        if prefix == "hdfs":
            pd_rdd = rdd.mapPartitions(
                lambda iter: read_pd_hdfs_file_list(iter, file_type, **kwargs))
        elif prefix == "s3":
            pd_rdd = rdd.mapPartitions(
                lambda iter: read_pd_s3_file_list(iter, file_type, **kwargs))
        else:
            def loadFile(iterator):
                dfs = []
                for x in iterator:
                    df = read_pd_file(x, file_type, **kwargs)
                    dfs.append(df)
                import pandas as pd
                return [pd.concat(dfs)]

            pd_rdd = rdd.mapPartitions(loadFile)
    else:  # Spark backend; spark.read.csv/json accepts a folder path as input
        assert file_type == "json" or file_type == "csv", \
            "Unsupported file type: %s. Only csv and json files are supported for now" % file_type
        from pyspark.sql import SQLContext
        sqlContext = SQLContext.getOrCreate(sc)
        spark = sqlContext.sparkSession
        # TODO: add S3 confidentials

        # The following implementation is adapted from
        # https://github.com/databricks/koalas/blob/master/databricks/koalas/namespace.py
        # with some modifications.

        if "mangle_dupe_cols" in kwargs:
            assert kwargs["mangle_dupe_cols"], "mangle_dupe_cols can only be True"
            kwargs.pop("mangle_dupe_cols")
        if "parse_dates" in kwargs:
            assert not kwargs["parse_dates"], "parse_dates can only be False"
            kwargs.pop("parse_dates")

        names = kwargs.get("names", None)
        if "names" in kwargs:
            kwargs.pop("names")
        usecols = kwargs.get("usecols", None)
        if "usecols" in kwargs:
            kwargs.pop("usecols")
        dtype = kwargs.get("dtype", None)
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        squeeze = kwargs.get("squeeze", False)
        if "squeeze" in kwargs:
            kwargs.pop("squeeze")
        index_col = kwargs.get("index_col", None)
        if "index_col" in kwargs:
            kwargs.pop("index_col")

        if file_type == "csv":
            # Handle pandas-compatible keyword arguments
            kwargs["inferSchema"] = True
            header = kwargs.get("header", "infer")
            if isinstance(names, str):
                kwargs["schema"] = names
            if header == "infer":
                header = 0 if names is None else None
            if header == 0:
                kwargs["header"] = True
            elif header is None:
                kwargs["header"] = False
            else:
                raise ValueError("Unknown header argument {}".format(header))
            if "quotechar" in kwargs:
                quotechar = kwargs["quotechar"]
                kwargs.pop("quotechar")
                kwargs["quote"] = quotechar
            if "escapechar" in kwargs:
                escapechar = kwargs["escapechar"]
                kwargs.pop("escapechar")
                kwargs["escape"] = escapechar
            # sep and comment are the same as pandas
            if "comment" in kwargs:
                comment = kwargs["comment"]
                if not isinstance(comment, str) or len(comment) != 1:
                    raise ValueError("Only length-1 comment characters supported")
            df = spark.read.csv(file_path, **kwargs)
            if header is None:
                df = df.selectExpr(
                    *["`%s` as `%s`" % (field.name, i) for i, field in enumerate(df.schema)])
        else:
            df = spark.read.json(file_path, **kwargs)

        # Handle pandas-compatible postprocessing arguments
        if usecols is not None and not callable(usecols):
            usecols = list(usecols)
        renamed = False
        if isinstance(names, list):
            if len(set(names)) != len(names):
                raise ValueError("Found duplicate names, please check your names input")
            if usecols is not None:
                if not callable(usecols):
                    # usecols is list
                    if len(names) != len(usecols) and len(names) != len(df.schema):
                        raise ValueError(
                            "Passed names did not match usecols"
                        )
                if len(names) == len(df.schema):
                    df = df.selectExpr(
                        *["`%s` as `%s`" % (field.name, name) for field, name
                          in zip(df.schema, names)]
                    )
                    renamed = True

            else:
                if len(names) != len(df.schema):
                    raise ValueError(
                        "The number of names [%s] does not match the number "
                        "of columns [%d]. Try names by a Spark SQL DDL-formatted "
                        "string." % (len(names), len(df.schema))
                    )
                df = df.selectExpr(
                    *["`%s` as `%s`" % (field.name, name) for field, name
                      in zip(df.schema, names)]
                )
                renamed = True
        index_map = dict([(i, field.name) for i, field in enumerate(df.schema)])
        if usecols is not None:
            if callable(usecols):
                cols = [field.name for field in df.schema if usecols(field.name)]
                missing = []
            elif all(isinstance(col, int) for col in usecols):
                cols = [field.name for i, field in enumerate(df.schema) if i in usecols]
                missing = [
                    col
                    for col in usecols
                    if col >= len(df.schema) or df.schema[col].name not in cols
                ]
            elif all(isinstance(col, str) for col in usecols):
                cols = [field.name for field in df.schema if field.name in usecols]
                if isinstance(names, list):
                    missing = [c for c in usecols if c not in names]
                else:
                    missing = [col for col in usecols if col not in cols]
            else:
                raise ValueError(
                    "usecols must only be list-like of all strings, "
                    "all unicode, all integers or a callable.")
            if len(missing) > 0:
                raise ValueError(
                    "usecols do not match columns, columns expected but not found: %s" % missing)
            if len(cols) > 0:
                df = df.select(cols)
                if isinstance(names, list):
                    if not renamed:
                        df = df.selectExpr(
                            *["`%s` as `%s`" % (col, name) for col, name in zip(cols, names)]
                        )
                        # update index map after rename
                        for index, col in index_map.items():
                            if col in cols:
                                index_map[index] = names[cols.index(col)]

        if df.rdd.getNumPartitions() < node_num:
            df = df.repartition(node_num)

        def to_pandas(columns, squeeze=False, index_col=None):
            def f(iter):
                import pandas as pd
                data = list(iter)
                pd_df = pd.DataFrame(data, columns=columns)
                if dtype is not None:
                    if isinstance(dtype, dict):
                        for col, type in dtype.items():
                            if isinstance(col, str):
                                if col not in pd_df.columns:
                                    raise ValueError("column to be set type is not"
                                                     " in current dataframe")
                                pd_df[col] = pd_df[col].astype(type)
                            elif isinstance(col, int):
                                if index_map[col] not in pd_df.columns:
                                    raise ValueError("column index to be set type is not"
                                                     " in current dataframe")
                                pd_df[index_map[col]] = pd_df[index_map[col]].astype(type)
                    else:
                        pd_df = pd_df.astype(dtype)
                if squeeze and len(pd_df.columns) == 1:
                    pd_df = pd_df.iloc[:, 0]
                if index_col:
                    pd_df = pd_df.set_index(index_col)

                return [pd_df]

            return f

        pd_rdd = df.rdd.mapPartitions(to_pandas(df.columns, squeeze, index_col))

    try:
        data_shards = SparkXShards(pd_rdd)
    except Exception as e:
        alternative_backend = "pandas" if backend == "spark" else "spark"
        print("An error occurred when reading files with '%s' backend, you may switch to '%s' "
              "backend for another try. You can set the backend using "
              "OrcaContext.pandas_read_backend" % (backend, alternative_backend))
        raise e
    return data_shards


def read_parquet(file_path, columns=None, **kwargs):
    """
    Read parquet files to SparkXShards of pandas DataFrames.

    :param file_path: Parquet file path, a list of multiple parquet file paths, or a directory
    containing parquet files. Local file system, HDFS, and AWS S3 are supported.
    :param columns: list of column name, default=None.
    If not None, only these columns will be read from the file.
    :param kwargs: Any additional kwargs.
    :return: An instance of SparkXShards.
    """
    sc = init_nncontext()
    from pyspark.sql import SQLContext
    sqlContext = SQLContext.getOrCreate(sc)
    spark = sqlContext.sparkSession
    df = spark.read.parquet(file_path)

    if columns:
        df = df.select(*columns)

    def to_pandas(columns):
        def f(iter):
            import pandas as pd
            data = list(iter)
            pd_df = pd.DataFrame(data, columns=columns)
            return [pd_df]

        return f

    pd_rdd = df.rdd.mapPartitions(to_pandas(df.columns))
    try:
        data_shards = SparkXShards(pd_rdd)
    except Exception as e:
        print("An error occurred when reading parquet files")
        raise e
    return data_shards
