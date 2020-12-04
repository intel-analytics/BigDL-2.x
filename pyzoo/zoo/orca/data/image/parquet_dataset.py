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

import json
from pyspark import SparkContext
from pyspark.sql import SparkSession

from zoo.orca.data import XShards, SparkXShards
from zoo.orca.data.file import open_text, write_text
from zoo.orca.data.image.utils import chunks, dict_to_row, EnumEncoder, as_enum, row_to_dict, encode_schema, \
    decode_schema, SchemaField, FeatureType, DType, ndarray_dtype_to_dtype
from bigdl.util.common import get_node_and_core_number
import os
import random


class ParquetDataset:

    @staticmethod
    def write(path, generator, schema, block_size=1000, write_mode="overwrite", **kwargs):
        """
        Take each record in the generator and write it to a parquet file.
        
        **generator**
        Each record in the generator is a dict, the key is a string and will be the column name of saved parquet record and
        the value is the data.
        
        **schema**
        schema defines the name, dtype, shape of a column, as well as the feature type of a column. The feature type, defines
        how to encode and decode the column value.
        
        There are three kinds of feature type:
        1. Scalar, such as a int or float number, or a string, which can be directly mapped to a parquet type
        2. NDarray, which takes a np.ndarray and save it serialized bytes. The corresponding parquet type is BYTE_ARRAY .
        3. Image, which takes a string representing a image file in local file system and save the raw file content bytes.
           The corresponding parquet type is BYTE_ARRAY.
        
        :param path: the output path, e.g. file:///output/path, hdfs:///output/path
        :param generator: generate a dict, whose key is a string and value is one of (a scalar value, ndarray, image file path)
        :param schema: a dict, whose key is a string, value is one of (schema_field.Scalar, schema_field.NDarray, schema_field.Image)
        :param kwargs: other args
        """

        sc = SparkContext.getOrCreate()
        spark = SparkSession(sc)
        node_num, core_num = get_node_and_core_number()
        for i, chunk in enumerate(chunks(generator, block_size)):
            chunk_path = os.path.join(path, f"chunk={i}")
            rows_rdd = sc.parallelize(chunk, core_num * node_num).map(lambda x: dict_to_row(schema, x))
            spark.createDataFrame(rows_rdd).write.mode(write_mode).parquet(chunk_path)
        metadata_path = os.path.join(path, "_orca_metadata")

        write_text(metadata_path, encode_schema(schema))

    @staticmethod
    def _read_as_dict_rdd(path):
        sc = SparkContext.getOrCreate()
        spark = SparkSession(sc)
        df = spark.read.parquet(path)
        schema_path = os.path.join(path, "_orca_metadata")

        j_str = open_text(schema_path)[0]

        schema = decode_schema(j_str)

        rdd = df.rdd.map(lambda r: row_to_dict(schema, r))
        return rdd, schema

    @staticmethod
    def _read_as_xshards(path):
        rdd, schema = ParquetDataset._read_as_dict_rdd(path)

        def merge_records(schema, iter):

            l = list(iter)
            result = {}
            for k in schema.keys():
                result[k] = []
            for i, rec in enumerate(l):

                for k in schema.keys():
                    result[k].append(rec[k])
            return [result]

        result_rdd = rdd.mapPartitions(lambda iter: merge_records(schema, iter))
        xshards = SparkXShards(result_rdd)
        return xshards


    @staticmethod
    def read_as_tf(path):
        """
        return a orca.data.tf.data.Dataset
        :param path: 
        :return: 
        """
        from zoo.orca.data.tf.data import Dataset
        xshards = ParquetDataset._read_as_xshards(path)
        return Dataset.from_tensor_slices(xshards)

    @staticmethod
    def read_as_torch(path):
        """
        return a orca.data.torch.data.DataLoader
        :param path: 
        :return: 
        """

def write_from_image_directory(directory, label_map, output_path,
                               shuffle=True,
                               **kwargs):
    labels = os.listdir(directory)
    valid_labels = [label for label in labels if label in label_map]
    generator = []
    for label in valid_labels:
        label_path = os.path.join(directory, label)
        images = os.listdir(label_path)
        for image in images:
            image_path = os.path.join(label_path, image)
            generator.append({"image": image_path,
                              "label": label_map[label],
                              "image_id": image_path,
                              "label_str": label})
    if shuffle:
        random.shuffle(generator)

    schema = {"image": SchemaField(feature_type=FeatureType.IMAGE,
                                   dtype=DType.BYTES,
                                   shape=()),
              "label": SchemaField(feature_type=FeatureType.SCALAR,
                                   dtype=DType.INT32,
                                   shape=()),
              "image_id": SchemaField(feature_type=FeatureType.SCALAR,
                                      dtype=DType.STRING,
                                      shape=()),
              "label_str": SchemaField(feature_type=FeatureType.SCALAR,
                                       dtype=DType.STRING,
                                       shape=())}

    ParquetDataset.write(output_path, generator, schema, **kwargs)

def write_ndarrays(images, labels, output_path, **kwargs):
    schema = {
        "image": SchemaField(feature_type=FeatureType.NDARRAY,
                             dtype=ndarray_dtype_to_dtype(images.dtype),
                             shape=images.shape[1:])
        "labels": SchemaField(feature_type=FeatureType.NDARRAY,
                              dtype=ndarray_dtype_to_dtype(labels.dtype),
                              shape=labels[1:])
    }

    def make_generator():
        for i in range(images.shape[0]):
            yield {"image": images[i], "labels": labels[i]}

    ParquetDataset.write(output_path, make_generator(), schema, **kwargs)
