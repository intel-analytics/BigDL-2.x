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

from pyspark import SparkContext
from pyspark.sql import SparkSession

from zoo import init_nncontext
from zoo.orca.data import SparkXShards
from zoo.orca.data.file import open_text, write_text
from zoo.orca.data.image.utils import chunks, dict_to_row, row_to_dict, encode_schema, \
    decode_schema, SchemaField, FeatureType, DType, ndarray_dtype_to_dtype
from zoo.orca.data.image.voc_dataset import VOCDatasets
from bigdl.util.common import get_node_and_core_number
import os
import numpy as np
import random
import io


class ParquetDataset:
    @staticmethod
    def write(path, generator, schema, block_size=1000, write_mode="overwrite", **kwargs):
        """
        Take each record in the generator and write it to a parquet file.

        **generator**
        Each record in the generator is a dict, the key is a string and will be the
        column name of saved parquet record and the value is the data.

        **schema**
        schema defines the name, dtype, shape of a column, as well as the feature
        type of a column. The feature type, defines how to encode and decode the column value.

        There are three kinds of feature type:
        1. Scalar, such as a int or float number, or a string, which can be directly mapped
           to a parquet type
        2. NDarray, which takes a np.ndarray and save it serialized bytes. The corresponding
           parquet type is BYTE_ARRAY .
        3. Image, which takes a string representing a image file in local file system and save
           the raw file content bytes.
           The corresponding parquet type is BYTE_ARRAY.

        :param path: the output path, e.g. file:///output/path, hdfs:///output/path
        :param generator: generate a dict, whose key is a string and value is one of
                          (a scalar value, ndarray, image file path)
        :param schema: a dict, whose key is a string, value is one of
                      (schema_field.Scalar, schema_field.NDarray, schema_field.Image)
        :param kwargs: other args
        """

        sc = init_nncontext()
        spark = SparkSession(sc)
        node_num, core_num = get_node_and_core_number()
        for i, chunk in enumerate(chunks(generator, block_size)):
            chunk_path = os.path.join(path, f"chunk={i}")
            rows_rdd = sc.parallelize(chunk, core_num * node_num) \
                .map(lambda x: dict_to_row(schema, x))
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
            for k, v in schema.items():
                if not v.feature_type == FeatureType.IMAGE:
                    result[k] = np.stack(result[k])

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
        raise NotImplementedError()


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_mnist_images(image_filepath):
    with open(image_filepath, "rb") as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, image_filepath))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def _extract_mnist_labels(labels_filepath):
    with open(labels_filepath, "rb") as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, labels_filepath.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def write_from_directory(directory, label_map, output_path, shuffle=True, **kwargs):
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
                                   dtype=DType.FLOAT32,
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


def _write_ndarrays(images, labels, output_path, **kwargs):
    images_shape = [int(x) for x in images.shape[1:]]
    labels_shape = [int(x) for x in labels.shape[1:]]
    schema = {
        "image": SchemaField(feature_type=FeatureType.NDARRAY,
                             dtype=ndarray_dtype_to_dtype(images.dtype),
                             shape=images_shape),
        "label": SchemaField(feature_type=FeatureType.NDARRAY,
                             dtype=ndarray_dtype_to_dtype(labels.dtype),
                             shape=labels_shape)
    }

    def make_generator():
        for i in range(images.shape[0]):
            yield {"image": images[i], "label": labels[i]}

    ParquetDataset.write(output_path, make_generator(), schema, **kwargs)


def write_mnist(image_file, label_file, output_path, **kwargs):
    images = _extract_mnist_images(image_filepath=image_file)
    labels = _extract_mnist_labels(labels_filepath=label_file)
    _write_ndarrays(images, labels, output_path, **kwargs)


def write_voc(voc_root_path, splits_names, output_path, **kwargs):
    custom_classes = kwargs.get("classes", None)
    voc_datasets = VOCDatasets(voc_root_path, splits_names, classes=custom_classes)

    def make_generator():
        for img_path, label in voc_datasets:
            yield {"image": img_path, "label": label, "image_id": img_path}

    image, label = voc_datasets[0]
    label_shape = (-1, label.shape[-1])
    schema = {
        "image": SchemaField(feature_type=FeatureType.IMAGE,
                             dtype=DType.FLOAT32,
                             shape=()),
        "label": SchemaField(feature_type=FeatureType.NDARRAY,
                             dtype=ndarray_dtype_to_dtype(label.dtype),
                             shape=label_shape),
        "image_id": SchemaField(feature_type=FeatureType.SCALAR,
                                dtype=DType.STRING,
                                shape=())
    }
    kwargs = {key: value for key, value in kwargs.items() if key not in ["classes"]}
    ParquetDataset.write(output_path, make_generator(), schema, **kwargs)


def _check_arguments(_format, kwargs, args):
    for keyword in args:
        assert keyword in kwargs, keyword + " is not specified for format " + _format + "."


def write_parquet(format, output_path, *args, **kwargs):
    supported_format = {"mnist", "image_folder", "voc"}
    if format not in supported_format:
        raise ValueError(format + " is not supported, should be one of 'mnist',"
                         "'image_folder' and 'voc'.")

    format_to_function = {"mnist": (write_mnist, ["image_file", "label_file"]),
                          "image_folder": (write_from_directory, ["directory", "label_map"]),
                          "voc": (write_voc, ["voc_root_path", "splits_names"])}
    func, required_args = format_to_function[format]
    _check_arguments(format, kwargs, required_args)
    func(output_path=output_path, *args, **kwargs)
