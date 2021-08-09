import shutil
import tempfile

from pyspark import SparkContext
from pyspark.sql import SparkSession

from zoo import init_nncontext
from zoo.orca.data import SparkXShards
from zoo.orca.data.file import open_text, write_text
from zoo.orca.data.image.utils import chunks, dict_to_row, row_to_dict, encode_schema, \
    decode_schema, SchemaField, FeatureType, DType, ndarray_dtype_to_dtype, \
    decode_feature_type_ndarray, pa_fs
from zoo.orca.data.image.voc_dataset import VOCDatasets
from bigdl.util.common import get_node_and_core_number
import os
import numpy as np
import random
import pyarrow.parquet as pq
import io
import math
import torch
from zoo.orca.data.image.parquet_dataset import read_parquet, write_parquet, ParquetDataset

#class_map = {name: idx for idx, name in enumerate(
#         open("/home/joy/workspace/colab/yolov3-tf2/data/coco.names").read().splitlines())}
#write_parquet(format="voc", voc_root_path="/home/joy/workspace/colab/yolov3-tf2/data/voc2009_raw/VOCdevkit",
#               output_path="./train_dataset",
#               splits_names=[("2009", "trainval")], classes=class_map)

path = "./train_dataset"
path, _ = pa_fs(path)
import tensorflow as tf

# schema_path = os.path.join(path, "_orca_metadata")
# j_str = open_text(schema_path)[0]
# schema = decode_schema(j_str)
#
# row_group = []
#
# for root, dirs, files in os.walk(path):
#     for name in dirs:
#         if name.startswith("chunk="):
#             chunk_path = os.path.join(path, name)
#             row_group.append(chunk_path)

#print(row_group)
# ['./train_dataset/chunk=5', './train_dataset/chunk=6', './train_dataset/chunk=2', './train_dataset/chunk=3', './train_dataset/chunk=7', './train_dataset/chunk=1', './train_dataset/chunk=0', './train_dataset/chunk=4']
#row_group.sort()
#filter_row_group_indexed = list(range(len(row_group)))
#print(filter_row_group_indexed)
#[0, 1, 2, 3, 4, 5, 6, 7]

# num_shards = 4
# rank = 0
# filter_row_group_indexed = [index for index in list(range(len(row_group)))
#                                             if index % num_shards == rank]
# print(filter_row_group_indexed)
#
# data_record = []
# for select_chunk_path in [row_group[i] for i in filter_row_group_indexed]:
#     pq_table = pq.read_table(select_chunk_path)
#     df = decode_feature_type_ndarray(pq_table.to_pandas(), schema)
#     # print("df", df) image col, image_id col, label col
#     data_record.extend(df.to_dict("records"))
# print("len", len(data_record))  #7054
#print("data_record", data_record[0]) {'image': byte, 'image_id': path, 'label': array[[6.00,..]]}

# cur = 0
# cur_tail = len(data_record)
# for i in range(cur_tail):
#     elem = data_record[i]
#     print(elem)

# def read_tfdataset(path, output_types, config=None, output_shapes=None, *args, **kwargs):
#     """
#     return a orca.data.tf.data.Dataset
#     :param path:
#     :return:
#     """
#     path, _ = pa_fs(path)
#     import tensorflow as tf
#
#     schema_path = os.path.join(path, "_orca_metadata")
#     j_str = open_text(schema_path)[0]
#     schema = decode_schema(j_str)
#
#     row_group = []
#
#     for root, dirs, files in os.walk(path):
#         for name in dirs:
#             if name.startswith("chunk="):
#                 chunk_path = os.path.join(path, name)
#                 row_group.append(chunk_path)
#
#     class ParquetIterableDataset:
#         def __init__(self, row_group, num_shards=None,
#                      rank=None):
#             #super(ParquetDataset).__init__()
#             self.row_group = row_group
#
#             # To get the indices we expect
#             self.row_group.sort()
#
#             self.num_shards = num_shards
#             self.rank = rank
#             self.datapiece = None
#
#             filter_row_group_indexed = []
#
#             if not self.num_shards or not self.rank:
#                 filter_row_group_indexed = list(range(len(self.row_group)))
#             else:
#                 assert self.num_shards <= len(
#                     self.row_group), "num_shards should be not larger than partitions." \
#                                      "but got num_shards {} with partitions {}." \
#                     .format(self.num_shards, len(self.row_group))
#                 assert self.rank < self.num_shards, \
#                     "shard index should be included in [0,num_shard)," \
#                     "but got rank {} with num_shard {}.".format(
#                         self.rank, self.num_shards)
#                 filter_row_group_indexed = [index for index in list(range(len(self.row_group)))
#                                             if index % self.num_shards == self.rank]
#
#             data_record = []
#             for select_chunk_path in [self.row_group[i] for i in filter_row_group_indexed]:
#                 pq_table = pq.read_table(select_chunk_path)
#                 df = decode_feature_type_ndarray(pq_table.to_pandas(), schema)
#                 data_record.extend(df.to_dict("records"))
#
#             self.datapiece = data_record
#             self.cur = 0
#             self.cur_tail = len(self.datapiece)
#
#         def __iter__(self):
#             return self
#
#         def __next__(self):
#             # move iter here so we can do transforms
#             if self.cur < self.cur_tail:
#                 elem = self.datapiece[self.cur]
#                 self.cur += 1
#                 return elem
#             else:
#                 raise StopIteration
#
#         def __call__(self):
#             self.cur = 0
#             return self
#
#     # def generator():
#     #     for root, dirs, files in os.walk(path):
#     #         for name in dirs:
#     #             if name.startswith("chunk="):
#     #                 chunk_path = os.path.join(path, name)
#     #                 pq_table = pq.read_table(chunk_path)
#     #                 df = decode_feature_type_ndarray(
#     #                     pq_table.to_pandas(), schema)
#     #                 for record in df.to_dict("records"):
#     #                     yield record
#     #
#     # dataset = tf.data.Dataset.from_generator(generator, output_types=output_types,
#     #                                          output_shapes=output_shapes)
#     dataset = ParquetIterableDataset(
#         row_group=row_group, num_shards=config.get("num_shards"),
#         rank=config.get("rank"))
#
#     return tf.data.Dataset.from_generator(dataset, output_types=output_types,
#                                              output_shapes=output_shapes)

#tf.enable_eager_execution()
output_types = {"image": tf.string, "label": tf.float32, "image_id": tf.string}
output_shapes = {"image": (), "label": (None, 5), "image_id": ()}
#dataset = read_as_tfdataset(path,  config={}, output_types=output_types,
#                  output_shapes=output_shapes)
#dataset = read_parquet("tf_dataset", path=path, output_types=output_types)

## second part
resource_path = "/home/joy/workspace/imageshard/analytics-zoo/pyzoo/test/zoo/resources"
WIDTH, HEIGHT, NUM_CHANNELS = 224, 224, 3

def images_generator():
    dataset_path = os.path.join(resource_path, "cat_dog")
    for root, dirs, files in os.walk(os.path.join(dataset_path, "cats")):
        for name in files:
            image_path = os.path.join(root, name)
            yield {"image": image_path, "label": 1, "id": image_path}

    for root, dirs, files in os.walk(os.path.join(dataset_path, "dogs")):
        for name in files:
            image_path = os.path.join(root, name)
            yield {"image": image_path, "label": 0, "id": image_path}


images_schema = {
    "image": SchemaField(feature_type=FeatureType.IMAGE, dtype=DType.FLOAT32, shape=()),
    "label": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.FLOAT32, shape=()),
    "id": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.STRING, shape=())
}

temp_dir = tempfile.mkdtemp()

try:
    ParquetDataset.write("file://" + temp_dir, images_generator(),
                                 images_schema, block_size=4)
    path = "file://" + temp_dir
    print("path")
    output_types = {"id": tf.string, "image": tf.string, "label": tf.float32}
    from parquet_dataset import read_as_tfdataset

    dataset1 = read_as_tfdataset(path=path, config={},
                             output_types=output_types)
    print("len1", len(list(dataset1)))
    dataset2 = read_as_tfdataset(path=path, config={"num_shards": 3, "rank": 1}, output_types=output_types)
    print("len2", len(list(dataset2)))
    for dt in dataset1.take(1):
        print(dt.keys())
    #dataset = read_as_tfdataset(path=path, config={}, output_types=output_types)

    from parquet_dataset import read_parquet
    dataloader = read_parquet("dataloader", path=path)
    cur_dl = iter(dataloader)

    while True:
        try:
            print(next(cur_dl)['label'])
        except StopIteration:
            break

    dataset3 = read_parquet("dataloader", config={"num_shards": 3, "rank": 0}, path=path)
    cur_dl = iter(dataset3)
    count = 0

    while True:
        try:
            print(next(cur_dl)['label'])
            count += 1
        except StopIteration:
            break
    print("count", count)
finally:
    shutil.rmtree(temp_dir)
# temp_dir = tempfile.mkdtemp()
# ParquetDataset.write("file://" + temp_dir, images_generator(),
#                                  images_schema, block_size=2)
# path = "file://" + temp_dir
# print("path", path)
