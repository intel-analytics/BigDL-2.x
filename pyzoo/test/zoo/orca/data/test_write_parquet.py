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

import tempfile
import shutil

import numpy as np
import os
from pyspark.sql import SparkSession
from zoo.orca.data.image.parquet_dataset import ParquetDataset
from zoo.orca.data.image.utils import DType, FeatureType, SchemaField


resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")


def test_write_parquet_simple(orca_context_fixture):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()

    def generator(num):
        for i in range(num):
            yield {"id": i, "feature": np.zeros((10,)), "label": np.ones((4,))}

    schema = {
        "id": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.INT32, shape=()),
        "feature": SchemaField(feature_type=FeatureType.NDARRAY, dtype=DType.FLOAT32, shape=(10,)),
        "label": SchemaField(feature_type=FeatureType.NDARRAY, dtype=DType.FLOAT32, shape=(4,))
    }

    try:

        ParquetDataset.write(temp_dir, generator(100), schema)
        data, schema = ParquetDataset._read_as_dict_rdd(temp_dir)
        data = data.collect()[0]
        assert data['id'] == 0
        assert np.all(data['feature'] == np.zeros((10,), dtype=np.float32))
        assert np.all(data['label'] == np.ones((4,), dtype=np.float32))

    finally:
        shutil.rmtree(temp_dir)


def test_write_parquet_images(orca_context_fixture):

    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()

    def generator():
        dataset_path = os.path.join(resource_path, "cat_dog")
        for root, dirs, files in os.walk(os.path.join(dataset_path, "cats")):
            for name in files:
                image_path = os.path.join(root, name)
                yield {"image": image_path, "label": 1, "id": image_path}

        for root, dirs, files in os.walk(os.path.join(dataset_path, "dogs")):
            for name in files:
                image_path = os.path.join(root, name)
                yield {"image": image_path, "label": 0, "id": image_path}

    schema = {
        "image": SchemaField(feature_type=FeatureType.IMAGE, dtype=DType.FLOAT32, shape=(10,)),
        "label": SchemaField(feature_type=FeatureType.NDARRAY, dtype=DType.FLOAT32, shape=(4,)),
        "id": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.STRING, shape=())
    }

    try:
        ParquetDataset.write(temp_dir, generator(), schema)
        data, schema = ParquetDataset._read_as_dict_rdd(temp_dir)
        data = data.collect()[0]
        image_path = data['id']
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        assert image_bytes == data['image']

    finally:
        shutil.rmtree(temp_dir)
