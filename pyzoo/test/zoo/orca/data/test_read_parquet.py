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

import pytest
from unittest import TestCase
import os
from zoo.orca.data.image.parquet_dataset import ParquetDataset, read_parquet
from zoo.orca.data.image.utils import DType, FeatureType, SchemaField
import tensorflow as tf

resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")


class TestSparkBackend(TestCase):
    def test_read_parquet_images_tf_dataset(orca_context_fixture):
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
            "image": SchemaField(feature_type=FeatureType.IMAGE, dtype=DType.FLOAT32, shape=()),
            "label": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.FLOAT32, shape=()),
            "id": SchemaField(feature_type=FeatureType.SCALAR, dtype=DType.STRING, shape=())
        }

        try:
            ParquetDataset.write("file://" + temp_dir, generator(), schema)
            path = "file://" + temp_dir
            output_types = {"id": tf.string, "image": tf.string, "label": tf.float32}
            dataset = read_parquet("tf_dataset", input_path=path,
                               output_types=output_types)
            for dt in dataset:
                print(dt.keys())

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
