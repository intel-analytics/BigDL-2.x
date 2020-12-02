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
from io import BytesIO

import numpy as np
from pyspark.sql import SparkSession
from zoo.orca.data.image.parquet_dataset import ParquetDataset
from zoo.orca.data.image.utils import DType, FeatureType, SchemaField


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
        data, schema = ParquetDataset._read_as_dict_rdd(temp_dir).collect()[0]
        assert data['id'] == 0
        assert np.all(data['feature'] == np.zeros((10,), dtype=np.float32))
        assert np.all(data['label'] == np.ones((4,), dtype=np.float32))

    finally:
        shutil.rmtree(temp_dir)
