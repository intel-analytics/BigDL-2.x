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
from pyspark.sql import SparkSession
from zoo.orca.data.image.parquet_dataset import ParquetDataset


def test_write_parquet_simple(orca_context_fixture):
    sc = orca_context_fixture
    temp_dir = tempfile.mkdtemp()

    def generator(num):
        for i in range(num):
            yield {"id": i}

    schema = {
        "id": {"type": "SCALAR"}
    }

    try:

        ParquetDataset.write(temp_dir, generator(100), schema)

        spark = SparkSession(sc)

        df = spark.read.parquet(temp_dir)

        data = df.rdd.map(lambda r: r["id"]).collect()
        assert data == list(range(100))

    finally:
        shutil.rmtree(temp_dir)
