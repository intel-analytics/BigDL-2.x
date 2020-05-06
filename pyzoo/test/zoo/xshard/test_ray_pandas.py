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

import os.path

import pytest
import ray

import zoo.xshard.pandas
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from test.zoo.xshard.conftest import get_ray_ctx


class TestDataShards(ZooTestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        self.ray_ctx = get_ray_ctx()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        pass

    def test_read_local_csv(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_csv(file_path, self.ray_ctx)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "location" in df.columns, "location is not in columns"

    def test_read_local_json(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_json(file_path, self.ray_ctx, orient='columns', lines=True)
        data = data_shard.collect()
        assert len(data) == 2, "number of shard should be 2"
        df = data[0]
        assert "value" in df.columns, "value is not in columns"

    def test_read_s3(self):
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key_id and secret_access_key:
            file_path = "s3://analytics-zoo-data/nyc_taxi.csv"
            data_shard = zoo.xshard.pandas.read_csv(file_path, self.ray_ctx)
            data = data_shard.collect()
            df = data[0]
            assert "value" in df.columns, "value is not in columns"

    def test_repartition(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_json(file_path, self.ray_ctx)
        partitions1 = data_shard.get_partitions()
        assert len(partitions1) == 2, "number of partition should be 2"
        data_shard.repartition(1)
        partitions2 = data_shard.get_partitions()
        assert len(partitions2) == 1, "number of partition should be 1"
        partition_data = ray.get(partitions2[0].get_data())
        assert len(partition_data) == 2, "partition 0 should have 2 objects"

    def test_apply(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_json(file_path, self.ray_ctx, orient='columns', lines=True)
        data = data_shard.collect()
        assert data[0]["value"].values[0] > 0, "value should be positive"

        def negative(df, column_name):
            df[column_name] = df[column_name] * (-1)
            return df

        data_shard.transform_shard(negative, "value")
        data2 = data_shard.collect()
        assert data2[0]["value"].values[0] < 0, "value should be negative"

    def test_max_single_index(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_csv(file_path, self.ray_ctx)
        max = data_shard.max()
        assert max.loc['ID'] == 101472
        assert max.loc['sale_price'] == 475000
        assert max.loc['location'] == 130

    def test_max_multi_index(self):
        import tempfile
        import pandas as pd
        idx = pd.MultiIndex.from_arrays([
            ['warm', 'warm', 'cold', 'cold'],
            ['dog', 'falcon', 'fish', 'spider']],
            names=['blooded', 'animal'])
        df1 = pd.DataFrame({'legs':[4, 2, 0, 8]}, index=idx)
        dir = tempfile.TemporaryDirectory()
        file_path_1 = os.path.join(dir.name, "animal1.csv")
        df1.to_csv(file_path_1)

        idx = pd.MultiIndex.from_arrays([
            ['warm', 'warm', 'cold', 'cold', 'neutral'],
            ['chicken', 'cat', 'dolphin', 'bee', 'dinosaur']],
            names=['blooded', 'animal'])
        df2 = pd.DataFrame({'legs': [2, 4, 0, 6, 4]}, index=idx)
        file_path_2 = os.path.join(dir.name, "animal2.csv")
        df2.to_csv(file_path_2)
        data_shard = zoo.xshard.pandas.read_csv(dir.name, self.ray_ctx, index_col=['blooded', 'animal'])
        max = data_shard.max(level='blooded')
        assert max.loc['warm', 'legs'] == 4
        assert max.loc['cold', 'legs'] == 8
        assert max.loc['neutral', 'legs'] == 4
        os.remove(file_path_1)
        os.remove(file_path_2)
        dir.cleanup()

    def test_min_single_index(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_csv(file_path, self.ray_ctx)
        min = data_shard.min()
        assert min.loc['ID'] == 100365
        assert min.loc['sale_price'] == 150000
        assert min.loc['location'] == 100

    def test_min_multi_index(self):
        import tempfile
        import pandas as pd
        idx = pd.MultiIndex.from_arrays([
            ['warm', 'warm', 'cold', 'cold'],
            ['dog', 'falcon', 'fish', 'spider']],
            names=['blooded', 'animal'])
        df1 = pd.DataFrame({'legs':[4, 2, 0, 8]}, index=idx)
        dir = tempfile.TemporaryDirectory()
        file_path_1 = os.path.join(dir.name, "animal1.csv")
        df1.to_csv(file_path_1)

        idx = pd.MultiIndex.from_arrays([
            ['warm', 'warm', 'cold', 'cold', 'neutral'],
            ['chicken', 'cat', 'dolphin', 'bee', 'dinosaur']],
            names=['blooded', 'animal'])
        df2 = pd.DataFrame({'legs': [2, 4, 0, 6, 4]}, index=idx)
        file_path_2 = os.path.join(dir.name, "animal2.csv")
        df2.to_csv(file_path_2)
        data_shard = zoo.xshard.pandas.read_csv(dir.name, self.ray_ctx, index_col=['blooded', 'animal'])
        min = data_shard.min(level='blooded')
        assert min.loc['warm', 'legs'] == 2
        assert min.loc['cold', 'legs'] == 0
        assert min.loc['neutral', 'legs'] == 4
        os.remove(file_path_1)
        os.remove(file_path_2)
        dir.cleanup()

    def test_get_column(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_csv(file_path, self.ray_ctx)
        data_shard_2 = data_shard['ID']
        max = data_shard_2.max()
        assert max == 101472

    def test_get_slice(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_csv(file_path, self.ray_ctx)
        data_shard_2 = data_shard[:2]
        max = data_shard_2.max()
        assert max.loc['ID'] == 101388
        assert max.loc['sale_price'] == 475000
        assert max.loc['location'] == 130

    def test_split(self):
        file_path = os.path.join(self.resource_path, "xshard")
        data_shard = zoo.xshard.pandas.read_csv(file_path, self.ray_ctx)
        train, test = data_shard.train_test_split(test_size=0.2)
        train_data = train.collect()
        test_data = test.collect()


if __name__ == "__main__":
    pytest.main([__file__])
