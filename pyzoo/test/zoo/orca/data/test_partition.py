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

from unittest import TestCase
import pytest


class TestSparkBackend(TestCase):

    def test_partition_ndarray(self):
        import numpy as np
        import zoo.orca.data as orca_data
        data = np.random.randn(10, 4)

        xshards = orca_data.partition(data)

        data_parts = xshards.rdd.collect()

        reconstructed = np.concatenate(data_parts)
        assert np.allclose(data, reconstructed)

    def test_partition_tuple(self):
        import numpy as np
        import zoo.orca.data as orca_data
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = orca_data.partition((data1, data2))

        data_parts = xshards.rdd.collect()

        data1_parts = [part[0] for part in data_parts]
        data2_parts = [part[1] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_list(self):
        import numpy as np
        import zoo.orca.data as orca_data
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = orca_data.partition([data1, data2])

        data_parts = xshards.rdd.collect()

        data1_parts = [part[0] for part in data_parts]
        data2_parts = [part[1] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_dict(self):
        import numpy as np
        import zoo.orca.data as orca_data
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = orca_data.partition({"x": data1, "y": data2})

        data_parts = xshards.rdd.collect()

        data1_parts = [part["x"] for part in data_parts]
        data2_parts = [part["y"] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

    def test_partition_nested(self):
        import numpy as np
        import zoo.orca.data as orca_data
        data1 = np.random.randn(10, 4)
        data2 = np.random.randn(10, 4)

        xshards = orca_data.partition({"x": (data1, ), "y": [data2]})

        data_parts = xshards.rdd.collect()

        data1_parts = [part["x"][0] for part in data_parts]
        data2_parts = [part["y"][0] for part in data_parts]

        reconstructed1 = np.concatenate(data1_parts)
        reconstructed2 = np.concatenate(data2_parts)
        assert np.allclose(data1, reconstructed1)
        assert np.allclose(data2, reconstructed2)

if __name__ == "__main__":
    pytest.main([__file__])
