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

import numpy as np
import pytest

from zoo import init_spark_on_local
np.random.seed(1337)  # for reproducibility


class TestSparkOnLocal():
    def hostname(self):
        import socket
        return socket.gethostname()


class TestUtil(TestCase):

    def test_local(self):
        node_num = 4
        sc = init_spark_on_local(cores=node_num)
        sc.stop()

if __name__ == "__main__":
    pytest.main([__file__])
