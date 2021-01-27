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


import numpy as np
import base64
from zoo.serving.client import InputQueue, OutputQueue


class TestSerialization:
    def test_encode(self):
        input_api = InputQueue()
        b64 = input_api.data_to_b64(t1=np.array([1, 2]), t2=np.array([3, 4]))
        byte = base64.b64decode(b64)

