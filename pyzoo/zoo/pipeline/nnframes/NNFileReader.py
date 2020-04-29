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

import sys
from zoo.common.utils import callZooFunc

if sys.version >= '3':
    long = int
    unicode = str


class NNFileReader:
    """
    Primary DataFrame-based file loading interface, defining API to read file from files
    to DataFrame.
    """

    @staticmethod
    def readCSV(path, sc=None, bigdl_type="float"):
        df = callZooFunc(bigdl_type, "nnReadCSV", path, sc)
        df._sc._jsc = sc._jsc
        return df
