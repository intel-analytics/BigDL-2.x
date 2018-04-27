#
# Copyright 2016 The BigDL Authors.
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
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *

if sys.version >= '3':
    long = int
    unicode = str


class NNFeatureTransformer(JavaValue):
    """
    NNFeatureTransformer is a transformer that transform
    """
    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
                bigdl_type, JavaValue.jvm_class_constructor(self), *args)

class NumToTensor(NNFeatureTransformer):
    def __init__(self, bigdl_type="float"):
        super(NumToTensor, self).__init__(bigdl_type)

class SeqToTensor(NNFeatureTransformer):
    def __init__(self, size, bigdl_type="float"):
        super(SeqToTensor, self).__init__(bigdl_type, size)

class MLlibVectorToTensor(NNFeatureTransformer):
    def __init__(self, size, bigdl_type="float"):
        super(MLlibVectorToTensor, self).__init__(bigdl_type, size)
