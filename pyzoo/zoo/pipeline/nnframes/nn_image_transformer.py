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
from bigdl.util.common import JavaValue, callBigDlFunc
from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaTransformer

if sys.version >= '3':
    long = int
    unicode = str

class NNImageTransformer(JavaTransformer, HasInputCol, HasOutputCol, JavaValue):
    """
    Provides DataFrame-based API for image pre-processing and feature transformation.
    NNImageTransformer follows the Spark Transformer API pattern and can be used as one stage
    in Spark ML pipeline.

    The input column can be either NNImageSchema.byteSchema or NNImageSchema.floatSchema. If
    using NNImageReader, the default format is NNImageSchema.byteSchema.
    The output column of NNImageTransformer is always NNImageSchema.floatSchema.
    """

    def __init__(self,  transformer, jvalue=None, bigdl_type="float"):
        super(NNImageTransformer, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), transformer)
        self._java_obj = self.value
        self.bigdl_type = bigdl_type
