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


class NNTransformer(JavaValue):
    """
    Python wrapper for the scala transformers
    """
    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
                bigdl_type, JavaValue.jvm_class_constructor(self), *args)

class ChainedTransformer(NNTransformer):
    """
    Pipeline of FeatureTransformer
    """

    def __init__(self, transformers, bigdl_type="float"):
        for transfomer in transformers:
            assert transfomer.__class__.__bases__[0].__name__ == "NNTransformer", \
                str(transfomer) + " should be subclass of NNTransformer"
        super(ChainedTransformer, self).__init__(bigdl_type, transformers)

class NumToTensor(NNTransformer):
    def __init__(self, bigdl_type="float"):
        super(NumToTensor, self).__init__(bigdl_type)

class SeqToTensor(NNTransformer):
    def __init__(self, size, bigdl_type="float"):
        super(SeqToTensor, self).__init__(bigdl_type, size)

class MLlibVectorToTensor(NNTransformer):
    def __init__(self, size, bigdl_type="float"):
        super(MLlibVectorToTensor, self).__init__(bigdl_type, size)

class ImageFeatureToTensor(NNTransformer):
    def __init__(self, bigdl_type="float"):
        super(ImageFeatureToTensor, self).__init__(bigdl_type)

class RowToImageFeature(NNTransformer):
    def __init__(self, bigdl_type="float"):
        super(RowToImageFeature, self).__init__(bigdl_type)

class FeatureLabelTransformer(NNTransformer):
    def __init__(self, feature_transformer, label_transformer, bigdl_type="float"):
        super(FeatureLabelTransformer, self).__init__(bigdl_type, feature_transformer, label_transformer)

class TensorToSample(NNTransformer):
    def __init__(self, bigdl_type="float"):
        super(TensorToSample, self).__init__(bigdl_type)
        
class FeatureToTupleAdapter(NNTransformer):
    def __init__(self, sample_transformer, bigdl_type="float"):
        super(FeatureToTupleAdapter, self).__init__(bigdl_type, sample_transformer)

