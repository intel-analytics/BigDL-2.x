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
import numpy as np
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import JTensor

if sys.version >= '3':
    long = int
    unicode = str


class FeatureTransformer(JavaValue):

    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)

    def transform(self, sample, bigdl_type="float"):
        return callBigDlFunc(bigdl_type, "transform", self.value, sample)

    def __call__(self, sample_rdd, bigdl_type="float"):
        return callBigDlFunc(bigdl_type, "transformRdd", self.value, sample_rdd)


class Pipeline(JavaValue):

    def __init__(self, transformers, bigdl_type="float"):
        self.transformer = callBigDlFunc(bigdl_type, "chainTransformer", transformers)

    def transform(self, sample, bigdl_type="float"):
        transformed = callBigDlFunc(bigdl_type, "transform", self.transformer, sample)
        return transformed[0].array.reshape(transformed[1].array)

    def __call__(self, sample_rdd, bigdl_type="float"):
        return callBigDlFunc(bigdl_type, "transformRdd", self.transformer, sample_rdd)


class Crop(FeatureTransformer):

    def __init__(self, start, patch_size, bigdl_type="float"):
        super(Crop, self).__init__(bigdl_type, start, patch_size)


class Rotate(FeatureTransformer):

    def __init__(self, rotationAngles, bigdl_type="float"):
        super(Rotate, self).__init__(bigdl_type, rotationAngles)


class AffineTransform(FeatureTransformer):

    def __init__(self, affine_mat, translation=JTensor.from_ndarray(np.zeros(3)), clamp_mode="clamp", pad_val=0.0, bigdl_type="float"):
        super(AffineTransform, self).__init__(bigdl_type, affine_mat, translation, clamp_mode, pad_val)