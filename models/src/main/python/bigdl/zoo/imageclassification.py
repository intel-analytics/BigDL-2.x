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
from bigdl.transform.vision.image import *

if sys.version >= '3':
    long = int
    unicode = str

class LabelOutput(FeatureTransformer):
    """
    Label Output tensor with corresponding real labels on specific dataset
    clses is the key in ImgFeature where you want to store all sorted mapped labels
    probs is the key in ImgFeature where you want to store all the sorted probilities for each class
    """
    def __init__(self, label_map, clses, probs, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), label_map, clses, probs)