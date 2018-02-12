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

class UnmodeDetection(FeatureTransformer):
    def __init__(self, bigdl_type="float"):
        super(UnmodeDetection, self).__init__(bigdl_type)


class ImageMeta(FeatureTransformer):
    def __init__(self, class_num, bigdl_type="float"):
        super(ImageMeta, self).__init__(bigdl_type, class_num)


class Visualizer(FeatureTransformer):

    def __init__(self, label_map, thresh = 0.3, encoding = "png",
                 bigdl_type="float"):
        super(Visualizer, self).__init__(bigdl_type, label_map, thresh, encoding)

def read_coco_label_map():
    """
    load coco label map
    """
    return callBigDlFunc("float", "readCocoLabelMap")
