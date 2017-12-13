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
from bigdl.transform.vision.image import ImageFrame

if sys.version >= '3':
    long = int
    unicode = str

class Predictor(JavaValue):

    @classmethod
    def predict(cls, model, image_frame, output_layer=None,
                share_buffer=False, predict_key="predict", bigdl_type="float"):
        res = callBigDlFunc(bigdl_type, "modelZooPredict", model,
                             image_frame, output_layer, share_buffer, predict_key)
        return ImageFrame(res)