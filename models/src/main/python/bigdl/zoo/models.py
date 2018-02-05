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

class Predictor(JavaValue):
    """
    Predictor for BigDL models
    :param model BigDL model
    :param configure configure includes preprocessor, postprocessor, batch size, label mapping
                     models from BigDL model zoo have their default configures
                     if you want to predict over your own model, or if you want to change the
                     default configure, you can pass in a user-defined configure
    """
    def __init__(self, model, configure=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self),
            model,
            configure)
        self.configure = Configure(jvalue=callBigDlFunc(self.bigdl_type, "getConfigure", self.value))

    def predict(self, image_frame, output_layer=None,
                share_buffer=False, predict_key="predict"):
        res = callBigDlFunc(self.bigdl_type, "modelZooPredict", self.value,
                             image_frame, output_layer, share_buffer, predict_key)
        return ImageFrame(res)

    def label_map(self):
        return self.configure.label_map()

class Configure(JavaValue):
    """
    predictor configure
    :param pre_processor preprocessor of ImageFrame before model inference
    :param post_processor postprocessor of ImageFrame after model inference
    :param batch_per_partition batch size per partition
    :param label_map mapping from prediction result indexes to real dataset labels
    :param feature_padding_param featurePaddingParam if the inputs have variant size
    """
    def __init__(self, pre_processor=None,
                 post_processor=None,
                 batch_per_partition=4,
                 label_map=None, feature_padding_param=None, jvalue=None, bigdl_type="float"):
        self.bigdl_type=bigdl_type
        if jvalue:
            self.value = jvalue
        else:
            if pre_processor:
                assert pre_processor.__class__.__bases__[0].__name__ == "FeatureTransformer",\
                    "the pre_processor should be subclass of FeatureTransformer"
            if post_processor:
                assert post_processor.__class__.__bases__[0].__name__ == "FeatureTransformer", \
                    "the pre_processor should be subclass of FeatureTransformer"
            self.value = callBigDlFunc(
                bigdl_type, JavaValue.jvm_class_constructor(self),
                pre_processor,
                post_processor,
                batch_per_partition,
                label_map,
                feature_padding_param)

    def label_map(self):
        return callBigDlFunc(self.bigdl_type, "getLabelMap", self.value)

class PaddingParam(JavaValue):

    def __init__(self, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self))


        