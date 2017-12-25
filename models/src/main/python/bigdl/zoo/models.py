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

class Configure(JavaValue):
    """
    predictor configure
    :param pre_processor preprocessor of ImageFrame before model inference
    :param post_processor postprocessor of ImageFrame after model inference
    :param batch_per_partition batch size per partition
    :param label_map mapping from prediction result indexes to real dataset labels
    """
    def __init__(self, pre_processor=None,
                 post_processor=None,
                 batch_per_partition=4,
                 label_map=None, jvalue=None, bigdl_type="float"):
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
                label_map)

    def label_map(self):
        return callBigDlFunc(self.bigdl_type, "getLabelMap", self.value)

class Visualizer(FeatureTransformer):
    """
    Visualizer is a transformer to visualize the detection results
    (tensors that encodes label, score, boundingbox)
    You can call image_frame.get_image() to get the visualized results
    """
    def __init__(self, label_map, thresh = 0.3, encoding = "png",
                 bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), label_map, thresh, encoding)
        
def read_pascal_label_map():
    """
    load pascal label map
    """
    return callBigDlFunc("float", "readPascalLabelMap")

def read_coco_label_map():
    """
    load coco label map
    """
    return callBigDlFunc("float", "readCocoLabelMap")

class ImInfo(FeatureTransformer):
    """
    Generate imInfo
    imInfo is a tensor that contains height, width, scaleInHeight, scaleInWidth
    """
    def __init__(self, bigdl_type="float"):
        super(ImInfo, self).__init__(bigdl_type)

class DecodeOutput(FeatureTransformer):
    """
    Decode the detection output
    The output of the model prediction is a 1-dim tensor
    The first element of tensor is the number(K) of objects detected,
    followed by [label score x1 y1 x2 y2] X K
    For example, if there are 2 detected objects, then K = 2, the tensor may
    looks like
    ```2, 1, 0.5, 10, 20, 50, 80, 3, 0.3, 20, 10, 40, 70```
    After decoding, it returns a 2-dim tensor, each row represents a detected object
    ```
    1, 0.5, 10, 20, 50, 80
    3, 0.3, 20, 10, 40, 70
    ```
    """
    def __init__(self, bigdl_type="float"):
        super(DecodeOutput, self).__init__(bigdl_type)

class ScaleDetection(FeatureTransformer):
    """
    If the detection is normalized, for example, ssd detected bounding box is in [0, 1],
    need to scale the bbox according to the original image size.
    Note that in this transformer, the tensor from model output will be decoded,
    just like `DecodeOutput`
    """
    def __init__(self, bigdl_type="float"):
        super(ScaleDetection, self).__init__(bigdl_type)