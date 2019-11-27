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

from bigdl.transform.vision.image import FeatureTransformer
from zoo.models.image.common.image_model import ImageModel
from zoo.feature.image.imageset import *
from zoo.common.utils import callZooFunc

if sys.version >= '3':
    long = int
    unicode = str


def read_imagenet_label_map():
    """
    load imagenet label map
    """
    return callZooFunc("float", "readImagenetLabelMap")


class ImageClassifier(ImageModel):
    """
    A pre-trained image classifier model.

    :param model_path The path containing the pre-trained model
    """

    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        super(ImageClassifier, self).__init__(None, bigdl_type)

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing object detection model (with weights).

        # Arguments
        path: The path to save the model. Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        """
        jmodel = callZooFunc(bigdl_type, "loadImageClassifier", path, weight_path)
        model = ImageModel._do_load(jmodel, bigdl_type)
        model.__class__ = ImageClassifier
        return model


class LabelOutput(FeatureTransformer):
    """
    Label Output tensor with corresponding real labels on specific dataset
    clses is the key in ImgFeature where you want to store all sorted mapped labels
    probs is the key in ImgFeature where you want to store all the sorted probilities for each class
    """

    def __init__(self, label_map, clses, probs, bigdl_type="float"):
        self.value = callZooFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), label_map, clses, probs)
