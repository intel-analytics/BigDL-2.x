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

import cv2
from optparse import OptionParser

from bigdl.nn.layer import Sequential, Transpose, Contiguous
from bigdl.transform.vision.image import *
from zoo.pipeline.api.net import TFNet
from zoo.common.nncontext import get_nncontext
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.models.image.objectdetection import *


class ImageFirstDetection(FeatureTransformer):
    def __init__(self, bigdl_type="float"):
        super(ImageFirstDetection, self).__init__(bigdl_type)


def to_image_set(image_frame):
    return callBigDlFunc("float", "imageFrameToImageSet", image_frame)


def predict(model_path, label_path,  img_path, output_path, partition_num=4):
    inputs = "ToFloat:0"
    outputs = ["num_detections:0", "detection_boxes:0",
               "detection_scores:0", "detection_classes:0"]
    detector = TFNet(model_path, inputs, outputs)
    model = Sequential()
    model.add(Transpose([(2, 4), (2, 3)]))
    model.add(Contiguous())
    model.add(detector)
    image_set = ImageSet.read(img_path, sc, partition_num)
    transformer = ChainedPreprocessing([ImageResize(256, 256), ImageMatToTensor(),
                                        ImageSetToSample()])
    transformed_image_set = image_set.transform(transformer)
    output = model.predict_image(transformed_image_set.to_image_frame(), batch_per_partition=1)
    prediction = output.transform(Pipeline([ImageFirstDetection(), ScaleDetection()]))
    label_map = {}
    fp = open(label_path, "r")
    for i, line in enumerate(fp):
        label_map[i] = line.strip()
    visualizer = Visualizer(label_map, encoding="jpg")
    visualized = visualizer(to_image_set(prediction)).get_image(to_chw=False).collect()
    for img_id in range(len(visualized)):
        cv2.imwrite(output_path + '/' + str(img_id) + '.jpg', visualized[img_id])


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored")
    parser.add_option("--label", type=str, dest="label_path",
                      help="The path where the label text is stored")
    parser.add_option("--model", type=str, dest="model_path",
                      help="The path of the TensorFlow object detection model")
    parser.add_option("--output_path", type=str, dest="output_path",
                      help="The path to store the detection results")
    parser.add_option("--partition_num", type=int, dest="partition_num", default=4)
    (options, args) = parser.parse_args(sys.argv)

    sc = get_nncontext("TFNet Object Detection Example")

    predict(options.model_path, options.label_path,
            options.img_path, options.output_path,
            options.partition_num)
