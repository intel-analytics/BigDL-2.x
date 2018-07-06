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

from optparse import OptionParser

from bigdl.nn.layer import Sequential, Transpose, Contiguous, SelectTable
from zoo.pipeline.api.net import TFNet
from zoo.common.nncontext import init_nncontext
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.models.image.objectdetection import *


def predict(model_path, img_path, partition_num=4):
    inputs = "ToFloat:0"
    outputs = ["num_detections:0", "detection_boxes:0",
               "detection_scores:0", "detection_classes:0"]
    model = Sequential()
    detector = TFNet(model_path, inputs, outputs)
    model.add(detector)
    # Select the detection_boxes from the output.
    model.add(SelectTable(2))
    image_set = ImageSet.read(img_path, sc, partition_num)
    transformer = ChainedPreprocessing([ImageResize(256, 256), ImageMatToTensor(format="NHWC"),
                                        ImageSetToSample()])
    transformed_image_set = image_set.transform(transformer)
    output = model.predict_image(transformed_image_set.to_image_frame(), batch_per_partition=1)
    # Print the detection box with the highest score of the first prediction result.
    result = output.get_predict().first()
    print(result[1][0])


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--model", type=str, dest="model_path",
                      help="The path of the TensorFlow object detection model")
    parser.add_option("--partition_num", type=int, dest="partition_num", default=4,
                      help="The number of partitions")
    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("TFNet Object Detection Example")

    predict(options.model_path, options.img_path, options.partition_num)
