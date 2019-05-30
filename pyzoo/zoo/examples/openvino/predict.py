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

import sys
import numpy as np
from os.path import join
from optparse import OptionParser

from zoo.common.nncontext import init_nncontext
from zoo.feature.image import ImageSet
from zoo.pipeline.inference import InferenceModel


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--model", type=str, dest="model_path",
                      help="Path to the TensorFlow model file")
    parser.add_option("--model_type", type=str, dest="model_type",
                      help="The type of the TensorFlow model",
                      default="faster_rcnn_resnet101_coco")

    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("OpenVINO Object Detection Inference Example")
    images = ImageSet.read(options.img_path, sc,
                           resize_height=600, resize_width=600).get_image().collect()
    input_data = np.concatenate([image.reshape((1, 1) + image.shape) for image in images], axis=0)
    model = InferenceModel()
    model.load_tf(join(options.model_path, "frozen_inference_graph.pb"),
                  backend="openvino", model_type=options.model_type,
                  ov_pipeline_config_path=join(options.model_path, "pipeline.config"))
    predictions = model.predict(input_data)
    # Print the detection result of the first image.
    print(predictions[0])
