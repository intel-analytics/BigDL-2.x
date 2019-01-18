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
from optparse import OptionParser

from zoo.common.nncontext import init_nncontext
from zoo.feature.image import ImageSet
from zoo.pipeline.inference import InferenceModel


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--base_dir", type=str, dest="base_dir",
                      help="The directory that contains frozen_inference_graph.xml and frozen_inference_graph.bin")
    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("TFNet Object Detection Example")
    images = ImageSet.read(options.img_path, resize_height=600, resize_width=600).get_image()
    model = InferenceModel()
    model.load_openvino_ir(options.base_dir + "/frozen_inference_graph.xml",
                           options.base_dir + "/frozen_inference_graph.bin")
    predictions = model.predict(images[0].reshape((1, 1, 3, 600, 600)))
    print(predictions)
