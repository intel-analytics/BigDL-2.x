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

import argparse
import cv2
from bigdl.zoo.models import *
from bigdl.zoo.objectdetection import *
from bigdl.nn.layer import *
from bigdl.transform.vision.image import *

JavaCreator.set_creator_class("com.intel.analytics.zoo.models.pythonapi.PythonModels")
sc = get_spark_context(conf=create_spark_conf())
init_engine()

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('img_path', help="Path where the images are stored")
parser.add_argument('output_path',  help="Path to store the detection results")

def predict(model_path, img_path, output_path):
    model = Model.loadModel(model_path)
    image_frame = ImageFrame.read(img_path, sc)
    predictor = Predictor(model)
    output = predictor.predict(image_frame)
    visualizer = Visualizer(predictor.label_map(), encoding = "jpg")
    visualized = visualizer(output).get_image(to_chw=False).collect()
    for img_id in range(len(visualized)):
        cv2.imwrite(output_path + str(img_id) + '.jpg', visualized[img_id])


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.model_path, args.img_path, args.output_path)