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

import argparse
import cv2

from zoo.common.nncontext import get_nncontext
from zoo.feature.image.imageset import *
from zoo.models.image.objectdetection.object_detector import *

from bigdl.util.common import *
from bigdl.transform.vision.image import *
import cv2
import numpy as np
from IPython.display import Image, display
init_engine()

path = '/home/ding/Downloads/data/ob/000019.jpg'


def transform_display(transformer, image_frame):
    out = transformer(image_frame)
    cv2.imwrite('/tmp/tmp.jpg', out.get_image(to_chw=False)[0])

brightness = Brightness(0.0, 32.0)
image_set = ImageSet.read(path)
out = image_set.transform(brightness)
cv2.imwrite('/home/ding/Downloads/tmp.jpg', out.get_image(to_chw=False)[0])



sc = get_nncontext(create_spark_conf().setAppName("Object Detection Example"))

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('img_path', help="Path where the images are stored")
parser.add_argument('output_path',  help="Path to store the detection results")


def predict(model_path, img_path, output_path):
    model = ObjectDetector.load_model(model_path)
    image_set = ImageSet.read(img_path, sc)
    output = model.predict_image_set(image_set)

    config = model.get_config()
    visualizer = Visualizer(config.label_map(), encoding="jpg")
    visualized = visualizer(output).get_image(to_chw=False).collect()
    for img_id in range(len(visualized)):
        cv2.imwrite(output_path + '/' + str(img_id) + '.jpg', visualized[img_id])


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.model_path, args.img_path, args.output_path)
