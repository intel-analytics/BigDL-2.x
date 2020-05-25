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
import random
import cv2
from os.path import join
from optparse import OptionParser

from zoo.common.nncontext import init_nncontext
from zoo.feature.image import ImageSet
from zoo.pipeline.inference import InferenceModel


def random_colour_masks(image):
    colours = [[0, 255, 0],
               [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180], [250, 80, 190],
               [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(result, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    masks, boxes, pred_cls = result
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        # color mask
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        # draw box
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        # class label
        cv2.putText(img, pred_cls[i], boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--model", type=str, dest="model_path",
                      help="Path to the TensorFlow model file")

    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("OpenVINO Mask_RCNN Inference Example")
    images = ImageSet.read(options.img_path, sc,
                           resize_height=800, resize_width=1365).get_image().collect()
    input_data = np.concatenate([image.reshape((1, 1) + image.shape) for image in images], axis=0)
    model_path = options.model_path
    model = InferenceModel()
    model.load_openvino(model_path,
                        weight_path=model_path[:model_path.rindex(".")] +
                        ".bin")
    predictions = model.predict(input_data)
    # Print the detection result of the first image.
    print(predictions[0])
    print("finished...")
    sc.stop()
