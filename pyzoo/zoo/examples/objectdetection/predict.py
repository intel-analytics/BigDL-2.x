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

from zoo.common.nncontext import init_nncontext
from zoo.models.image.objectdetection import *

from zoo.feature.common import ChainedPreprocessing
from zoo.models.image.common.image_config import ImageConfigure
from moviepy.editor import *

sc = init_nncontext(create_spark_conf().setAppName("Object Detection Example"))

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('img_path', help="Path where the images are stored")
parser.add_argument('output_path',  help="Path to store the detection results")


def predict(model_path, img_path, output_path):
    model = ObjectDetector.load_model(model_path)
    # image_set = ImageSet.read(img_path, sc, image_codec=1)
    path = "messi_clip.mp4"
    myclip = VideoFileClip(path)

    video_rdd = sc.parallelize(myclip.iter_frames(fps=5))
    image_set = DistributedImageSet(video_rdd)

    # preprocess = ChainedPreprocessing([ImageAspectScale(600, 1), ImageChannelNormalize(122.7717, 115.9465, 102.9801),
    #                                    ImageMatToTensor(), ImInfo(), ImageSetToSample(["imageTensor", "ImInfo"])])
    # postprocess = DecodeOutput()
    preprocess = ChainedPreprocessing(
        [ImageResize(300, 300), ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageSetToSample()])
    postprocess = ScaleDetection()

    label_map = {0: '__background__', 1: 'messi'}
    config = ImageConfigure(preprocess, postprocess, 1, label_map)

    output = model.predict_image_set(image_set, config)

    # output = model.predict_image_set(image_set)

    config = model.get_config()
    visualizer = Visualizer(config.label_map(), encoding="jpg")
    visualized = visualizer(output).get_image(to_chw=False).collect()
    for img_id in range(len(visualized)):
        cv2.imwrite(output_path + '/' + str(img_id) + '.jpg', visualized[img_id])


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.model_path, args.img_path, args.output_path)
