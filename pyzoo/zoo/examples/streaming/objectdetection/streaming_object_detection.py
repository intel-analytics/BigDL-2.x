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
from pyspark.streaming import StreamingContext

from PIL import Image
from datetime import datetime
from zoo.common.nncontext import *
from zoo.models.image.objectdetection import *


def read_image_file(path):
    """
    Read image file into NDArray
    :param path: String
    :return: NDArray
    """
    print("Reading image from " + path)
    img = Image.open(path)
    nd_img = np.array(img)
    # print(nd_img.shape)
    return nd_img


def write_image_file(image, output_path):
    """
    Write image (NDArray) to file
    :param image: NDArrary
    :param output_path: String
    :return:
    """
    # The only problem of output result is that
    # image path is lost after converting to ND array.
    # So, we set current filename as timestamp with first 10 number.
    write_path = output_path + '/' \
        + datetime.now().strftime("%Y%m%d-%H%M%S") + '-'\
        + "".join([str(int(i * 100)) for i in image[:8, 0, 0]])
    print("Writing image to " + write_path)
    cv2.imwrite(write_path + '.jpg', image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="Path where the model is stored")
    parser.add_argument('--output_path', help="Path for detection results",
                        default="/tmp/zoo/output")
    parser.add_argument('--streaming_path', help="Path for streaming text",
                        default="/tmp/zoo/streaming")
    parser.add_argument("--partition_num", type=int,
                        default=1, help="The number of partitions")

    args = parser.parse_args()

    sc = init_nncontext("Streaming Object Detection Example")
    ssc = StreamingContext(sc, 3)
    lines = ssc.textFileStream(args.streaming_path)

    model = ObjectDetector.load_model(args.model)

    def predict(batch_path):
        if batch_path.getNumPartitions() == 0:
            return
        # print(batch_path.top(1))
        # Read local
        image_set = DistributedImageSet(batch_path.map(read_image_file))
        output = model.predict_image_set(image_set)
        # Save to output
        config = model.get_config()
        visualizer = Visualizer(config.label_map(), encoding="jpg")
        visualizer(output).get_image(to_chw=False)\
            .foreach(lambda x: write_image_file(x, args.output_path))

    lines.foreachRDD(predict)
    # Start the computation
    ssc.start()
    # Wait for the computation to terminate
    ssc.awaitTermination()
