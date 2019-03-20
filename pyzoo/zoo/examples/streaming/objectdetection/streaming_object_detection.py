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

from zoo.common.nncontext import *
from zoo.models.image.objectdetection import *


def read_image_file(path):
    # with open(path, "rb") as image_file:
    #     bytes_array = image_file.read()
    print("Reading image from " + path)
    return np.random.uniform(0, 1, (200, 200, 3))


def write_image_file(image, output_path):
    print("Writing image to " + output_path)
    cv2.imwrite(output_path + '/' + str(image[:10]) + '.jpg', image)


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
        print(batch_path.top(1))
        # Read local
        image_set = DistributedImageSet(batch_path.map(read_image_file))
        output = model.predict_image_set(image_set)
        # Save to output
        config = model.get_config()
        visualizer = Visualizer(config.label_map(), encoding="jpg")
        visualizer(output).get_image(to_chw=False).map(lambda x: write_image_file(x, args.output_path))


    lines.foreachRDD(predict)
    # Start the computation
    ssc.start()
    # Wait for the computation to terminate
    ssc.awaitTermination()
