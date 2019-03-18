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

from zoo.common.nncontext import init_nncontext
from zoo.models.image.objectdetection import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="Path where the model is stored")
    # parser.add_argument('--img_path', help="Path where the images are stored")
    parser.add_argument('--output_path', help="Path to store the detection results")
    parser.add_argument('--streaming_path', help="Path to store the streaming text")
    parser.add_argument("--partition_num", type=int, default=1, help="The number of partitions")

    args = parser.parse_args()

    sc = init_nncontext("Streaming Object Detection Example")
    ssc = StreamingContext(sc, 3)
    lines = ssc.textFileStream(args.streaming_path)

    model = ObjectDetector.load_model(args.model)

    def predict(path):
        image_set = ImageSet.read(path, sc, image_codec=1, min_partitions=args.partition_num)
        output = model.predict_image_set(image_set)

        # Save to output
        config = model.get_config()
        visualizer = Visualizer(config.label_map(), encoding="jpg")
        visualized = visualizer(output).get_image(to_chw=False).collect()
        for img_id in range(len(visualized)):
            cv2.imwrite(args.output_path + '/' + str(img_id) + '.jpg', visualized[img_id])


    lines.foreachRDD(predict)

    # Start the computation
    ssc.start()
    # Wait for the computation to terminate
    ssc.awaitTermination()
