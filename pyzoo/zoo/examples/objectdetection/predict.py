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


sc = init_nncontext("Object Detection Example")


from zoo.models.seq2seq import *
input_data = [np.random.randint(20, size=(1, 2, 4)),
              np.random.randint(20, size=(1, 2, 4))]
encoder = RNNEncoder.initialize("LSTM", 1, 4)
decoder = RNNDecoder.initialize("LSTM", 1, 4)
bridge = Bridge.initialize("dense", 4)
model = Seq2seq(encoder, decoder, [2, 4], [2, 4], bridge)

sent1 = np.random.randint(20, size=(1, 2, 4))
sent2 = np.random.randint(20, size=(1, 4))
result = model.infer(sent1, sent2, 3)

t = encoder.forward(sent1)



parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('img_path', help="Path where the images are stored")
parser.add_argument('output_path', help="Path to store the detection results")
parser.add_argument("--partition_num", type=int, default=1, help="The number of partitions")


def predict(model_path, img_path, output_path, partition_num):
    model = ObjectDetector.load_model(model_path)
    image_set = ImageSet.read(img_path, sc, image_codec=1, min_partitions=partition_num)
    output = model.predict_image_set(image_set)

    config = model.get_config()
    visualizer = Visualizer(config.label_map(), encoding="jpg")
    visualized = visualizer(output).get_image(to_chw=False).collect()
    for img_id in range(len(visualized)):
        cv2.imwrite(output_path + '/' + str(img_id) + '.jpg', visualized[img_id])


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.model_path, args.img_path, args.output_path, args.partition_num)
