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

from optparse import OptionParser

from zoo.pipeline.inference import InferenceModel
from zoo.common.nncontext import init_nncontext
from zoo.feature.image import *
from zoo.pipeline.nnframes import *

BATCH_SIZE = 1


def predict(model_path, img_path):
    model = InferenceModel()
    model.load_openvino(model_path,
                        weight_path=model_path[:model_path.rindex(".")] + ".bin",
                        batch_size=BATCH_SIZE)
    sc = init_nncontext("OpenVINO Python SSD_Mobilenet_v1_PPN Inference Example")
    # pre-processing
    infer_transformer = ChainedPreprocessing([ImageBytesToMat(image_codec=1),
                                              ImageResize(256, 256),
                                              ImageRandomCrop(224, 224),
                                              ImageRandomPreprocessing(ImageHFlip(), 0.5),
                                              ImageChannelNormalize(123.0, 117.0, 104.0),
                                              ImageMatToTensor(format="NCHW", to_RGB=False),
                                              ImageSetToSample(input_keys=["imageTensor"], target_keys=["label"])
                                              ])
    image_set = ImageSet.read(img_path, sc).\
        transform(infer_transformer).get_image().collect()
    image_set = np.expand_dims(image_set, axis=1)

    for i in range(len(image_set) // BATCH_SIZE + 1):
        index = i * BATCH_SIZE
        # check whether out of index
        if index >= len(image_set):
            break
        batch = image_set[index]
        # put 4 images in one batch
        for j in range(index + 1, min(index + BATCH_SIZE, len(image_set))):
            batch = np.vstack((batch, image_set[j]))
        batch = np.expand_dims(batch, axis=0)
        # predict batch
        predictions = model.predict(batch)
        print(predictions[0])


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--model", type=str, dest="model_path",
                      help="OpenVINO IR Path")

    (options, args) = parser.parse_args(sys.argv)
    predict(options.model_path, options.img_path)
