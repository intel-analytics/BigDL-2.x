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
from bigdl.zoo.models import *
from bigdl.zoo.imageclassification import *
from bigdl.nn.layer import *
from bigdl.transform.vision.image import *

JavaCreator.set_creator_class("com.intel.analytics.zoo.models.pythonapi.PythonModels")
init_engine()

parser = argparse.ArgumentParser()
parser.add_argument('modelPath', help="Path where the model is stored")
parser.add_argument('imgPath', help="Path where the images are stored")
parser.add_argument('topN', type= int,  help="Number to specify top results")

def predict(modelPath, imgPath, topN):
    print "ImageClassification prediction"
    print ("Model Path %s" % modelPath)
    print ("Image Path %s" % imgPath)
    print  ("Top N : %d" % topN)
    model = Model.loadModel(modelPath)
    imageFrame = ImageFrame.read(imgPath)
    predictor = Predictor(model)
    labelMap = predictor.label_map()
    output = predictor.predict(imageFrame)
    predicts = imageFrame.get_predict()
    for predict in predicts:
        (uri, probs) = predict
        sortedProbs = [(prob, index) for index, prob in enumerate(probs)]
        sortedProbs.sort()
        print ("Image : %s, top %d prediction result" % (uri, topN))
        for i in range(topN):
            print ("\t%s, %f" % (labelMap[sortedProbs[999- i][1]], sortedProbs[999- i][0]))



if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.modelPath, args.imgPath, args.topN)