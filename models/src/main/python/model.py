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

import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *

if sys.version >= '3':
    long = int
    unicode = str

def toPredictResult(jRecord):
    """
    From flatted prediction result to python prediction object
    :param jRecord: java result with (JList, JList, JList)
    :return: 
    """
    infor = jRecord[0][0]
    clses = jRecord[1]
    credits = jRecord[2]
    clsWithCredits = []
    for i in range(len(clses)):
        clsWithCredit = ClassWithCredit(clses[i], credits[i])
        clsWithCredits.append(clsWithCredit)
    return PredictResult(infor, clsWithCredits)

class ClassWithCredit():
    """
    Class representing a classification category with the possibility compare to others
    """
    def __init__(self, clsName, credit):
        self.clsName = clsName
        self.credit = credit
    def __str__(self):
        return "%s:%s" % (self.clsName, self.credit)

class PredictResult():
    """
    Class representing a classfication result with infor and a list of `ClassWithCredit` attached
    where len(clsWithCredits) is the number of top categories
    """

    def __init__(self, infor, clsWithCredits):
        self.infor = infor
        self.clsWithCredits = clsWithCredits
    def __str__(self):
        str=self.infor
        clsWithCreditsStr = ["\t".join(["", cls.__str__()]) for cls in self.clsWithCredits]
        return  "%s prediction result\n%s" % (self.infor, "\n".join([cls for cls in clsWithCreditsStr]))

class Predictor(JavaValue):
    """
    Basic class for Predicors, provide basic operations as predictLocal and predictDistributed
    """

    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)


    def predictLocal(self, path, topN, bigdl_type="float"):
        """
        Local prediction API 
        :param path: where the original image stored
        :param topN: top which you want to get related categories and corresponding credits
        :param bigdl_type: data type
        :return: `PredictResult` instance
        """
        predictOutput = callBigDlFunc(bigdl_type,
                      "predictLocal",self.value, path, topN)
        clsList = predictOutput[0]
        creditList = predictOutput[1]
        predictResults = []
        for i in range(len(clsList)):
            clsWithCredit = ClassWithCredit(clsList[i], creditList[i])
            predictResults.append(clsWithCredit)
        return predictResults

    def predictDistributed(self, rdd, topN, bigdl_type="float"):
        """
        Distributed prediction API
        :param rdd: rdd of original images stored
        :param topN: number top which you want to get related categories and corresponding credits
        :param bigdl_type: data type
        :return: RDD of `PredictResult`
        """
        predictOutput = callBigDlFunc(bigdl_type,
                                      "predictDistributed",self.value, rdd, topN)
        return predictOutput.map(toPredictResult)

class AlexnetPredictor(Predictor):
    def __init__(self, modelPath, meanPath, bigdl_type="float"):
        print "creating alexnet predictor", modelPath
        super(AlexnetPredictor, self).__init__(bigdl_type,
                                                 modelPath,
                                                 meanPath)
class InceptionV1Predictor(Predictor):
    def __init__(self, modelPath,bigdl_type="float"):
        print "creating inceptionv1 predictor", modelPath
        super(InceptionV1Predictor, self).__init__(bigdl_type,
                                               modelPath)

class ResnetPredictor(Predictor):
    def __init__(self, modelPath,bigdl_type="float"):
        print "creating resnet predictor", modelPath
        super(ResnetPredictor, self).__init__(bigdl_type,
                                                modelPath)

class VGGPredictor(Predictor):
    def __init__(self, modelPath,bigdl_type="float"):
        print "creating vgg predictor", modelPath
        super(VGGPredictor, self).__init__(bigdl_type,
                                              modelPath)

class DensenetPredictor(Predictor):
    def __init__(self, modelPath,bigdl_type="float"):
        print "creating densenet predictor", modelPath
        super(DensenetPredictor, self).__init__(bigdl_type,
                                           modelPath)

class MobilenetPredictor(Predictor):
    def __init__(self, modelPath,bigdl_type="float"):
        print "creating mobilenet predictor", modelPath
        super(MobilenetPredictor, self).__init__(bigdl_type,
                                                modelPath)

class SqueezenetPredictor(Predictor):
    def __init__(self, modelPath,bigdl_type="float"):
        print "creating squeezenet predictor", modelPath
        super(SqueezenetPredictor, self).__init__(bigdl_type,
                                                 modelPath)