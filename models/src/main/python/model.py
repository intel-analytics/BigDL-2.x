import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *

if sys.version >= '3':
    long = int
    unicode = str

def toPredictResult(jRecord):
    infor = jRecord[0][0]
    clses = jRecord[1]
    credits = jRecord[2]
    clsWithCredits = []
    for i in range(len(clses)):
        clsWithCredit = ClassWithCredit(clses[i], credits[i])
        clsWithCredits.append(clsWithCredit)
    return PredictResult(infor, clsWithCredits)

class ClassWithCredit():

    def __init__(self, clsName, credit):
        self.clsName = clsName
        self.credit = credit

class PredictResult():

    def __init__(self, infor, clsWithCredits):
        self.infor = infor
        self.clsWithCredits = clsWithCredits

class Predictor(JavaValue):

    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)

    def predictLocal(self, path, topN, bigdl_type="float"):
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