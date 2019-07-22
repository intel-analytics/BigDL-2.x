
from optparse import OptionParser

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from zoo.common.nncontext import *
from bigdl.optim.optimizer import Adam
from zoo.feature.common import *
from zoo.feature.image import *
from zoo.pipeline.api.net.torch_net import TorchNet
from zoo.pipeline.api.net.torch_criterion import TorchCriterion
from zoo.pipeline.nnframes import *
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, DoubleType
from torchvision.models.resnet import BasicBlock
from bigdl.nn.criterion import CrossEntropyCriterion
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession


sparkConf = init_spark_conf().setAppName("test_pytorch_lenet").setMaster("local[1]").set('spark.driver.memory', '10g')
sc = init_nncontext(sparkConf)
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

resnet = torchvision.models.inception_v3(pretrained=True).eval()

# Define model with Pytorch
class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.dense1 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.log_softmax(self.dense1(x), dim=1)
        return x

def lossFunc(input, target):
    return nn.CrossEntropyLoss().forward(input, target.flatten().long())

az_resnet = TorchNet.from_pytorch(resnet, [2, 3, 224, 224])
az_catdog = TorchNet.from_pytorch(CatDogModel(), [2, 1000])
torchcriterion = TorchCriterion.from_pytorch(lossFunc=lossFunc,
                                  input_shape=[1, 2], sample_label=torch.LongTensor([1]))


labelDF = spark.read.load("/home/yuhao/PycharmProjects/pytorch_test/data/catdog.parquet")

# compose a pipeline that includes feature transform, pretrained model and Logistic Regression
transformer = ChainedPreprocessing(
    [RowToImageFeature(), ImageCenterCrop(224, 224),
     ImageChannelNormalize(0, 0, 0, 255.0, 255.0, 255.0),
     ImageChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
     ImageMatToTensor(to_RGB=True), ImageFeatureToTensor()])

preTrainedNNModel = NNModel(az_resnet, transformer) \
    .setFeaturesCol("image") \
    .setPredictionCol("embedding")

featureDF = preTrainedNNModel.transform(labelDF)

(trainingDF, validationDF) = featureDF.randomSplit([0.9, 0.1])

classifier = NNClassifier(az_catdog, torchcriterion, SeqToTensor([1000])) \
    .setLearningRate(0.001) \
    .setOptimMethod(Adam()) \
    .setBatchSize(16) \
    .setMaxEpoch(2) \
    .setFeaturesCol("embedding")

classifierModel = classifier.fit(trainingDF)

shift = udf(lambda p: p, DoubleType())
predictionDF = classifierModel.transform(validationDF).withColumn("prediction", shift(col('prediction'))).cache()
predictionDF.sample(False, 0.1).show()

correct = predictionDF.filter("label=prediction").count()
overall = predictionDF.count()
accuracy = correct * 1.0 / overall

print("Test Error = %g " % (1.0 - accuracy))
