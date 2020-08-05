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
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import col, udf
from bigdl.optim.optimizer import *
from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.net.torch_net import TorchNet
from zoo.pipeline.api.net.torch_criterion import TorchCriterion
from zoo.pipeline.nnframes import *
from zoo.pipeline.api.keras.metrics import Accuracy


# Define model with Pytorch
class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.features = torchvision.models.resnet18(pretrained=True).eval()
        self.dense1 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.features(x)
        x = F.log_softmax(self.dense1(x), dim=1)
        return x


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(sys.argv)
        print("Need parameters: <imagePath>")
        exit(-1)

    hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')

    if hadoop_conf_dir:
        num_executors = 2
        num_cores_per_executor = 4
        sc = init_spark_on_yarn(
            hadoop_conf=hadoop_conf_dir,
            conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
            num_executors=num_executors,
            executor_cores=num_cores_per_executor,
            executor_memory="8g",
            driver_memory="2g",
            driver_cores=1)
    else:
        num_executors = 1
        num_cores_per_executor = 4
        sc = init_spark_on_local(cores=4, conf={"spark.driver.memory": "10g"})

    torchnet = TorchNet.from_pytorch(CatDogModel(), [4, 3, 224, 224])

    def lossFunc(input, target):
        return nn.NLLLoss().forward(input, target.flatten().long())

    torchcriterion = TorchCriterion.from_pytorch(lossFunc, [1, 2], torch.LongTensor([1]))

    # prepare training data as Spark DataFrame
    image_path = sys.argv[1]
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=256, resizeW=256, image_codec=1)
    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 0.0, DoubleType())
    labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name'))).cache()
    (trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])

    # run training and evaluation
    featureTransformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0, 255.0, 255.0, 255.0),
         ImageMatToTensor(), ImageFeatureToTensor()])

    classifier = NNClassifier(torchnet, torchcriterion, featureTransformer) \
        .setLearningRate(0.001) \
        .setBatchSize(16) \
        .setMaxEpoch(1) \
        .setFeaturesCol("image") \
        .setCachingSample(False) \
        .setValidation(EveryEpoch(), validationDF, [Accuracy()], 16)

    catdogModel = classifier.fit(trainingDF)

    shift = udf(lambda p: p - 1, DoubleType())
    predictionDF = catdogModel.transform(validationDF) \
        .withColumn("prediction", shift(col('prediction'))).cache()
    predictionDF.sample(False, 0.1).show()

    correct = predictionDF.filter("label=prediction").count()
    overall = predictionDF.count()
    accuracy = correct * 1.0 / overall

    # expecting: accuracy > 96%
    print("Validation accuracy = %g " % accuracy)
