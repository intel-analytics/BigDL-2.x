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

import re
from bigdl.util.common import *
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from pyspark import SparkConf
from pyspark.ml import Pipeline
from zoo.pipeline.api.net import *

from zoo.common.nncontext import *
from zoo.pipeline.nnframes.nn_classifier import *
from zoo.pipeline.nnframes.nn_image_reader import *
from zoo.feature.common import *
from zoo.feature.image.imagePreprocessing import *


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(sys.argv)
        print("Need parameters: <modelPath> <imagePath>")
        exit(-1)

    sparkConf = SparkConf().setAppName("ImageTransferLearningExample")
    sc = get_nncontext(sparkConf)

    model_path = sys.argv[1]
    image_path = sys.argv[2] + '/*/*'
    imageDF = NNImageReader.readImages(image_path, sc)

    getName = udf(lambda row:
                  re.search(r'(cat|dog)\.([\d]*)\.jpg', row[0], re.IGNORECASE).group(0),
                  StringType())
    getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 2.0, DoubleType())
    labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name')))
    (trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])

    # compose a pipeline that includes feature transform, pretrained model and Logistic Regression
    transformer = ChainedPreprocessing(
        [RowToImageFeature(), Resize(256, 256), CenterCrop(224, 224),
         ChannelNormalize(123.0, 117.0, 104.0), MatToTensor(), ImageFeatureToTensor()])

    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5/drop_7x7_s1"])  # create a new model by remove layers after pool5/drop_7x7_s1
    model.freeze_up_to(["pool4/3x3_s2"])  # freeze layers from input to pool4/3x3_s2 inclusive

    lrModel = Sequential().add(model).add(Reshape([1024])).add(Linear(1024, 2)).add(LogSoftMax())

    classifier = NNClassifier.create(lrModel, ClassNLLCriterion(), transformer) \
        .setLearningRate(0.003).setBatchSize(40).setMaxEpoch(1).setFeaturesCol("image")

    pipeline = Pipeline(stages=[classifier])

    catdogModel = pipeline.fit(trainingDF)
    predictionDF = catdogModel.transform(validationDF).cache()
    predictionDF.show()

    correct = predictionDF.filter("label=prediction").count()
    overall = predictionDF.count()
    accuracy = correct * 1.0 / overall

    print("Test Error = %g " % (1.0 - accuracy))