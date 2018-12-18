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

from bigdl.nn.criterion import CrossEntropyCriterion
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.net import *
from zoo.pipeline.nnframes import *

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(sys.argv)
        print("Need parameters: <modelPath> <imagePath>")
        exit(-1)

    sc = init_nncontext("ImageFineTuningExample")

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    imageDF = NNImageReader.readImages(image_path, sc, resizeH=300, resizeW=300, image_codec=1)

    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 2.0, DoubleType())
    labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name')))
    (trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])

    # compose a pipeline that includes feature transform, pretrained model and Logistic Regression
    transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])

    full_model = Net.load_bigdl(model_path)
    # create a new model by remove layers after pool5/drop_7x7_s1
    model = full_model.new_graph(["pool5/drop_7x7_s1"])
    # freeze layers from input to pool4/3x3_s2 inclusive
    model.freeze_up_to(["pool4/3x3_s2"])

    inputNode = Input(name="input", shape=(3, 224, 224))
    inception = model.to_keras()(inputNode)
    flatten = Flatten()(inception)
    logits = Dense(2)(flatten)

    lrModel = Model(inputNode, logits)

    classifier = NNClassifier(lrModel, CrossEntropyCriterion(), transformer) \
        .setLearningRate(0.003) \
        .setBatchSize(32) \
        .setMaxEpoch(2) \
        .setFeaturesCol("image") \
        .setCachingSample(False)

    pipeline = Pipeline(stages=[classifier])

    catdogModel = pipeline.fit(trainingDF)
    predictionDF = catdogModel.transform(validationDF).cache()
    predictionDF.sample(False, 0.1).show()

    correct = predictionDF.filter("label=prediction").count()
    overall = predictionDF.count()
    accuracy = correct * 1.0 / overall

    print("Test Error = %g " % (1.0 - accuracy))
